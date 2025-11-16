#!/usr/bin/env python3
"""
RL Policy Node
Loads trained models and executes policy inference
Integrates with observation manager and FSM controller
"""

import rospy
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
from sensor_msgs.msg import JointState, Image
from robot_msgs.msg import MotorCommand
from cv_bridge import CvBridge
from threading import Lock
import cv2

# Add path to rsl_rl modules - check multiple possible locations

class RLPolicyNode:
    def __init__(self):
        rospy.init_node('rl_policy_node', anonymous=False)
        
        # Load configuration
        self.load_config()
        
        # Initialize device
        self.device = torch.device(self.config['device'])
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load models
        self.load_models()
        
        # State
        self.enabled = False
        self.global_counter = 0
        self.visual_update_interval = rospy.get_param('~depth_camera/visual_update_interval', 5)
        self.warmup_steps = 20  # Wait 20 steps (0.4s at 50Hz) to fill history before sending commands
        
        # Buffers
        self.last_depth_image = None
        self.depth_latent = torch.zeros(1, 32, device=self.device)  # 32 depth latent (NOT 34)
        self.last_proprio = None  # Store proprioception (53 dims)
        self.last_action = torch.zeros(self.config['num_dof'], device=self.device)
        
        # History buffer for proprioception
        self.n_proprio = 53
        self.n_hist_len = 10
        self.proprio_history_buf = torch.zeros(1, self.n_hist_len, self.n_proprio, device=self.device, dtype=torch.float32)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Thread lock
        self.lock = Lock()
        
        # Setup ROS interfaces
        self.setup_subscribers()
        self.setup_publishers()
        
        rospy.loginfo("RL Policy Node initialized (DISABLED)")
    
    def load_config(self):
        """Load configuration"""
        self.config = {}
        self.config['device'] = rospy.get_param('~model/device', 'cpu')
        self.device = torch.device(self.config['device'])
        
        self.config['base_model'] = rospy.get_param('~model/base_model')
        self.config['vision_model'] = rospy.get_param('~model/vision_model')
        
        self.config['n_proprio'] = rospy.get_param('~observation/n_proprio', 53)
        self.config['n_depth_latent'] = rospy.get_param('~observation/n_depth_latent', 32)
        self.config['n_priv_explicit'] = rospy.get_param('~observation/n_priv_explicit', 9)
        self.config['n_priv_latent'] = rospy.get_param('~observation/n_priv_latent', 29)
        self.config['history_len'] = rospy.get_param('~observation/history_len', 10)
        self.config['num_dof'] = rospy.get_param('~robot/num_dof', 12)
        
        # Control
        self.config['action_scale'] = rospy.get_param('~control/action_scale', 0.25)
        self.config['clip_actions'] = rospy.get_param('~normalization/clip_actions', 100.0)
        self.control_dt = rospy.get_param('~control/decimation', 4) * rospy.get_param('~control/dt', 0.005)  # 0.02s = 50Hz
        
        # PD gains for RL mode
        rl_gains = rospy.get_param('~control/rl_mode')
        self.kp = rl_gains['stiffness']
        self.kd = rl_gains['damping']
        
        # Default joint positions for action computation
        joint_default = rospy.get_param('~joint_positions/default')
        joint_names = rospy.get_param('~robot/joint_names')
        self.joint_names = joint_names
        self.default_dof_pos = torch.tensor(
            [joint_default[name] for name in joint_names],
            device=self.device, dtype=torch.float32
        )
        
        # Joint state tracking
        self.current_joint_pos = np.zeros(self.config['num_dof'])
        self.current_joint_vel = np.zeros(self.config['num_dof'])
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load base model (contains actor, estimator, history encoder)
            base_path = os.path.expanduser(self.config['base_model'])
            rospy.loginfo(f"Loading base model from: {base_path}")
            
            self.base_model = torch.jit.load(base_path, map_location=self.device)
            self.base_model.eval()
            
            # Extract individual components for potential use
            self.estimator = self.base_model.estimator.estimator
            self.hist_encoder = self.base_model.actor.history_encoder
            self.actor_backbone = self.base_model.actor.actor_backbone
            
            rospy.loginfo("✓ Base model loaded successfully")
            
            # Load vision model
            vision_path = os.path.expanduser(self.config['vision_model'])
            rospy.loginfo(f"Loading vision model from: {vision_path}")
            
            # Try loading as JIT model first (Isaac Lab format), then as state dict (onboard format)
            try:
                # Try JIT model (Isaac Lab exported format)
                vision_model = torch.jit.load(vision_path, map_location=self.device)
                if hasattr(vision_model, 'depth_encoder'):
                    self.depth_encoder = vision_model.depth_encoder
                else:
                    self.depth_encoder = vision_model
                self.depth_encoder.eval()
                rospy.loginfo("✓ Vision model loaded successfully (JIT format)")
            except Exception as e:
                # Fall back to state dict format (onboard training format)
                rospy.loginfo(f"Not a JIT model, trying state dict format: {e}")
                vision_weights = torch.load(vision_path, map_location=self.device)
                
                # Instantiate vision architecture
                from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
                
                depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
                self.depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(self.device)
                self.depth_encoder.load_state_dict(vision_weights['depth_encoder_state_dict'])
                self.depth_encoder.eval()
                rospy.loginfo("✓ Vision model loaded successfully (state dict format)")
            
            # Set all models to eval mode
            self.estimator.eval()
            self.hist_encoder.eval()
            self.actor_backbone.eval()
            
            rospy.loginfo("All models ready for inference")
            
        except Exception as e:
            import traceback
            rospy.logerr(f"Failed to load models: {e}")
            rospy.logerr(traceback.format_exc())
            rospy.signal_shutdown("Model loading failed")
    
    def setup_subscribers(self):
        """Setup ROS subscribers"""
        # Enable/disable from FSM
        self.enable_sub = rospy.Subscriber(
            '/rl/enable',
            String,
            self.enable_callback,
            queue_size=1
        )
        
        # Proprioception from observation manager
        self.proprio_sub = rospy.Subscriber(
            '/rl/proprioception',
            Float32MultiArray,
            self.proprio_callback,
            queue_size=1
        )
        
        # Depth image (from camera)
        self.depth_sub = rospy.Subscriber(
            rospy.get_param('~depth_camera/topic', '/depth_camera/depth/image_rect_raw'),
            Image,
            self.depth_callback,
            queue_size=1
        )
        
        # Joint states for computing torques
        self.joint_state_sub = rospy.Subscriber(
            rospy.get_param('~joint_states/topic', '/go2_gazebo/joint_states'),
            JointState,
            self.joint_state_callback,
            queue_size=1
        )
        
        # Note: We'll request history and depth from observation manager
    
    def setup_publishers(self):
        """Setup ROS publishers - individual motor command publishers"""
        self.joint_cmd_pubs = {}
        joint_controller_names = [
            'FR_hip_controller', 'FR_thigh_controller', 'FR_calf_controller',
            'FL_hip_controller', 'FL_thigh_controller', 'FL_calf_controller',
            'RR_hip_controller', 'RR_thigh_controller', 'RR_calf_controller',
            'RL_hip_controller', 'RL_thigh_controller', 'RL_calf_controller'
        ]
        
        for controller_name in joint_controller_names:
            topic = f'/go2_gazebo/{controller_name}/command'
            self.joint_cmd_pubs[controller_name] = rospy.Publisher(
                topic,
                MotorCommand,
                queue_size=1
            )
        
        # Last action (for observation manager)
        self.action_pub = rospy.Publisher(
            '/rl/last_action',
            Float32MultiArray,
            queue_size=1
        )
        
        # Observation request
        self.obs_request_pub = rospy.Publisher(
            '/rl/observation_request',
            String,
            queue_size=1
        )
        
        # Debug: depth image visualization
        self.debug_depth_pub = rospy.Publisher(
            '/rl/debug/depth_processed',
            Image,
            queue_size=1
        )
    
    def enable_callback(self, msg):
        """Handle enable/disable from FSM"""
        with self.lock:
            if msg.data == "enable":
                self.enabled = True
                self.global_counter = 0
                rospy.loginfo("RL Policy ENABLED")
            else:
                self.enabled = False
                rospy.loginfo("RL Policy DISABLED")
    
    def joint_state_callback(self, msg):
        """Update current joint states for PD control"""
        joint_map = {name: msg.name.index(name) for name in self.joint_names if name in msg.name}
        
        for i, name in enumerate(self.joint_names):
            if name in joint_map:
                idx = joint_map[name]
                self.current_joint_pos[i] = msg.position[idx]
                self.current_joint_vel[i] = msg.velocity[idx] if msg.velocity else 0.0
    
    def proprio_callback(self, msg):
        """Store proprioception"""
        # Observation manager publishes 123 dims, but we only need first 53 (proprio)
        with self.lock:
            if len(msg.data) >= 53:
                self.last_proprio = torch.tensor(msg.data[:53], device=self.device, dtype=torch.float32)
                # Update history buffer
                self.proprio_history_buf = torch.roll(self.proprio_history_buf, -1, dims=1)
                self.proprio_history_buf[0, -1, :] = self.last_proprio
    
    def depth_callback(self, msg):
        """Process and store depth image for encoding"""
        try:
            # Convert ROS image to numpy array
            if msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            elif msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth_image = depth_image.astype(np.float32) / 1000.0  # mm to meters
            else:
                rospy.logwarn_throttle(5.0, f"Unsupported depth encoding: {msg.encoding}")
                return
            
            # Replace NaN and Inf with max depth
            depth_image = np.nan_to_num(depth_image, nan=3.0, posinf=3.0, neginf=0.0)
            
            # Resize to 58x87 (expected by vision encoder)
            depth_resized = cv2.resize(depth_image, (87, 58), interpolation=cv2.INTER_LINEAR)
            
            # Replace any NaN created by resize
            depth_resized = np.nan_to_num(depth_resized, nan=3.0, posinf=3.0, neginf=0.0)
            
            # Normalize to [-0.5, 0.5]
            depth_max = 3.0  # meters
            depth_normalized = np.clip(depth_resized, 0.0, depth_max)
            depth_normalized = (depth_normalized / depth_max) - 0.5
            
            with self.lock:
                self.last_depth_image = depth_normalized
            
            # Publish debug visualization (convert [-0.5, 0.5] to [0, 255] for visualization)
            if self.debug_depth_pub.get_num_connections() > 0:
                depth_vis = ((depth_normalized + 0.5) * 255).astype(np.uint8)
                debug_msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding='mono8')
                debug_msg.header = msg.header
                self.debug_depth_pub.publish(debug_msg)
                
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error processing depth image: {e}")
    
    @torch.no_grad()
    def encode_depth(self, depth_image, proprio):
        """Encode depth image with proprioception"""
        try:
            # depth_image shape: (1, H, W)
            # proprio shape: (1, 53)
            depth_latent_yaw = self.depth_encoder(depth_image, proprio)
            
            # Extract just depth latent (first 32 dims, ignore yaw)
            depth_latent = depth_latent_yaw[:, :-2]  # 32 dims
            
            # Check for NaN
            if torch.isnan(depth_latent).any():
                rospy.logwarn("Depth encoding produced NaN, using last valid encoding")
                return self.depth_latent
            
            return depth_latent
            
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error encoding depth: {e}")
            return self.depth_latent
    
    @torch.no_grad()
    def construct_observation(self, proprio, depth_latent_yaw, proprio_history):
        """Construct full observation vector for policy"""
        # Extract components
        depth_latent = depth_latent_yaw[:, :-2]  # First 32 dims
        yaw = depth_latent_yaw[:, -2:] * 1.5     # Last 2 dims, scaled
        
        # Update yaw in proprioception (indices 6:8)
        proprio_updated = proprio.clone()
        proprio_updated[:, 6:8] = yaw
        
        # Get estimated linear velocity
        lin_vel_latent = self.estimator(proprio_updated)
        
        # Get history latent encoding
        activation = nn.ELU()
        priv_latent = self.hist_encoder(
            activation,
            proprio_history.view(-1, self.config['history_len'], self.config['n_proprio'])
        )
        
        # Concatenate full observation
        obs = torch.cat([
            proprio_updated,   # 53
            depth_latent,      # 32
            lin_vel_latent,    # 9
            priv_latent        # 29
        ], dim=-1)  # Total: 123
        
        return obs
    
    @torch.no_grad()
    def compute_action(self, obs):
        """Compute action from observation"""
        # Get action from actor
        action = self.actor(obs)
        
        # Clip actions
        action = torch.clamp(action, -self.config['clip_actions'], self.config['clip_actions'])
        
        return action
    
    def action_to_joint_positions(self, action):
        """Convert action to joint position targets"""
        # action is residual from default position
        # target_pos = default_pos + action * action_scale
        joint_targets = self.default_dof_pos + action[0] * self.config['action_scale']
        
        return joint_targets
    
    def publish_joint_command(self, joint_positions):
        """Publish joint command using PD control via MotorCommand"""
        # Convert desired joint positions to numpy
        joint_pos_np = joint_positions.cpu().numpy()
        
        # Publish MotorCommand to each joint controller
        joint_controller_names = [
            'FR_hip_controller', 'FR_thigh_controller', 'FR_calf_controller',
            'FL_hip_controller', 'FL_thigh_controller', 'FL_calf_controller',
            'RR_hip_controller', 'RR_thigh_controller', 'RR_calf_controller',
            'RL_hip_controller', 'RL_thigh_controller', 'RL_calf_controller'
        ]
        
        for i, controller_name in enumerate(joint_controller_names):
            cmd = MotorCommand()
            cmd.q = float(joint_pos_np[i])
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = self.kp
            cmd.kd = self.kd
            self.joint_cmd_pubs[controller_name].publish(cmd)
        
        # Also publish as last action for observation manager
        action_msg = Float32MultiArray()
        action_msg.data = ((joint_positions - self.default_dof_pos) / self.config['action_scale']).cpu().numpy().tolist()
        self.action_pub.publish(action_msg)
    
    def run(self):
        """Main control loop"""
        rospy.sleep(0.1)  # Wait for simulation time to start
        rospy.loginfo("RL Policy control loop starting...")
        
        rate = rospy.Rate(1.0 / self.control_dt)  # 50Hz
        
        while not rospy.is_shutdown():
            with self.lock:
                if self.enabled and self.last_proprio is not None:
                    # Encode depth image every N steps (10Hz vision update at 50Hz control)
                    if self.global_counter % self.visual_update_interval == 0 and self.last_depth_image is not None:
                        try:
                            # Convert numpy to tensor: (H, W) -> (1, H, W) for batch
                            depth_tensor = torch.from_numpy(self.last_depth_image).unsqueeze(0)  # (1, 58, 87)
                            depth_tensor = depth_tensor.to(device=self.device, dtype=torch.float32)
                            
                            # Check for NaN/Inf in depth image
                            if torch.isnan(depth_tensor).any() or torch.isinf(depth_tensor).any():
                                rospy.logwarn_throttle(1.0, "Depth image contains NaN/Inf values, skipping encoding")
                            else:
                                # Encode with vision model
                                proprio_for_vision = self.last_proprio.unsqueeze(0)  # (1, 53)
                                depth_latent_yaw = self.depth_encoder(depth_tensor, proprio_for_vision)  # (1, 34)
                                
                                # Check for NaN in encoded result
                                if torch.isnan(depth_latent_yaw).any() or torch.isinf(depth_latent_yaw).any():
                                    rospy.logwarn_throttle(1.0, "Depth encoding produced NaN/Inf, keeping previous latent")
                                else:
                                    # Extract just depth latent (first 32 dims, ignore yaw)
                                    self.depth_latent = depth_latent_yaw[:, :-2]  # (1, 32)
                        except Exception as e:
                            rospy.logerr_throttle(2.0, f"Error encoding depth: {e}")
                    
                    # Increment counter
                    self.global_counter += 1
                    
                    # Skip first few steps to fill history buffer and get valid depth
                    if self.global_counter < self.warmup_steps:
                        rate.sleep()
                        continue
                    
                    # Build observation tensor matching deployment format (no height scanner)
                    # Training obs: 53 (proprio) + 132 (scandot) + 9 (vel) + 29 (priv) + 530 (hist) = 753
                    # Deployment obs: 53 (proprio) + 0 (no scandot) + 9 (vel) + 29 (priv) + 530 (hist) = 621
                    # Model extracts: [0:53]=proprio, [185:194]=vel_est (filled by estimator), [-530:]=history
                    
                    obs = torch.zeros(1, 621, device=self.device)
                    obs[0, :53] = self.last_proprio  # proprio at indices 0:53
                    obs[0, -530:] = self.proprio_history_buf.view(-1)  # history at indices 91:621
                    # indices 185:194 will be filled by estimator inside model.forward()
                    
                    # Depth latent (32 dims)
                    depth_latent = self.depth_latent
                    
                    # Run full model forward (includes estimator and actor)
                    action = self.base_model.forward(obs, depth_latent).squeeze(0)  # (12,)
                    
                    # Check for NaN/Inf in action
                    if torch.isnan(action).any() or torch.isinf(action).any():
                        rospy.logerr("Policy produced NaN/Inf actions! Robot will hold position.")
                        rospy.logerr(f"Obs stats - min: {obs.min().item():.3f}, max: {obs.max().item():.3f}, mean: {obs.mean().item():.3f}")
                        rospy.logerr(f"Depth latent stats - min: {depth_latent.min().item():.3f}, max: {depth_latent.max().item():.3f}")
                        # Use zero action (hold default position)
                        action = torch.zeros_like(action)
                    
                    # Clip actions
                    action = torch.clamp(action, -self.config['clip_actions'], self.config['clip_actions'])
                    
                    # Store for next iteration
                    self.last_action = action
                    
                    # Convert action to joint targets
                    target_positions = self.default_dof_pos + action * self.config['action_scale']
                    
                    # Safety check: clamp joint positions to reasonable ranges
                    # Go2 joint limits approximately: hip [-1, 1], thigh [-3, 4], calf [-3, -0.5]
                    target_positions = torch.clamp(target_positions, -3.0, 4.0)
                    
                    # Check for NaN in target positions
                    if torch.isnan(target_positions).any() or torch.isinf(target_positions).any():
                        rospy.logerr("Target positions contain NaN/Inf! Using default position.")
                        target_positions = self.default_dof_pos
                    
                    # Publish commands
                    joint_pos_np = target_positions.detach().cpu().numpy()
                    joint_controller_names = [
                        'FR_hip_controller', 'FR_thigh_controller', 'FR_calf_controller',
                        'FL_hip_controller', 'FL_thigh_controller', 'FL_calf_controller',
                        'RR_hip_controller', 'RR_thigh_controller', 'RR_calf_controller',
                        'RL_hip_controller', 'RL_thigh_controller', 'RL_calf_controller'
                    ]
                    
                    for i, controller_name in enumerate(joint_controller_names):
                        cmd = MotorCommand()
                        cmd.q = float(joint_pos_np[i])
                        cmd.dq = 0.0
                        cmd.tau = 0.0
                        cmd.kp = self.kp
                        cmd.kd = self.kd
                        self.joint_cmd_pubs[controller_name].publish(cmd)
            
            rate.sleep()
    
    def run_inference_step(self, proprio, depth_image, proprio_history):
        """Run one step of inference"""
        with self.lock:
            if not self.enabled:
                return None
            
            # Update depth encoding periodically
            if self.global_counter % self.visual_update_interval == 0:
                if self.global_counter == 0:
                    self.last_depth_image = depth_image
                
                self.depth_latent_yaw = self.encode_depth(self.last_depth_image, proprio)
                self.last_depth_image = depth_image
            
            # Construct observation
            obs = self.construct_observation(proprio, self.depth_latent_yaw, proprio_history)
            
            # Compute action
            action = self.compute_action(obs)
            
            # Convert to joint positions
            joint_positions = self.action_to_joint_positions(action)
            
            # Publish command
            self.publish_joint_command(joint_positions)
            
            self.global_counter += 1
            
            return joint_positions


if __name__ == '__main__':
    try:
        policy_node = RLPolicyNode()
        rospy.loginfo("RL Policy Node ready. Waiting for FSM to enable...")
        policy_node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"RL Policy Node error: {e}")
        import traceback
        traceback.print_exc()
