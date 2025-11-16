#!/usr/bin/env python3
"""
Observation Manager Node
Aggregates sensor data and constructs observation vector for RL policy
"""

import rospy
import numpy as np
import torch
import cv2
from collections import deque
from threading import Lock

from sensor_msgs.msg import Image, Imu, JointState
from gazebo_msgs.msg import ContactsState
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import tf.transformations as tft


class ObservationManager:
    def __init__(self):
        rospy.init_node('observation_manager', anonymous=False)
        
        # Load configuration
        self.load_config()
        
        # Initialize device
        self.device = torch.device(self.config['device'])
        
        # CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Thread locks
        self.lock = Lock()
        
        # Initialize buffers
        self.initialize_buffers()
        
        # Subscribers
        self.setup_subscribers()
        
        # Publishers
        self.obs_pub = rospy.Publisher('/rl/observation', Float32MultiArray, queue_size=1)
        self.proprio_pub = rospy.Publisher('/rl/proprioception', Float32MultiArray, queue_size=1)
        
        # State flags
        self.imu_received = False
        self.joints_received = False
        self.depth_received = False
        
        rospy.loginfo("Observation Manager initialized")
    
    def load_config(self):
        """Load configuration from parameter server"""
        self.config = {}
        self.config['device'] = rospy.get_param('~model/device', 'cpu')
        self.device = torch.device(self.config['device'])
        
        self.config['n_proprio'] = rospy.get_param('~observation/n_proprio', 53)
        self.config['history_len'] = rospy.get_param('~observation/history_len', 10)
        self.config['num_dof'] = rospy.get_param('~robot/num_dof', 12)
        
        # Joint names
        self.joint_names = rospy.get_param('~robot/joint_names')
        self.foot_names = rospy.get_param('~robot/foot_names')
        
        # Default joint positions
        default_pos = rospy.get_param('~joint_positions/default')
        self.default_dof_pos = torch.tensor([default_pos[name] for name in self.joint_names], 
                                           device=self.device, dtype=torch.float32)
        
        # Observation scales
        self.obs_scales = rospy.get_param('~observation/scales')
        
        # Depth camera config
        self.depth_config = rospy.get_param('~depth_camera')
        
        # Contact threshold
        self.contact_threshold = rospy.get_param('~contact/force_threshold', 25.0)
    
    def initialize_buffers(self):
        """Initialize all data buffers"""
        # Proprioception buffers
        self.ang_vel = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.imu_orientation = torch.zeros(2, device=self.device, dtype=torch.float32)  # roll, pitch
        self.yaw = 0.0
        
        # Joint states
        self.dof_pos = torch.zeros(self.config['num_dof'], device=self.device, dtype=torch.float32)
        self.dof_vel = torch.zeros(self.config['num_dof'], device=self.device, dtype=torch.float32)
        
        # Actions
        self.last_actions = torch.zeros(self.config['num_dof'], device=self.device, dtype=torch.float32)
        
        # Contact states (4 feet)
        self.contact_states = torch.zeros(4, device=self.device, dtype=torch.float32)
        
        # Commands (vx, vy, vyaw)
        self.commands = torch.zeros(3, device=self.device, dtype=torch.float32)
        
        # Terrain mode: 'parkour' or 'walk'
        self.terrain_mode = 'parkour'  # Default to parkour mode
        
        # History buffer
        self.proprio_history = deque(maxlen=self.config['history_len'])
        for _ in range(self.config['history_len']):
            self.proprio_history.append(torch.zeros(self.config['n_proprio'], 
                                                    device=self.device, dtype=torch.float32))
        
        # Depth image
        self.depth_image = None
        
        # Episode counter
        self.episode_length = 0
    
    def setup_subscribers(self):
        """Setup ROS subscribers"""
        # IMU
        self.imu_sub = rospy.Subscriber(
            rospy.get_param('~imu/topic', '/imu/data'),
            Imu,
            self.imu_callback,
            queue_size=1
        )
        
        # Joint states
        self.joint_sub = rospy.Subscriber(
            rospy.get_param('~joint_states/topic', '/joint_states'),
            JointState,
            self.joint_state_callback,
            queue_size=1
        )
        
        # Depth camera
        self.depth_sub = rospy.Subscriber(
            self.depth_config['topic'],
            Image,
            self.depth_callback,
            queue_size=1
        )
        
        # Contact sensors
        contact_topics = rospy.get_param('~contact/topics')
        self.contact_subs = []
        for i, (foot, topic) in enumerate(contact_topics.items()):
            sub = rospy.Subscriber(
                topic,
                ContactsState,
                lambda msg, idx=i: self.contact_callback(msg, idx),
                queue_size=1
            )
            self.contact_subs.append(sub)
        
        # Commands (from FSM controller)
        self.cmd_sub = rospy.Subscriber(
            '/rl/commands',
            Float32MultiArray,
            self.command_callback,
            queue_size=1
        )
        
        # Last action (from policy node)
        self.action_sub = rospy.Subscriber(
            '/rl/last_action',
            Float32MultiArray,
            self.action_callback,
            queue_size=1
        )
        
        # Terrain mode (from FSM)
        self.terrain_mode_sub = rospy.Subscriber(
            '/rl/terrain_mode',
            String,
            self.terrain_mode_callback,
            queue_size=1
        )
    
    def imu_callback(self, msg):
        """Process IMU data"""
        with self.lock:
            # Angular velocity (in body frame)
            self.ang_vel = torch.tensor([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ], device=self.device, dtype=torch.float32)
            
            # Orientation
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            roll, pitch, yaw = tft.euler_from_quaternion(q)
            
            self.imu_orientation = torch.tensor([roll, pitch], device=self.device, dtype=torch.float32)
            self.yaw = yaw
            
            self.imu_received = True
    
    def joint_state_callback(self, msg):
        """Process joint states"""
        with self.lock:
            # Create mapping from name to index
            joint_map = {name: msg.name.index(name) for name in self.joint_names if name in msg.name}
            
            # Extract positions and velocities in correct order
            for i, name in enumerate(self.joint_names):
                if name in joint_map:
                    idx = joint_map[name]
                    self.dof_pos[i] = msg.position[idx]
                    self.dof_vel[i] = msg.velocity[idx]
            
            self.joints_received = True
    
    def depth_callback(self, msg):
        """Process depth image"""
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
            
            # Preprocess: crop
            h, w = depth_image.shape
            top = self.depth_config['crop_top']
            bottom = h - self.depth_config['crop_bottom']
            left = self.depth_config['crop_left']
            right = w - self.depth_config['crop_right']
            
            depth_cropped = depth_image[top:bottom, left:right]
            
            # Resize
            depth_resized = cv2.resize(
                depth_cropped,
                (self.depth_config['resize_width'], self.depth_config['resize_height']),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize
            depth_min = self.depth_config['depth_min']
            depth_max = self.depth_config['depth_max']
            depth_normalized = np.clip(depth_resized, depth_min, depth_max)
            depth_normalized = (depth_normalized / depth_max) - 0.5  # [-0.5, 0.5]
            
            # Convert to tensor
            with self.lock:
                self.depth_image = torch.from_numpy(depth_normalized).to(
                    device=self.device, dtype=torch.float32
                ).unsqueeze(0)  # Add batch dimension
                self.depth_received = True
                
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Error processing depth image: {e}")
    
    def contact_callback(self, msg, foot_idx):
        """Process contact sensor data"""
        with self.lock:
            # Check if contact force exceeds threshold
            if len(msg.states) > 0:
                # Sum all contact forces
                total_force = sum([abs(state.total_wrench.force.z) for state in msg.states])
                self.contact_states[foot_idx] = 0.5 if total_force > self.contact_threshold else -0.5
            else:
                self.contact_states[foot_idx] = -0.5
    
    def command_callback(self, msg):
        """Process velocity commands"""
        with self.lock:
            if len(msg.data) >= 3:
                self.commands = torch.tensor(msg.data[:3], device=self.device, dtype=torch.float32)
    
    def action_callback(self, msg):
        """Store last action"""
        with self.lock:
            if len(msg.data) == self.config['num_dof']:
                self.last_actions = torch.tensor(msg.data, device=self.device, dtype=torch.float32)
    
    def terrain_mode_callback(self, msg):
        """Update terrain mode"""
        with self.lock:
            if msg.data in ['parkour', 'walk']:
                old_mode = self.terrain_mode
                self.terrain_mode = msg.data
                if old_mode != self.terrain_mode:
                    rospy.loginfo(f"Terrain mode changed: {old_mode} -> {self.terrain_mode}")
    
    def get_proprioception(self):
        """Construct proprioception vector (53 dims)"""
        with self.lock:
            # Angular velocity (3) - scaled
            ang_vel = self.ang_vel * self.obs_scales['ang_vel']
            
            # IMU orientation (2) - roll, pitch
            imu = self.imu_orientation
            
            # Delta yaw placeholders (3) - not used without height scanner
            yaw_info = torch.zeros(3, device=self.device, dtype=torch.float32)
            
            # Commands (3) - vx, vy, vyaw
            commands = self.commands
            
            # Terrain flags (2) - parkour mode = [1, 0], walk mode = [0, 1]
            if self.terrain_mode == 'parkour':
                terrain_flags = torch.tensor([1.0, 0.0], device=self.device, dtype=torch.float32)
            else:  # walk mode
                terrain_flags = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float32)
            
            # Debug log terrain mode periodically
            if not hasattr(self, '_last_terrain_log_time'):
                self._last_terrain_log_time = rospy.Time.now()
            if (rospy.Time.now() - self._last_terrain_log_time).to_sec() > 2.0:
                rospy.loginfo(f"Current terrain mode: {self.terrain_mode.upper()}, flags: {terrain_flags.cpu().numpy()}")
                self._last_terrain_log_time = rospy.Time.now()
            
            # Joint positions - default (12) - scaled
            dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos']
            
            # Joint velocities (12) - scaled
            dof_vel = self.dof_vel * self.obs_scales['dof_vel']
            
            # Last actions (12)
            last_actions = self.last_actions
            
            # Contact states (4)
            contact = self.contact_states
            
            # Concatenate all components
            proprio = torch.cat([
                ang_vel,         # 3
                imu,             # 2
                yaw_info,        # 3
                commands,        # 3
                terrain_flags,   # 2
                dof_pos,         # 12
                dof_vel,         # 12
                last_actions,    # 12
                contact          # 4
            ])  # Total: 53 dims
            
            return proprio
    
    def update_history(self, proprio):
        """Update proprioception history buffer"""
        self.proprio_history.append(proprio.clone())
        self.episode_length += 1
    
    def get_depth_image(self):
        """Get current depth image"""
        with self.lock:
            if self.depth_image is not None:
                return self.depth_image.clone()
            else:
                # Return zeros if no depth available
                return torch.zeros(
                    1, self.depth_config['resize_height'], self.depth_config['resize_width'],
                    device=self.device, dtype=torch.float32
                )
    
    def get_history_buffer(self):
        """Get stacked history buffer"""
        history = torch.stack(list(self.proprio_history), dim=0)
        return history.unsqueeze(0)  # Add batch dimension: (1, history_len, n_proprio)
    
    def reset(self):
        """Reset observation buffers"""
        with self.lock:
            self.episode_length = 0
            self.last_actions.zero_()
            
            # Reset history with current proprio
            proprio = self.get_proprioception()
            for i in range(self.config['history_len']):
                self.proprio_history[i] = proprio.clone()
            
            rospy.loginfo("Observation buffers reset")
    
    def is_ready(self):
        """Check if all required sensors have published data"""
        return self.imu_received and self.joints_received
    
    def publish_proprioception(self):
        """Publish current proprioception for debugging"""
        proprio = self.get_proprioception()
        msg = Float32MultiArray()
        msg.data = proprio.cpu().numpy().tolist()
        self.proprio_pub.publish(msg)


if __name__ == '__main__':
    try:
        obs_manager = ObservationManager()
        
        # Wait for sensors
        rospy.loginfo("Waiting for sensor data...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not obs_manager.is_ready():
            rate.sleep()
        
        rospy.loginfo("All sensors ready!")
        
        # Main loop - publish observations at control rate
        control_rate = rospy.Rate(50)  # 50Hz
        
        while not rospy.is_shutdown():
            # Get proprioception
            proprio = obs_manager.get_proprioception()
            
            # Update history
            obs_manager.update_history(proprio)
            
            # Publish for debugging
            obs_manager.publish_proprioception()
            
            control_rate.sleep()
            
    except rospy.ROSInterruptException:
        pass
