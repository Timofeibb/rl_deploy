#!/usr/bin/env python3
"""
FSM Controller Node
Manages robot state machine: Idle -> Sit -> Stand -> RL Mode
Handles joystick input and sends appropriate joint commands
"""

import rospy
import numpy as np
from enum import Enum
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import String, Float32MultiArray
from robot_msgs.msg import MotorCommand
import threading


class FSMState(Enum):
    """Robot FSM States"""
    IDLE = 0        # Robot spawned but not controlled
    SIT = 1         # Moving to or holding sit position
    STAND = 2       # Moving to or holding stand position
    RL_MODE = 3     # RL policy control active


class FSMController:
    def __init__(self):
        rospy.init_node('fsm_controller', anonymous=False)
        
        # Load configuration
        self.load_config()
        
        # Current state
        self.state = FSMState.IDLE
        self.last_state = FSMState.IDLE
        
        # Joint position targets and current
        self.target_joint_pos = np.zeros(self.num_dof)
        self.current_joint_pos = np.zeros(self.num_dof)
        self.current_joint_vel = np.zeros(self.num_dof)
        
        # Interpolation
        self.interpolation_start = None
        self.interpolation_duration = 0.0
        self.start_joint_pos = np.zeros(self.num_dof)
        self.target_joint_pos_interp = np.zeros(self.num_dof)
        
        # Joystick state
        self.joy_msg = None
        self.last_button_state = {}
        
        # Velocity commands
        self.cmd_vel = np.zeros(3)  # [vx, vy, vyaw]
        
        # Terrain mode for RL policy
        self.terrain_mode = 'parkour'  # 'parkour' or 'walk'
        
        # Thread lock (REENTRANT to allow nested locking)
        self.lock = threading.RLock()
        
        # Setup ROS interfaces (publishers first, then subscribers to avoid callback issues)
        self.setup_publishers()
        self.setup_subscribers()
        
        # Controller manager service (for switching PD gains if needed)
        self.controller_switch_service = None
        
        rospy.loginfo(f"FSM Controller initialized in state: {self.state.name}")
    
    def load_config(self):
        """Load configuration from parameter server"""
        # Robot config
        self.num_dof = rospy.get_param('~robot/num_dof', 12)
        self.joint_names = rospy.get_param('~robot/joint_names')
        
        # Joint positions for each state
        sit_config = rospy.get_param('~joint_positions/sit')
        stand_config = rospy.get_param('~joint_positions/stand')
        default_config = rospy.get_param('~joint_positions/default')
        
        self.sit_pos = np.array([sit_config[name] for name in self.joint_names])
        self.stand_pos = np.array([stand_config[name] for name in self.joint_names])
        self.default_pos = np.array([default_config[name] for name in self.joint_names])
        
        # Control configuration
        self.control_dt = rospy.get_param('~control/control_dt', 0.02)
        
        # PD gains
        static_gains = rospy.get_param('~control/static_mode')
        self.kp_static = static_gains['stiffness']
        self.kd_static = static_gains['damping']
        
        # FSM timing
        self.sit_duration = rospy.get_param('~fsm/sit_duration', 2.0)
        self.stand_duration = rospy.get_param('~fsm/stand_duration', 2.0)
        
        # Joystick config
        self.joy_config = rospy.get_param('~joystick')
        self.button_map = self.joy_config['buttons']
        self.axis_map = self.joy_config['axes']
        self.deadband = self.joy_config['deadband']
        
        # Safety
        self.emergency_button = rospy.get_param('~safety/emergency_stop_button', 6)
    
    def setup_subscribers(self):
        """Setup ROS subscribers"""
        self.joy_sub = rospy.Subscriber(
            self.joy_config['topic'],
            Joy,
            self.joy_callback,
            queue_size=1
        )
        
        self.joint_state_sub = rospy.Subscriber(
            rospy.get_param('~joint_states/topic', '/joint_states'),
            JointState,
            self.joint_state_callback,
            queue_size=1
        )
    
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
        
        # State publisher for monitoring
        self.state_pub = rospy.Publisher(
            '/fsm/state',
            String,
            queue_size=1
        )
        
        # RL mode enable flag
        self.rl_enable_pub = rospy.Publisher(
            '/rl/enable',
            String,
            queue_size=1
        )
        
        # Velocity commands publisher
        self.cmd_vel_pub = rospy.Publisher(
            '/rl/commands',
            Float32MultiArray,
            queue_size=1
        )
        
        # Terrain mode publishers (latched so new subscribers get last value)
        self.terrain_mode_pub = rospy.Publisher(
            '/rl/terrain_mode',
            String,
            queue_size=1,
            latch=True
        )
        self.terrain_mode_pub2 = rospy.Publisher(
            '/terrain_mode',
            String,
            queue_size=1,
            latch=True
        )
    
    def joint_state_callback(self, msg):
        """Update current joint positions"""
        with self.lock:
            joint_map = {name: msg.name.index(name) for name in self.joint_names if name in msg.name}
            
            for i, name in enumerate(self.joint_names):
                if name in joint_map:
                    idx = joint_map[name]
                    self.current_joint_pos[i] = msg.position[idx]
                    self.current_joint_vel[i] = msg.velocity[idx]
    
    def joy_callback(self, msg):
        """Process joystick input"""
        self.joy_msg = msg
        
        # Detect button presses (rising edge)
        current_buttons = {
            'A': msg.buttons[self.button_map['A']] if len(msg.buttons) > self.button_map['A'] else 0,
            'B': msg.buttons[self.button_map['B']] if len(msg.buttons) > self.button_map['B'] else 0,
            'X': msg.buttons[self.button_map['X']] if len(msg.buttons) > self.button_map['X'] else 0,
            'Y': msg.buttons[self.button_map['Y']] if len(msg.buttons) > self.button_map['Y'] else 0,
            'EMERGENCY': msg.buttons[self.emergency_button] if len(msg.buttons) > self.emergency_button else 0,
        }
        
        # Handle state transitions on button press (rising edge)
        with self.lock:
            # A button: Go to SIT
            if current_buttons.get('A', False) and not self.last_button_state.get('A', False):
                if self.state in [FSMState.IDLE, FSMState.STAND, FSMState.RL_MODE]:
                    rospy.loginfo("Button A pressed: Transitioning to SIT")
                    self.transition_to_sit()
            
            # B button: Go to STAND
            elif current_buttons.get('B', False) and not self.last_button_state.get('B', False):
                if self.state in [FSMState.SIT]:
                    rospy.loginfo("Button B pressed: Transitioning to STAND")
                    self.transition_to_stand()
            
            # Y button: Enable RL MODE
            elif current_buttons.get('Y', False) and not self.last_button_state.get('Y', False):
                if self.state == FSMState.STAND:
                    rospy.loginfo("Button Y pressed: Enabling RL MODE")
                    self.transition_to_rl_mode()
            
            # X button: Toggle terrain mode (parkour <-> walk)
            elif current_buttons.get('X', False) and not self.last_button_state.get('X', False):
                rospy.loginfo("Button X pressed: Toggling terrain mode")
                self.toggle_terrain_mode()
            
            # Emergency stop
            if current_buttons.get('EMERGENCY', False) and not self.last_button_state.get('EMERGENCY', False):
                rospy.logwarn("Emergency stop pressed!")
                self.emergency_stop()
            
            self.last_button_state = current_buttons.copy()
        
                # Update velocity commands (always, for RL mode)
        # Hardcoded axis mapping for Xbox 360 controller:
        # axes[1] = left stick up/down (forward/back linear velocity)
        # axes[2] = right stick left/right (yaw rotation angular velocity)
        if len(msg.axes) >= 3:
            vyaw_raw = msg.axes[1]    # Left stick up/down - rotation (SWAPPED)
            vx_raw = msg.axes[2]      # Right stick left/right - forward/back (SWAPPED)
            
            # Apply deadband
            vx = vx_raw if abs(vx_raw) > self.deadband else 0.0
            vyaw = vyaw_raw if abs(vyaw_raw) > self.deadband else 0.0
            
            # Scale to max velocities
            cmd_vel_cfg = self.joy_config['cmd_vel']
            vx_scaled = np.clip(vx * cmd_vel_cfg['x_max'], cmd_vel_cfg['x_min'], cmd_vel_cfg['x_max'])
            vyaw_scaled = np.clip(vyaw * cmd_vel_cfg['yaw_max'], cmd_vel_cfg['yaw_min'], cmd_vel_cfg['yaw_max'])
            
            self.cmd_vel = np.array([vx_scaled, 0.0, vyaw_scaled])  # [vx, vy=0, vyaw]
            
            # Publish velocity commands for RL policy
            cmd_msg = Float32MultiArray()
            cmd_msg.data = self.cmd_vel.tolist()
            self.cmd_vel_pub.publish(cmd_msg)
    
    def transition_to_sit(self):
        """Transition to SIT state"""
        self.state = FSMState.SIT
        self.start_interpolation(self.sit_pos, self.sit_duration)
        self.rl_enable_pub.publish(String(data="disable"))
        rospy.loginfo("Transitioning to SIT")
    
    def transition_to_stand(self):
        """Transition to STAND state"""
        self.state = FSMState.STAND
        self.start_interpolation(self.stand_pos, self.stand_duration)
        
        # Disable RL
        self.rl_enable_pub.publish(String(data="disable"))
    
    def transition_to_rl_mode(self):
        """Transition to RL MODE"""
        self.state = FSMState.RL_MODE
        
        # Move to default position first (quick transition)
        self.start_interpolation(self.default_pos, 0.5)
        
        # Enable RL policy
        self.rl_enable_pub.publish(String(data="enable"))
        
        rospy.loginfo("RL MODE ACTIVE - Policy taking control")
    
    def emergency_stop(self):
        """Emergency stop - go to sit position"""
        rospy.logwarn("EMERGENCY STOP - Transitioning to SIT")
        self.transition_to_sit()
    
    def toggle_terrain_mode(self):
        """Toggle between parkour and walk modes"""
        with self.lock:
            if self.terrain_mode == 'parkour':
                self.terrain_mode = 'walk'
                rospy.loginfo("Terrain mode: WALK (conservative, flat terrain)")
            else:
                self.terrain_mode = 'parkour'
                rospy.loginfo("Terrain mode: PARKOUR (aggressive, obstacles)")
            
            # Publish terrain mode to both topics
            self.terrain_mode_pub.publish(String(data=self.terrain_mode))
            self.terrain_mode_pub2.publish(String(data=self.terrain_mode))
    
    def start_interpolation(self, target_pos, duration):
        """Start smooth interpolation to target position"""
        with self.lock:
            self.interpolation_start = rospy.Time.now()
            self.interpolation_duration = duration
            self.start_joint_pos = self.current_joint_pos.copy()
            self.target_joint_pos_interp = target_pos.copy()
    
    def get_interpolated_position(self):
        """Get current interpolated joint position"""
        if self.interpolation_start is None:
            return self.target_joint_pos_interp
        
        elapsed = (rospy.Time.now() - self.interpolation_start).to_sec()
        
        if elapsed >= self.interpolation_duration:
            # Interpolation complete
            self.interpolation_start = None
            return self.target_joint_pos_interp
        
        # Linear interpolation
        alpha = elapsed / self.interpolation_duration
        # Use smooth step for better motion
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        
        return self.start_joint_pos + alpha_smooth * (self.target_joint_pos_interp - self.start_joint_pos)
    
    def update(self):
        """Main update loop - called at control rate"""
        with self.lock:
            state = self.state
            
            # Publish current state
            self.state_pub.publish(String(data=state.name))
            
            if state in [FSMState.SIT, FSMState.STAND]:
                # Send interpolated position command via PD control
                target_pos = self.get_interpolated_position()
                
                # Publish MotorCommand to each joint controller
                joint_controller_names = [
                    'FR_hip_controller', 'FR_thigh_controller', 'FR_calf_controller',
                    'FL_hip_controller', 'FL_thigh_controller', 'FL_calf_controller',
                    'RR_hip_controller', 'RR_thigh_controller', 'RR_calf_controller',
                    'RL_hip_controller', 'RL_thigh_controller', 'RL_calf_controller'
                ]
                
                for i, controller_name in enumerate(joint_controller_names):
                    cmd = MotorCommand()
                    cmd.q = target_pos[i]
                    cmd.dq = 0.0
                    cmd.tau = 0.0
                    cmd.kp = self.kp_static
                    cmd.kd = self.kd_static
                    self.joint_cmd_pubs[controller_name].publish(cmd)
            
            elif state == FSMState.RL_MODE:
                # In RL mode, joint commands come from policy node
                # FSM just monitors and can interrupt if needed
                pass
            
            elif state == FSMState.IDLE:
                # Do nothing, wait for first command
                pass
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("FSM Controller running. Waiting for joystick input...")
        rospy.loginfo("Button mapping:")
        rospy.loginfo("  A: Sit position")
        rospy.loginfo("  B: Stand position (from sit)")
        rospy.loginfo("  X: Toggle terrain mode (parkour/walk)")
        rospy.loginfo("  Y: Enable RL mode (from stand)")
        rospy.loginfo(f"  {self.emergency_button}: Emergency stop")
        
        # Publish initial terrain mode
        self.terrain_mode_pub.publish(String(data=self.terrain_mode))
        self.terrain_mode_pub2.publish(String(data=self.terrain_mode))
        rospy.loginfo(f"Initial terrain mode: {self.terrain_mode.upper()}")
        
        # Wait for simulation time to start
        rospy.sleep(0.1)  # This will block until /clock is being published
        rospy.loginfo(f"Simulation time started. FSM ready.")
        
        rate = rospy.Rate(1.0 / self.control_dt)  # 50Hz default
        while not rospy.is_shutdown():
            try:
                self.update()
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error in FSM main loop: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())


if __name__ == '__main__':
    try:
        controller = FSMController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
