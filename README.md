# RL Deploy - Go2 Parkour Deployment for ROS Noetic Gazebo

Deploy RL parkour policies for Go2 quadruped robot in Gazebo simulation based on Extreme-Parkour-Onboard code (https://github.com/change-every/Extreme-Parkour-Onboard)

## Overview

This package provides a complete deployment system for running trained RL policies on the Go2 quadruped robot in Gazebo. It includes:

- **FSM Controller**: Finite State Machine for managing robot states (Idle → Sit → Stand → RL Mode)
- **Observation Manager**: Aggregates sensor data and constructs observation vectors
- **RL Policy Node**: Executes trained policy inference with vision and proprioception
- **Joystick Interface**: Control robot states and velocity commands

### Dependencies

```bash
# ROS Noetic
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-gazebo-ros-pkgs
sudo apt install ros-noetic-controller-manager
sudo apt install ros-noetic-joint-state-controller
sudo apt install ros-noetic-position-controllers
sudo apt install ros-noetic-joy

# Python packages
pip3 install torch torchvision
pip3 install opencv-python
pip3 install numpy
```

### Trained Models

Place your trained models in `~/go2_parkour/traced/`:
- `base_jit.pt` - Base policy (actor, estimator, history encoder)
- `vision_weight.pt` - Vision encoder weights

## Usage

### 1. Launch Complete System

```bash
roslaunch rl_deploy complete_system.launch world:=parkour
```

This launches:
- Gazebo with Go2 robot
- All deployment nodes
- Joystick interface
- RViz (optional)

### 2. FSM Control Flow

#### State Machine
```
    IDLE (Robot spawned)
      │
      │ Press A
      ▼
    SIT (Sitting position)
      │
      │ Press B
      ▼
    STAND (Standing position)
      │
      │ Press Y
      ▼
    RL_MODE (Policy control active)
```

## Configuration

Edit `config/go2_parkour_config.yaml` to customize:

### Joint Positions
- `joint_positions/sit`: Sitting pose
- `joint_positions/stand`: Standing pose
- `joint_positions/default`: RL mode neutral pose

### Control Parameters
```yaml
control:
  static_mode:
    stiffness: 40.0  # PD gains for sit/stand
    damping: 2.0
  rl_mode:
    stiffness: 40.0  # PD gains for RL control
    damping: 1.0
  action_scale: 0.25
  control_dt: 0.02  # 50Hz
```

### Observation Configuration
```yaml
observation:
  n_proprio: 53
  n_depth_latent: 32
  history_len: 10
  scales:
    lin_vel: 2.0
    ang_vel: 0.25
    dof_pos: 1.0
    dof_vel: 0.05
```

### Depth Camera
```yaml
depth_camera:
  crop_top: 60
  crop_bottom: 100
  crop_left: 80
  crop_right: 36
  resize_width: 87
  resize_height: 58
  visual_update_interval: 5  # 10Hz
```

## Observation Space

The system constructs a **123-dimensional** observation vector matching the trained policy:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Angular velocity | 3 | Body frame gyroscope |
| IMU orientation | 2 | Roll, pitch |
| Yaw info | 3 | Delta yaw (from vision) |
| Commands | 3 | vx, vy, vyaw from joystick |
| Terrain flags | 2 | Parkour/walk mode |
| Joint positions | 12 | Relative to default |
| Joint velocities | 12 | Scaled |
| Last actions | 12 | Previous motor commands |
| Contact states | 4 | Foot contact flags |
| **Depth latent** | **32** | **Vision-encoded features** |
| **Estimated lin vel** | **9** | **From estimator network** |
| **History latent** | **29** | **From history encoder** |
| **Total** | **123** | |

**Note**: No height scanner is used (matching real hardware deployment).

## Topics

### Published
- `/go2_joint_controller/command` - Joint position commands
- `/rl/commands` - Velocity commands (vx, vy, vyaw)
- `/rl/proprioception` - Current proprioception vector
- `/rl/last_action` - Last action for observation
- `/fsm/state` - Current FSM state

### Subscribed
- `/joy` - Joystick input
- `/imu/data` - IMU data
- `/joint_states` - Joint positions/velocities
- `/camera/depth/image_raw` - Depth camera
- `/FR_foot_contact`, `/FL_foot_contact`, etc. - Contact sensors

Based on the Isaac Lab extreme parkour training framework and real hardware deployment code.
