#!/usr/bin/env python3
"""
Quick test script to verify ROS setup and models
"""
import rospy
import os
import sys

def test_ros():
    """Test basic ROS functionality"""
    print("="*60)
    print("Testing ROS Setup")
    print("="*60)
    
    try:
        rospy.init_node('test_setup', anonymous=True)
        print("✓ ROS node initialized")
    except Exception as e:
        print(f"✗ ROS initialization failed: {e}")
        return False
    
    return True

def test_models():
    """Test if models exist"""
    print("\n" + "="*60)
    print("Testing Model Files")
    print("="*60)
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'traced')
    base_model = os.path.join(model_dir, 'base_jit.pt')
    vision_model = os.path.join(model_dir, 'vision_weight.pt')
    
    if os.path.exists(base_model):
        print(f"✓ Base model found: {base_model}")
    else:
        print(f"✗ Base model NOT found: {base_model}")
        return False
    
    if os.path.exists(vision_model):
        print(f"✓ Vision model found: {vision_model}")
    else:
        print(f"✗ Vision model NOT found: {vision_model}")
        return False
    
    return True

def test_config():
    """Test config loading"""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    try:
        # Check if we can load parameters
        rospy.set_param('/test_param', 'test')
        val = rospy.get_param('/test_param')
        print("✓ Parameter server working")
        
        # Check if config file exists
        import rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('rl_deploy')
        config_file = os.path.join(pkg_path, 'config', 'go2_parkour_config.yaml')
        
        if os.path.exists(config_file):
            print(f"✓ Config file found: {config_file}")
        else:
            print(f"✗ Config file NOT found: {config_file}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    return True

def test_topics():
    """Test expected Gazebo topics"""
    print("\n" + "="*60)
    print("Waiting for Gazebo topics...")
    print("="*60)
    
    import time
    timeout = 10
    start_time = time.time()
    
    expected_topics = [
        '/clock',
        '/gazebo/model_states'
    ]
    
    while time.time() - start_time < timeout:
        topics = rospy.get_published_topics()
        topic_names = [t[0] for t in topics]
        
        found = sum(1 for t in expected_topics if t in topic_names)
        
        if found == len(expected_topics):
            print(f"✓ Found all expected Gazebo topics")
            return True
        
        time.sleep(0.5)
    
    print(f"✗ Timeout waiting for Gazebo topics")
    print(f"  Available topics: {[t[0] for t in rospy.get_published_topics()][:10]}")
    return False

if __name__ == '__main__':
    success = True
    
    success &= test_ros()
    success &= test_models()
    success &= test_config()
    # success &= test_topics()  # Skip for now
    
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    sys.exit(0 if success else 1)
