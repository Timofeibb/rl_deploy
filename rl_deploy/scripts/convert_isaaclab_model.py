#!/usr/bin/env python3
"""
Convert Isaac Lab exported models to onboard deployment format.
This script renames the Isaac Lab model files and creates a config.json
to match the expected onboard deployment structure.
"""

import os
import sys
import shutil
import json
import argparse


def convert_models(isaac_model_dir, output_dir):
    """
    Convert Isaac Lab models to onboard format.
    
    Args:
        isaac_model_dir: Path to Isaac Lab exported_deploy folder containing policy.pt and depth_latest.pt
        output_dir: Output directory for converted models (will contain base_jit.pt, vision_weight.pt, config.json)
    """
    
    # Check input files exist
    policy_path = os.path.join(isaac_model_dir, "policy.pt")
    depth_path = os.path.join(isaac_model_dir, "depth_latest.pt")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy model not found: {policy_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth model not found: {depth_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy and rename policy.pt -> base_jit.pt
    base_jit_path = os.path.join(output_dir, "base_jit.pt")
    print(f"Converting {policy_path} -> {base_jit_path}")
    shutil.copy2(policy_path, base_jit_path)
    
    # Copy and rename depth_latest.pt -> vision_weight.pt
    vision_path = os.path.join(output_dir, "vision_weight.pt")
    print(f"Converting {depth_path} -> {vision_path}")
    shutil.copy2(depth_path, vision_path)
    
    # Create config.json (matches onboard format)
    config = {
        "model_type": "parkour_deploy",
        "exported_from": "isaac_lab",
        "description": "Converted from Isaac Lab exported_deploy format"
    }
    
    config_path = os.path.join(output_dir, "config.json")
    print(f"Creating config: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Files created:")
    print(f"    - base_jit.pt")
    print(f"    - vision_weight.pt")
    print(f"    - config.json")
    print(f"\nYou can now use these models with the onboard deployment code.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab models to onboard deployment format"
    )
    parser.add_argument(
        "isaac_model_dir",
        type=str,
        help="Path to Isaac Lab exported_deploy directory (contains policy.pt and depth_latest.pt)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for converted models"
    )
    
    args = parser.parse_args()
    
    try:
        convert_models(args.isaac_model_dir, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
