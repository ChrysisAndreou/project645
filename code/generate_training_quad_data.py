import read_bvh
import numpy as np
from os import listdir
import os
import torch
import read_bvh_hierarchy
from code import rotation_conversions as rc # Assuming rotation_conversions.py is in the same dir or accessible

# Global skeleton definition from standard.bvh
# Assuming scripts are run from the root of 'project645' directory
standard_bvh_file_path = "train_data_bvh/standard.bvh"

# Global skeleton and non_end_bones, loaded once
if os.path.exists(standard_bvh_file_path):
    skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file_path)
    # Create an ordered list of bones that have rotations, hip first, then others from non_end_bones
    ordered_bones_with_rotations = ['hip'] + [bone for bone in non_end_bones if bone != 'hip']
else:
    skeleton, non_end_bones, ordered_bones_with_rotations = None, None, None
    print(f"ERROR: Standard BVH file {standard_bvh_file_path} not found. Quaternion processing will likely fail.")

def get_euler_convention_and_values(raw_frame_data_segment, joint_channel_order_from_skeleton):
    """Helper to extract euler angles and convention string for pytorch3d."""
    angles_deg = list(raw_frame_data_segment)
    convention_str = ""
    
    # Map BVH channel names (like 'Zrotation') to axes ('Z') for convention string
    # The order in joint_channel_order_from_skeleton is the order of application (e.g., Rz then Ry then Rx for ZYX)
    for channel_name in joint_channel_order_from_skeleton:
        if 'rotation' in channel_name.lower():
            convention_str += channel_name[0].upper() # Z, Y, X
            
    # Angles are already in the order of convention_str because they come from raw_frame_data_segment
    # which follows the joint_channel_order_from_skeleton for rotations.
    euler_angles_rad = torch.tensor(angles_deg, dtype=torch.float32) * (np.pi / 180.0)
    return euler_angles_rad, convention_str

def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if skeleton is None: # Check if skeleton loaded
        print("ERROR: Skeleton not loaded. Cannot generate quaternion training data.")
        return

    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
        print(f"Created directory: {tar_traindata_folder}")
    bvh_files = [f for f in listdir(src_bvh_folder) if f.endswith(".bvh")]
    print(f"Found {len(bvh_files)} BVH files in {src_bvh_folder}")

    for bvh_file_name in bvh_files:
        full_bvh_path = os.path.join(src_bvh_folder, bvh_file_name)
        print(f"Encoding Quaternion data from: {full_bvh_path}")
        raw_motion_frames = read_bvh.parse_frames(full_bvh_path)

        if raw_motion_frames is None or raw_motion_frames.shape[0] == 0:
            print(f"Skipping {bvh_file_name}, no motion frames found.")
            continue
        
        all_frames_quad_data = []
        for frame_idx in range(raw_motion_frames.shape[0]):
            raw_frame = raw_motion_frames[frame_idx]
            processed_frame_quad_data = []

            # Hip translation (first 3 values)
            hip_translation = raw_frame[:3]
            processed_frame_quad_data.extend(hip_translation)

            current_raw_frame_offset = 3 # Start after hip translation for rotations

            for bone_name in ordered_bones_with_rotations:
                joint_spec = skeleton[bone_name]
                joint_channels = joint_spec['channels'] # e.g., ['Xposition', ..., 'Zrotation', 'Yrotation', 'Xrotation'] for hip
                                                  # or ['Zrotation', 'Xrotation', 'Yrotation'] for others
                
                rotation_channels_for_joint = [ch for ch in joint_channels if 'rotation' in ch.lower()]
                num_rot_channels = len(rotation_channels_for_joint)

                if num_rot_channels == 3:
                    euler_values_deg = raw_frame[current_raw_frame_offset : current_raw_frame_offset + 3]
                    
                    angles_rad, convention = get_euler_convention_and_values(euler_values_deg, rotation_channels_for_joint)
                    
                    rot_matrix = rc.euler_angles_to_matrix(angles_rad.unsqueeze(0), convention) # Add batch dim
                    quaternion = rc.matrix_to_quaternion(rot_matrix).squeeze(0) # Remove batch dim
                    processed_frame_quad_data.extend(quaternion.tolist())
                
                current_raw_frame_offset += num_rot_channels
            
            all_frames_quad_data.append(processed_frame_quad_data)
        
        if all_frames_quad_data:
            target_npy_path = os.path.join(tar_traindata_folder, bvh_file_name + ".npy")
            np.save(target_npy_path, np.array(all_frames_quad_data, dtype=np.float32))
            # Get shape of one frame for logging: print(f"Shape of one quad frame: {np.array(all_frames_quad_data[0]).shape}")
            print(f"Saved Quaternion encoded data to: {target_npy_path} (frames: {len(all_frames_quad_data)}, data_dim_per_frame: {len(all_frames_quad_data[0]) if all_frames_quad_data else 0})")
        else:
            print(f"No quaternion data generated for {bvh_file_name}")

def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder, standard_bvh_for_header):
    if skeleton is None:
        print("ERROR: Skeleton not loaded. Cannot reconstruct BVH from quaternion data.")
        return

    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
        print(f"Created directory: {tar_bvh_folder}")
    npy_files = [f for f in listdir(src_train_folder) if f.endswith(".npy")]
    print(f"Found {len(npy_files)} .npy files in {src_train_folder}")

    for npy_file_name in npy_files:
        full_npy_path = os.path.join(src_train_folder, npy_file_name)
        print(f"Decoding Quaternion data from: {full_npy_path}")
        quad_motion_frames = np.load(full_npy_path)

        if quad_motion_frames is None or quad_motion_frames.shape[0] == 0:
            print(f"Skipping {npy_file_name}, no motion frames found.")
            continue

        reconstructed_euler_motion_data = []
        for frame_idx in range(quad_motion_frames.shape[0]):
            quad_frame = quad_motion_frames[frame_idx]
            bvh_format_frame = []

            # Hip translation
            hip_translation = quad_frame[:3]
            bvh_format_frame.extend(hip_translation)

            current_quad_frame_offset = 3 # Start after hip translation for quaternions

            for bone_name in ordered_bones_with_rotations:
                joint_spec = skeleton[bone_name]
                joint_channels = joint_spec['channels']
                rotation_channels_for_joint = [ch for ch in joint_channels if 'rotation' in ch.lower()]
                num_rot_channels = len(rotation_channels_for_joint)

                if num_rot_channels == 3:
                    quaternion_values = quad_frame[current_quad_frame_offset : current_quad_frame_offset + 4]
                    # Determine original Euler convention for this joint from skeleton to convert back
                    # This is the same convention used for encoding.
                    _, convention_to_reconstruct = get_euler_convention_and_values([0,0,0], rotation_channels_for_joint) # Dummy angles to get convention

                    rot_matrix = rc.quaternion_to_matrix(torch.tensor(quaternion_values, dtype=torch.float32).unsqueeze(0)) # Add batch dim
                    euler_angles_rad = rc.matrix_to_euler_angles(rot_matrix, convention_to_reconstruct).squeeze(0) # Remove batch dim
                    euler_angles_deg = (euler_angles_rad * (180.0 / np.pi)).tolist()
                    bvh_format_frame.extend(euler_angles_deg)
                
                current_quad_frame_offset += 4 # Quaternions have 4 values
            
            reconstructed_euler_motion_data.append(bvh_format_frame)

        if reconstructed_euler_motion_data:
            base_name = npy_file_name[:-4] # Remove .npy suffix
            target_bvh_path = os.path.join(tar_bvh_folder, base_name)
            read_bvh.write_frames(standard_bvh_for_header, target_bvh_path, np.array(reconstructed_euler_motion_data, dtype=np.float32))
            print(f"Saved reconstructed BVH to: {target_bvh_path}")
        else:
            print(f"No BVH data reconstructed for {npy_file_name}")

# Paths for salsa dataset
bvh_dir_path = "train_data_bvh/salsa/"
quad_enc_dir_path = "train_data_quad/salsa/"
bvh_reconstructed_dir_path = "reconstructed_bvh_data_quad/salsa/"

if skeleton: # Proceed only if skeleton loaded correctly
    print(f"Starting Quaternion data generation for salsa dataset...")
    print(f"Source BVH folder: {bvh_dir_path}")
    print(f"Target Quaternion encoded data folder: {quad_enc_dir_path}")
    print(f"Target reconstructed BVH folder (from Quaternion): {bvh_reconstructed_dir_path}")
    print(f"Using standard BVH for hierarchy: {standard_bvh_file_path}")

    # Encode data from bvh to Quaternion representation
    if not os.path.exists(bvh_dir_path):
        print(f"ERROR: Source BVH directory {bvh_dir_path} not found. Please check the path.")
    else:
        generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

    # Decode from Quaternion representation to bvh for verification
    if not os.path.exists(quad_enc_dir_path):
        print(f"INFO: Quaternion encoded data directory {quad_enc_dir_path} not found. Skipping decoding.")
    else:
        generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path, standard_bvh_file_path)

    print("Quaternion data processing script finished.")
else:
    print("Script finished early due to SKELETON NOT LOADED.")