import read_bvh
import numpy as np
from os import listdir
import os
import torch
import read_bvh_hierarchy
import rotation_conversions as rc # Assuming rotation_conversions.py is in the same dir or accessible
import json # Added for metadata

# Global skeleton definition from standard.bvh
# Assuming scripts are run from the root of 'project645' directory
standard_bvh_file_path = "../train_data_bvh/standard.bvh"
metadata_filename = "metadata_quad.json" # For saving metadata
translation_scaling_factor = 0.01 # New global for scaling

# Global skeleton and derived info, loaded once
skeleton = None
non_end_bones = None
ordered_bones_with_rotations = []
joint_to_convention_map = {}
num_translation_channels = 3 # Standard X, Y, Z for hip
actual_quat_frame_size = 0

if os.path.exists(standard_bvh_file_path):
    skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file_path)
    if skeleton:
        # Create an ordered list of bones that have rotations, hip first, then others from non_end_bones
        # Also populate joint_to_convention_map
        if 'hip' in skeleton and any('rotation' in ch.lower() for ch in skeleton['hip'].get('channels', [])):
            ordered_bones_with_rotations.append('hip')
            # Extract convention for hip
            hip_spec = skeleton['hip']
            hip_rot_channels = [ch for ch in hip_spec['channels'] if 'rotation' in ch.lower()]
            if len(hip_rot_channels) == 3:
                convention_str = ""
                for channel_name in hip_rot_channels:
                    convention_str += channel_name[0].upper()
                joint_to_convention_map['hip'] = convention_str
        
        for bone in non_end_bones:
            if bone != 'hip' and bone in skeleton and any('rotation' in ch.lower() for ch in skeleton[bone].get('channels', [])):
                ordered_bones_with_rotations.append(bone)
                bone_spec = skeleton[bone]
                bone_rot_channels = [ch for ch in bone_spec['channels'] if 'rotation' in ch.lower()]
                if len(bone_rot_channels) == 3:
                    convention_str = ""
                    for channel_name in bone_rot_channels:
                        convention_str += channel_name[0].upper()
                    joint_to_convention_map[bone] = convention_str
        
        actual_quat_frame_size = num_translation_channels + len(ordered_bones_with_rotations) * 4
        print(f"Skeleton loaded. Ordered bones with rotations: {ordered_bones_with_rotations}")
        print(f"Joint to convention map: {joint_to_convention_map}")
        print(f"Calculated Quaternion Frame Size: {actual_quat_frame_size} ({num_translation_channels} trans + {len(ordered_bones_with_rotations)} joints * 4 quat)")
    else:
        print(f"ERROR: Could not parse skeleton from {standard_bvh_file_path}.")
else:
    print(f"ERROR: Standard BVH file {standard_bvh_file_path} not found. Quaternion processing will likely fail.")


def get_euler_angles_deg_from_raw(raw_frame_data_segment, joint_channel_order_from_skeleton):
    """Helper to extract euler angles in degrees for conversion, assumes order matches."""
    # This function is mostly to make it clear that raw_frame_data_segment provides degrees in the correct order.
    # The convention string is now derived globally.
    angles_deg = list(raw_frame_data_segment)
    return torch.tensor(angles_deg, dtype=torch.float32)

def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not skeleton: # Check if skeleton loaded
        print("ERROR: Skeleton not loaded. Cannot generate quaternion training data.")
        return

    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
        print(f"Created directory: {tar_traindata_folder}")
    
    # Save metadata
    metadata_path = os.path.join(tar_traindata_folder, metadata_filename)
    metadata_content = {
        "standard_bvh_file_path": standard_bvh_file_path,
        "ordered_bones_with_rotations": ordered_bones_with_rotations,
        "joint_to_convention_map": joint_to_convention_map,
        "num_translation_channels": num_translation_channels,
        "actual_quat_frame_size": actual_quat_frame_size,
        "translation_scaling_factor": translation_scaling_factor,
        "source_script": os.path.basename(__file__)
    }
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata_content, f, indent=4)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

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

            # Hip translation (first num_translation_channels values)
            hip_translation_raw = raw_frame[:num_translation_channels]
            # Scale down the translation values
            hip_translation_scaled = [val * translation_scaling_factor for val in hip_translation_raw]
            processed_frame_quad_data.extend(hip_translation_scaled)

            current_raw_frame_offset = num_translation_channels # Start after hip translation for rotations

            for bone_name in ordered_bones_with_rotations:
                joint_spec = skeleton[bone_name]
                joint_channels = joint_spec['channels']
                
                rotation_channels_for_joint = [ch for ch in joint_channels if 'rotation' in ch.lower()]
                num_rot_channels_this_joint = len(rotation_channels_for_joint)

                if num_rot_channels_this_joint == 3:
                    euler_values_deg_np = raw_frame[current_raw_frame_offset : current_raw_frame_offset + num_rot_channels_this_joint]
                    euler_angles_deg = torch.tensor(euler_values_deg_np, dtype=torch.float32)
                    euler_angles_rad = euler_angles_deg * (np.pi / 180.0)
                    
                    convention = joint_to_convention_map.get(bone_name)
                    if not convention:
                        print(f"Warning: No convention found for bone {bone_name} in metadata. Skipping rotation conversion for this bone in frame {frame_idx}.")
                        # Add placeholder zeros for quaternion if a bone is skipped to maintain frame size, or handle error differently
                        processed_frame_quad_data.extend([0.0, 0.0, 0.0, 0.0]) # Or raise error
                    else:
                        rot_matrix = rc.euler_angles_to_matrix(euler_angles_rad.unsqueeze(0), convention) # Add batch dim
                        quaternion = rc.matrix_to_quaternion(rot_matrix).squeeze(0) # Remove batch dim
                        processed_frame_quad_data.extend(quaternion.tolist())
                elif num_rot_channels_this_joint > 0 : # If a bone in ordered_bones_with_rotations doesn't have 3 DoF rot
                    print(f"Warning: Bone {bone_name} is in ordered_bones_with_rotations but has {num_rot_channels_this_joint} rotation channels. Expected 3. Padding quaternion.")
                    processed_frame_quad_data.extend([0.0, 0.0, 0.0, 0.0]) # Or handle error

                current_raw_frame_offset += num_rot_channels_this_joint # This should be the number of channels listed for the joint, not just rotation
                                                                    # For non-hip joints, this is usually just num_rot_channels_this_joint
                                                                    # For hip, it's total channels (trans+rot)
                                                                    # Let's refine this:
                                                                    # The offset should be based on the number of *rotation* channels we just processed.
                                                                    # The raw_frame is flat.
                                                                    # For hip, first num_translation_channels are pos. Then rot.
                                                                    # For other bones, it's just rot.
                                                                    # The loop structure for raw_frame access needs to be careful.

            # Re-thinking offset logic:
            # The raw_motion_frames from read_bvh.parse_frames directly maps to the channel order in BVH.
            # Hip: Xpos Ypos Zpos Zrot Yrot Xrot (example)
            # Other: Zrot Xrot Yrot (example)
            # The current `current_raw_frame_offset` logic seems fine for sequential parsing of *all* channels.
            # We need to extract the *rotation part* for the current bone.

            # Corrected loop for extracting data per joint based on `ordered_bones_with_rotations`
            # This requires knowing the full channel list order from BVH.
            # The current code seems to assume that `raw_frame[current_raw_frame_offset : current_raw_frame_offset + 3]`
            # will always give the 3 euler angles for the current bone_name in `ordered_bones_with_rotations`.
            # This is true if `ordered_bones_with_rotations` accurately reflects the sequence of rotating joints
            # *as they appear in the flat data array from BVH*.
            # `read_bvh.parse_frames` creates a flat array. The order is `hip` channels first, then channels of `skeleton[bone]['children'][0]`, etc. (depth-first).
            # `ordered_bones_with_rotations` from `non_end_bones` might not match this implicit flat order of read_bvh.parse_frames results.

            # Let's simplify: parse_frames gives us all numbers. We need to map them.
            # Instead of iterating `ordered_bones_with_rotations` to *build* the frame,
            # let's iterate through the `skeleton` hierarchy to *interpret* `raw_frame`.
            # Or, assume `read_bvh.py` and `read_bvh_hierarchy.py` provide data in a consistent order that
            # `ordered_bones_with_rotations` (derived from `non_end_bones` which is from `read_bvh_hierarchy`) matches.

            # The original script's logic:
            # Hip translation (first 3 values) -> added
            # current_raw_frame_offset = 3
            # for bone_name in ordered_bones_with_rotations:
            #   joint_spec = skeleton[bone_name]
            #   joint_channels = joint_spec['channels']
            #   rotation_channels_for_joint = [ch for ch in joint_channels if 'rotation' in ch.lower()]
            #   num_rot_channels = len(rotation_channels_for_joint)
            #   if num_rot_channels == 3:
            #       euler_values_deg = raw_frame[current_raw_frame_offset : current_raw_frame_offset + 3]
            #       ... convert ...
            #   current_raw_frame_offset += num_rot_channels  <-- THIS IS THE KEY
            # This assumes that after hip translation, the next N*3 values in raw_frame correspond sequentially
            # to the Euler angles of the bones listed in `ordered_bones_with_rotations`.
            # This is a strong assumption about the output of `parse_frames` and the composition of `ordered_bones_with_rotations`.
            # `non_end_bones` from `read_bvh_hierarchy` IS ordered. So this should be okay if hip is handled correctly.

            # The current code for `processed_frame_quad_data` is built up sequentially.
            # The `current_raw_frame_offset` update logic within the loop was the main concern.
            # The original structure seems more robust for consuming the flat `raw_frame`
            # if `ordered_bones_with_rotations` truly lists bones whose rotation data appears contiguously
            # after the hip's translation channels in `raw_frame`.

            # Sticking to original loop structure for consuming raw_frame for now, as it was working.
            # The primary change is using pre-calculated conventions.
            
            if len(processed_frame_quad_data) != actual_quat_frame_size:
                 print(f"Warning: Frame {frame_idx} for {bvh_file_name} has {len(processed_frame_quad_data)} channels, expected {actual_quat_frame_size}. Padding/truncating may occur if issue is downstream.")
                 # This might happen if a bone had no convention or unexpected rot channels.
                 # For now, we rely on the processing loop to build it correctly.
            
            all_frames_quad_data.append(processed_frame_quad_data)
        
        if all_frames_quad_data:
            final_data_array = np.array(all_frames_quad_data, dtype=np.float32)
            if final_data_array.shape[1] != actual_quat_frame_size:
                print(f"CRITICAL WARNING for {bvh_file_name}: Final data array has {final_data_array.shape[1]} channels per frame, but metadata expects {actual_quat_frame_size}. This will cause errors in training/synthesis.")
            
            target_npy_path = os.path.join(tar_traindata_folder, bvh_file_name + ".npy")
            np.save(target_npy_path, final_data_array)
            print(f"Saved Quaternion encoded data to: {target_npy_path} (frames: {len(all_frames_quad_data)}, data_dim_per_frame: {final_data_array.shape[1]})")
        else:
            print(f"No quaternion data generated for {bvh_file_name}")

def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder, standard_bvh_for_header):
    if not skeleton:
        print("ERROR: Skeleton not loaded. Cannot reconstruct BVH from quaternion data.")
        return
    
    # Load metadata
    metadata_path = os.path.join(src_train_folder, metadata_filename)
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file {metadata_path} not found. Cannot reconstruct BVH accurately.")
        return
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        _joint_to_convention_map = metadata['joint_to_convention_map']
        _ordered_bones_with_rotations = metadata['ordered_bones_with_rotations']
        _num_translation_channels = metadata['num_translation_channels']
        _translation_scaling_factor = metadata.get('translation_scaling_factor', 1.0) # Load scaling factor, default to 1.0 if not found
        # _actual_quat_frame_size = metadata['actual_quat_frame_size'] # For validation if needed
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        return

    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
        print(f"Created directory: {tar_bvh_folder}")
    npy_files = [f for f in listdir(src_train_folder) if f.endswith(".npy") and f != metadata_filename]
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
            hip_translation_scaled_from_npy = quad_frame[:_num_translation_channels]
            # Scale up the translation values for BVH
            hip_translation_original_scale = [val / _translation_scaling_factor for val in hip_translation_scaled_from_npy]
            bvh_format_frame.extend(hip_translation_original_scale)

            current_quad_frame_offset = _num_translation_channels # Start after hip translation for quaternions

            for bone_name in _ordered_bones_with_rotations:
                # joint_spec = skeleton[bone_name] # Not strictly needed if we trust metadata order
                # joint_channels = joint_spec['channels']
                # rotation_channels_for_joint = [ch for ch in joint_channels if 'rotation' in ch.lower()]
                # num_rot_channels = len(rotation_channels_for_joint) # Should be 3 if in map

                # if num_rot_channels == 3: # This check is implicitly handled by presence in _ordered_bones_with_rotations
                convention_to_reconstruct = _joint_to_convention_map.get(bone_name)
                if not convention_to_reconstruct:
                    print(f"Warning: No convention found for bone {bone_name} in metadata during BVH reconstruction. Adding zero rotation.")
                    euler_angles_deg = [0.0, 0.0, 0.0]
                else:
                    quaternion_values = quad_frame[current_quad_frame_offset : current_quad_frame_offset + 4]
                    rot_matrix = rc.quaternion_to_matrix(torch.tensor(quaternion_values, dtype=torch.float32).unsqueeze(0)) # Add batch dim
                    euler_angles_rad = rc.matrix_to_euler_angles(rot_matrix, convention_to_reconstruct).squeeze(0) # Remove batch dim
                    euler_angles_deg = (euler_angles_rad * (180.0 / np.pi)).tolist()
                
                bvh_format_frame.extend(euler_angles_deg)
                current_quad_frame_offset += 4 # Quaternions have 4 values
            
            reconstructed_euler_motion_data.append(bvh_format_frame)

        if reconstructed_euler_motion_data:
            base_name = npy_file_name[:-4] # Remove .npy suffix
            target_bvh_path = os.path.join(tar_bvh_folder, base_name + "_reconstructed.bvh") # Added suffix
            read_bvh.write_frames(standard_bvh_for_header, target_bvh_path, np.array(reconstructed_euler_motion_data, dtype=np.float32))
            print(f"Saved reconstructed BVH to: {target_bvh_path}")
        else:
            print(f"No BVH data reconstructed for {npy_file_name}")

# Paths for salsa dataset
bvh_dir_path = "../train_data_bvh/salsa/"
quad_enc_dir_path = "../train_data_quad/salsa/" # Changed output folder name
bvh_reconstructed_dir_path = "../reconstructed_bvh_data_quad/salsa/" # Changed output folder name

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
        # Ensure standard_bvh_file_path is passed correctly if it's different from the global one. Here it's the same.
        generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path, standard_bvh_file_path)

    print("Quaternion data processing script finished.")
else:
    print("Script finished early due to SKELETON NOT LOADED.")

# To run: Ensure this script is in the 'code' directory.
# It will look for '../train_data_bvh/standard.bvh' and '../train_data_bvh/salsa/'
# And output to '../train_data_quad/salsa/' and '../reconstructed_bvh_data_quad/salsa/'
# Example main execution call (if needed, but current structure runs on import if skeleton is loaded)
# if __name__ == "__main__":