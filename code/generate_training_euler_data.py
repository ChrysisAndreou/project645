import read_bvh
import numpy as np
import os

HIP_POS_SCALE_FACTOR = 0.01

def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    """
    Extracts Euler angle data from BVH files.
    Hip X, Z are made relative to the first frame and then scaled.
    Hip Y is scaled but remains absolute relative to BVH origin.
    Euler angles are converted to radians.
    """
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
        print(f"Created directory: {tar_traindata_folder}")
    
    bvh_files = [f for f in os.listdir(src_bvh_folder) if f.endswith(".bvh")]
    print(f"Found {len(bvh_files)} BVH files in {src_bvh_folder}")
    
    for bvh_file_name in bvh_files:
        src_path = os.path.join(src_bvh_folder, bvh_file_name)
        print(f"Processing (encode): {src_path}")
        
        raw_frames = read_bvh.parse_frames(src_path)
        
        if raw_frames is None or len(raw_frames) == 0:
            print(f"Warning: No motion data found in {bvh_file_name}. Skipping.")
            continue
            
        processed_data = raw_frames.copy()

        # Normalize hip positions (first 3 channels)
        if processed_data.shape[0] > 0:
            # Store first frame's hip X and Z for relative calculation
            first_frame_hip_x = processed_data[0, 0]
            first_frame_hip_z = processed_data[0, 2]

            # Make X and Z relative to the first frame, then scale
            processed_data[:, 0] = (processed_data[:, 0] - first_frame_hip_x) * HIP_POS_SCALE_FACTOR
            processed_data[:, 2] = (processed_data[:, 2] - first_frame_hip_z) * HIP_POS_SCALE_FACTOR
            
            # Scale Y (absolute)
            processed_data[:, 1] = processed_data[:, 1] * HIP_POS_SCALE_FACTOR
        else:
            print(f"Warning: Empty motion data in {bvh_file_name} after parsing. Skipping normalization for this file.")


        # Convert rotation angles (from 4th column onwards) from degrees to radians
        processed_data[:, 3:] = np.deg2rad(processed_data[:, 3:])
        
        target_filename = os.path.splitext(bvh_file_name)[0] + ".npy"
        target_path = os.path.join(tar_traindata_folder, target_filename)
        np.save(target_path, processed_data)
        print(f"Saved normalized Euler data to {target_path} with shape {processed_data.shape}")

def generate_euler_bvh_from_traindata(src_train_folder, tar_bvh_folder, standard_bvh_ref_path):
    """
    Converts normalized Euler angle data (.npy) back to BVH format.
    Hip positions are inverse-scaled. First frame offsets are NOT added back here.
    Euler angles are converted from radians to degrees.
    """
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
        print(f"Created directory: {tar_bvh_folder}")
    
    npy_files = [f for f in os.listdir(src_train_folder) if f.endswith(".npy")]
    print(f"Found {len(npy_files)} NPY files in {src_train_folder}")
    
    for npy_file_name in npy_files:
        src_path = os.path.join(src_train_folder, npy_file_name)
        print(f"Processing (decode): {src_path}")
        
        normalized_euler_data_rad = np.load(src_path)
        
        if normalized_euler_data_rad is None or len(normalized_euler_data_rad) == 0:
            print(f"Warning: No motion data found in {npy_file_name}. Skipping.")
            continue
            
        bvh_motion_data = normalized_euler_data_rad.copy()

        # Inverse scale hip positions (first 3 channels)
        # Note: This does not add back the first_frame_hip_x/z offsets here.
        # That level of reconstruction is more for the synthesis script.
        if bvh_motion_data.shape[0] > 0:
            bvh_motion_data[:, 0] = bvh_motion_data[:, 0] / HIP_POS_SCALE_FACTOR
            bvh_motion_data[:, 1] = bvh_motion_data[:, 1] / HIP_POS_SCALE_FACTOR
            bvh_motion_data[:, 2] = bvh_motion_data[:, 2] / HIP_POS_SCALE_FACTOR
        
        # Convert rotation angles (from 4th column onwards) from radians back to degrees
        bvh_motion_data[:, 3:] = np.rad2deg(bvh_motion_data[:, 3:])
        
        target_bvh_filename = os.path.splitext(npy_file_name)[0] + ".bvh"
        target_path = os.path.join(tar_bvh_folder, target_bvh_filename)
        
        read_bvh.write_frames(standard_bvh_ref_path, target_path, bvh_motion_data)
        print(f"Saved reconstructed BVH file to {target_path} (hip positions inverse-scaled only)")

if __name__ == "__main__":
    # Define paths - assuming the script is run from project645/code/
    base_bvh_dir = "../train_data_bvh/"
    data_type = "salsa" # Choose 'salsa', 'indian', or 'martial'
    
    src_bvh_folder = os.path.join(base_bvh_dir, data_type)
    # Standard BVH file for hierarchy reference when writing BVH
    standard_bvh_ref_path = os.path.join(base_bvh_dir, "standard.bvh") 

    euler_traindata_folder = f"../train_data_euler/{data_type}/"
    reconstructed_bvh_folder = f"../reconstructed_bvh_data_euler/{data_type}/"
    
    print(f"--- Euler Data Generation (with Hip Position Normalization) ---")
    print(f"Source BVH: {src_bvh_folder}")
    print(f"Target .npy (Normalized Euler): {euler_traindata_folder}")
    print(f"Target reconstructed BVH: {reconstructed_bvh_folder}")
    print(f"Standard BVH for reconstruction: {standard_bvh_ref_path}")
    print(f"Hip Position Scale Factor: {HIP_POS_SCALE_FACTOR}")

    if not os.path.exists(standard_bvh_ref_path):
        print(f"ERROR: Standard BVH file not found at {standard_bvh_ref_path}. This file is required for BVH reconstruction.")
    else:
        print("\nStep 1: Converting BVH files to Normalized Euler representation (.npy)...")
        generate_euler_traindata_from_bvh(src_bvh_folder, euler_traindata_folder)
        
        print("\nStep 2: Converting Normalized Euler .npy data back to BVH for verification (hip inverse-scaled)...")
        generate_euler_bvh_from_traindata(euler_traindata_folder, reconstructed_bvh_folder, standard_bvh_ref_path)
        
        print("\nNormalized Euler data processing completed.")
