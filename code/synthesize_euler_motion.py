import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh  # Assumed to contain write_euler_data_to_bvh
import argparse

# Default Euler frame size, matches pytorch_train_euler_aclstm.py
EULER_FRAME_CHANNELS = 132 
HIDDEN_SIZE_DEFAULT = 1024

# Added Constants
HIP_POS_SCALE_FACTOR = 0.01
HIP_X_CHANNEL, HIP_Y_CHANNEL, HIP_Z_CHANNEL = 0, 1, 2

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=EULER_FRAME_CHANNELS, hidden_size=HIDDEN_SIZE_DEFAULT, out_frame_size=EULER_FRAME_CHANNELS):
        super(acLSTM, self).__init__()
        
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size
        
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    def init_hidden(self, batch):
        device = next(self.parameters()).device # Get the model's current device
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).to(device))
        c1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).to(device))
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).to(device))
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).to(device))
        h1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).to(device))
        h2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).to(device))
        return ([h0, h1, h2], [c0, c1, c2])
    
    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h0, (vec_h[1], vec_c[1])) # Corrected: input to lstm2 is vec_h0
        vec_h2, vec_c2 = self.lstm3(vec_h1, (vec_h[2], vec_c[2])) # Corrected: input to lstm3 is vec_h1
     
        out_frame = self.decoder(vec_h2)
        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        
        return (out_frame, vec_h_new, vec_c_new)
        
    def forward(self, initial_seq_processed_lstm, generate_frames_number):
        batch = initial_seq_processed_lstm.size(0) # Corrected: size(0)
        initial_processed_len = initial_seq_processed_lstm.size(1) # Corrected: size(1)

        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq_list = [] # Use a list to collect output frames
        current_frame_for_generation = None

        # Process initial sequence (priming) to set LSTM state
        if initial_processed_len > 0:
            for i in range(initial_processed_len):
                in_frame = initial_seq_processed_lstm[:, i, :]
                (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
                if i == initial_processed_len - 1: # Keep last output of seed phase for generation
                    current_frame_for_generation = out_frame
        else: # initial_processed_len is 0 (e.g. initial_seq_len was 1)
            # Fallback: Initialize current_frame_for_generation to a zero tensor
            # This assumes out_frame_size is available and model is on a device
            device = next(self.parameters()).device
            current_frame_for_generation = torch.zeros(batch, self.out_frame_size, device=device)
            # We also need to initialize vec_h, vec_c if not done by init_hidden or if init_hidden needs batch
            # init_hidden is already called, so vec_h, vec_c should be fine.

        # Generate new frames
        for i in range(generate_frames_number):
            # The first in_frame for generation is current_frame_for_generation set above
            # For subsequent steps, in_frame is the out_frame from the previous step
            in_frame = current_frame_for_generation 
            (out_frame, vec_h, vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq_list.append(out_frame.unsqueeze(1)) # Add batch and seq_len dim
            current_frame_for_generation = out_frame # Next input is current output
    
        if not out_seq_list: # If generate_frames_number was 0
            device = next(self.parameters()).device
            return torch.zeros(batch, 0, self.out_frame_size, device=device)

        out_seq = torch.cat(out_seq_list, 1) # Only the generated frames
        return out_seq

    def calculate_loss(self, out_seq, groundtruth_seq):
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss

# numpy array initial_seq_np: batch * initial_seq_len * frame_size
# return numpy b * generate_frames_number * frame_data (raw model output: X,Z diffs, Y abs, scaled, angles in rad)
def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder, 
                 standard_bvh_ref_path, in_frame_size_param, # in_frame_size_param might be EULER_FRAME_CHANNELS
                 ground_truth_continuation_np=None, quantitative_comparison_len=0,
                 full_ground_truth_for_bvh_np=None):

    # 1. Seed Preprocessing (for model.forward)
    # initial_seq_np: Hip X, Z are scaled & relative to first frame of initial_seq_np.
    #                 Hip Y is scaled absolute. Angles are in radians.
    # We need to convert Hip X, Z to frame-to-frame differences.
    # initial_seq_len must be >= 2 for this to be meaningful.
    if initial_seq_np.shape[1] < 2:
        # This case should be handled by a check in main, but as a safeguard:
        print("Error in generate_seq: initial_seq_np must have at least 2 frames to calculate differences.")
        # Return empty or handle error appropriately
        return np.array([]) 

    dif_hip_x_scaled = initial_seq_np[:, 1:, HIP_X_CHANNEL] - initial_seq_np[:, :-1, HIP_X_CHANNEL]
    dif_hip_z_scaled = initial_seq_np[:, 1:, HIP_Z_CHANNEL] - initial_seq_np[:, :-1, HIP_Z_CHANNEL]
    
    # initial_seq_processed_for_lstm_np will have shape (batch, initial_seq_len - 1, features)
    initial_seq_processed_for_lstm_np = initial_seq_np[:, :-1].copy()
    initial_seq_processed_for_lstm_np[:, :, HIP_X_CHANNEL] = dif_hip_x_scaled
    initial_seq_processed_for_lstm_np[:, :, HIP_Z_CHANNEL] = dif_hip_z_scaled

    device = next(model.parameters()).device
    initial_seq_processed_for_lstm_torch = torch.autograd.Variable(torch.FloatTensor(initial_seq_processed_for_lstm_np)).to(device)
 
    # 2. Model Output (predict_seq_raw)
    # model.forward returns only the generated part of the sequence
    # Format: Hip X,Z are scaled differences; Hip Y is scaled absolute; angles are in radians.
    predict_seq_raw_torch = model.forward(initial_seq_processed_for_lstm_torch, generate_frames_number)
    predict_seq_raw_np = predict_seq_raw_torch.data.cpu().numpy() # Shape: (batch, generate_frames_number, features)
    
    batch_size = initial_seq_np.shape[0]
    generated_motion_all_batches_raw_output = [] # To store the raw model outputs for return
   
    for b in range(batch_size):
        current_batch_output_np = predict_seq_raw_np[b] # Raw output for this batch item
        generated_motion_all_batches_raw_output.append(current_batch_output_np.copy()) # Store raw output

        # 3. Postprocessing Generated Motion for BVH
        bvh_ready_generated_motion = current_batch_output_np.copy()

        # 3.a. Accumulate Hip X,Z Differences
        # initial_seq_np stores X,Z scaled and relative to their original dance's first frame.
        # This means initial_seq_np[b, -1, HIP_POS] is the scaled absolute position 
        # (relative to the start of the original full dance) of the last seed frame.
        # This is the value we need to start accumulating the generated differences from.
        # Ensure initial_seq_np[b] has content; initial_seq_len >= 2 is checked in main.
        last_seed_frame_scaled_abs_x = initial_seq_np[b, -1, HIP_X_CHANNEL]
        last_seed_frame_scaled_abs_z = initial_seq_np[b, -1, HIP_Z_CHANNEL]
        
        current_abs_scaled_x = last_seed_frame_scaled_abs_x
        current_abs_scaled_z = last_seed_frame_scaled_abs_z
        for frame_idx in range(bvh_ready_generated_motion.shape[0]):
            bvh_ready_generated_motion[frame_idx, HIP_X_CHANNEL] += current_abs_scaled_x
            current_abs_scaled_x = bvh_ready_generated_motion[frame_idx, HIP_X_CHANNEL]
            bvh_ready_generated_motion[frame_idx, HIP_Z_CHANNEL] += current_abs_scaled_z
            current_abs_scaled_z = bvh_ready_generated_motion[frame_idx, HIP_Z_CHANNEL]

        # 3.b. Inverse Scale Hip Positions
        bvh_ready_generated_motion[:, HIP_X_CHANNEL] /= HIP_POS_SCALE_FACTOR
        bvh_ready_generated_motion[:, HIP_Y_CHANNEL] /= HIP_POS_SCALE_FACTOR # Y was absolute scaled
        bvh_ready_generated_motion[:, HIP_Z_CHANNEL] /= HIP_POS_SCALE_FACTOR
        
        # 3.c. Convert Rotations to Degrees
        bvh_ready_generated_motion[:, HIP_Z_CHANNEL+1:] = np.rad2deg(bvh_ready_generated_motion[:, HIP_Z_CHANNEL+1:])
        
        # 3.d. Write BVH
        try:
            # Assuming read_bvh.write_frames is the correct function from your environment
            # Original script used read_bvh.write_euler_data_to_bvh, which might be a wrapper.
            # We need a function that takes (standard_ref_path, output_filepath, motion_frames_data)
            # Let's assume read_bvh.write_frames is available and correct for this processed data.
            # If write_euler_data_to_bvh was specific, its logic needs to be integrated or called appropriately.
            # For now, using a generic name that implies writing processed frames.
            if hasattr(read_bvh, 'write_frames'): 
                read_bvh.write_frames(standard_bvh_ref_path, 
                                      os.path.join(save_dance_folder, f"out{b:02d}.bvh"), 
                                      bvh_ready_generated_motion,
                                      frame_time_override=output_bvh_frame_time)
            elif hasattr(read_bvh, 'write_euler_data_to_bvh'): # Fallback to original name if it exists
                print("Using write_euler_data_to_bvh for generated motion.")
                read_bvh.write_euler_data_to_bvh(os.path.join(save_dance_folder, f"out{b:02d}.bvh"), 
                                                 bvh_ready_generated_motion, 
                                                 standard_bvh_ref_path)
            else:
                raise AttributeError("No suitable BVH writing function (write_frames or write_euler_data_to_bvh) found in read_bvh.")
        except AttributeError as e:
            print(f"ERROR during generated BVH writing: {e}")
            # Decide if to return or continue

        # 4. Postprocessing Ground Truth for BVH (full_ground_truth_for_bvh_np)
        if full_ground_truth_for_bvh_np is not None and b < full_ground_truth_for_bvh_np.shape[0]:
            gt_frames_for_bvh_raw = full_ground_truth_for_bvh_np[b]
            bvh_ready_gt_motion = gt_frames_for_bvh_raw.copy()
            
            # GT from .npy has X,Z relative to its own start, and Y absolute (all scaled), angles in radians
            # For BVH, we need absolute, unscaled positions and degrees.
            # The X,Z are already relative to *their own start*, so we don't need to re-calculate diffs and accumulate here.
            # We just need to unscale and convert angles.

            # 4.a. Inverse Scale Hip Positions
            bvh_ready_gt_motion[:, HIP_X_CHANNEL] /= HIP_POS_SCALE_FACTOR
            bvh_ready_gt_motion[:, HIP_Y_CHANNEL] /= HIP_POS_SCALE_FACTOR
            bvh_ready_gt_motion[:, HIP_Z_CHANNEL] /= HIP_POS_SCALE_FACTOR
            
            # 4.b. Convert Rotations to Degrees
            bvh_ready_gt_motion[:, HIP_Z_CHANNEL+1:] = np.rad2deg(bvh_ready_gt_motion[:, HIP_Z_CHANNEL+1:])
            
            try:
                if hasattr(read_bvh, 'write_frames'):
                    read_bvh.write_frames(standard_bvh_ref_path,
                                          os.path.join(save_dance_folder, f"gt{b:02d}.bvh"),
                                          bvh_ready_gt_motion,
                                          frame_time_override=output_bvh_frame_time)
                elif hasattr(read_bvh, 'write_euler_data_to_bvh'):
                    print("Using write_euler_data_to_bvh for GT motion.")
                    read_bvh.write_euler_data_to_bvh(os.path.join(save_dance_folder, f"gt{b:02d}.bvh"), 
                                                     bvh_ready_gt_motion, 
                                                     standard_bvh_ref_path)
                else:
                    raise AttributeError("No suitable BVH writing function for GT found in read_bvh.")
                print(f"Batch {b}: Saved ground truth BVH to " + os.path.join(save_dance_folder, f"gt{b:02d}.bvh"))
            except AttributeError as e:
                 print(f"ERROR during GT BVH writing: {e}")

        # 5. Quantitative Evaluation Data Preparation
        if ground_truth_continuation_np is not None and quantitative_comparison_len > 0 and b < ground_truth_continuation_np.shape[0]:
            # Model output part: current_batch_output_np (X,Z diffs, Y abs, scaled, rad angles)
            # This is already in the format for comparison (direct model output)
            if current_batch_output_np.shape[0] >= quantitative_comparison_len:
                generated_to_compare = current_batch_output_np[0:quantitative_comparison_len, :]
                
                # Ground Truth part: ground_truth_continuation_np[b] (X,Z rel to start, Y abs, scaled, rad angles)
                gt_raw_to_compare = ground_truth_continuation_np[b, 0:quantitative_comparison_len, :]
                
                if gt_raw_to_compare.shape[0] < quantitative_comparison_len:
                    print(f"Batch {b}: Warning - GT continuation for quant comparison is shorter ({gt_raw_to_compare.shape[0]}) than requested ({quantitative_comparison_len}). Comparing with available length.")
                    actual_quant_len = gt_raw_to_compare.shape[0]
                    generated_to_compare = current_batch_output_np[0:actual_quant_len, :]
                else:
                    actual_quant_len = quantitative_comparison_len
                
                # To make gt_raw_to_compare comparable, its hip X,Z need to be differences.
                # The first difference needs the last frame of the original *seed* sequence.
                last_seed_frame = initial_seq_np[b, -1, :] # Shape: (features,)
                
                # Create an extended GT segment: [last_seed_frame, gt_raw_to_compare_segment...]
                # Ensure gt_raw_to_compare segment used here matches actual_quant_len
                gt_extended_for_diff = np.vstack((last_seed_frame.reshape(1, -1), gt_raw_to_compare[:actual_quant_len, :]))
                
                gt_diff_hip_x_scaled = gt_extended_for_diff[1:, HIP_X_CHANNEL] - gt_extended_for_diff[:-1, HIP_X_CHANNEL]
                gt_diff_hip_z_scaled = gt_extended_for_diff[1:, HIP_Z_CHANNEL] - gt_extended_for_diff[:-1, HIP_Z_CHANNEL]
                
                gt_to_compare_processed = gt_raw_to_compare[:actual_quant_len, :].copy()
                gt_to_compare_processed[:, HIP_X_CHANNEL] = gt_diff_hip_x_scaled
                gt_to_compare_processed[:, HIP_Z_CHANNEL] = gt_diff_hip_z_scaled

                # Calculate MSE (using PyTorch for consistency, though model.calculate_loss is on GPU tensors)
                # For a simple MSE here, can use numpy or convert to torch tensors without GPU
                # mse_error = np.mean((generated_to_compare - gt_to_compare_processed)**2)
                # print(f"Batch {b}: Quantitative Evaluation MSE (first {actual_quant_len} frames, on X,Z diffs): {mse_error:.4f}")

                # --- New Quantitative Evaluation ---
                # Separate hip and rotation data
                # generated_to_compare and gt_to_compare_processed have:
                # - Hip X, Z as scaled differences
                # - Hip Y as scaled absolute
                # - Rotations as radians
                hip_pos_pred_eval = generated_to_compare[:, HIP_X_CHANNEL:HIP_Z_CHANNEL+1]
                hip_pos_gt_eval = gt_to_compare_processed[:, HIP_X_CHANNEL:HIP_Z_CHANNEL+1]

                rot_pred_rad_eval = generated_to_compare[:, HIP_Z_CHANNEL+1:]
                rot_gt_rad_eval = gt_to_compare_processed[:, HIP_Z_CHANNEL+1:]

                # MSE for hip positions (on scaled diffs for X,Z and scaled abs for Y)
                loss_hip_eval = np.mean((hip_pos_pred_eval - hip_pos_gt_eval)**2)

                # Mean angular distance for rotations (on radians)
                cos_error_eval = np.cos(rot_pred_rad_eval - rot_gt_rad_eval)
                loss_rot_ad_eval = np.mean(1.0 - cos_error_eval)
                
                print(f"Batch {b}: Quantitative Eval (first {actual_quant_len} frames) - Hip MSE: {loss_hip_eval:.4f}, Rot Ang Dist: {loss_rot_ad_eval:.4f}")
                # --- End New Quantitative Evaluation ---
            else:
                print(f"Batch {b}: Not enough generated frames ({current_batch_output_np.shape[0]}) for quantitative comparison of length {quantitative_comparison_len}.")

    # 6. Return Value of generate_seq
    # Return the direct output from model.forward() (hip X,Z diffs, Y abs, scaled, angles rad)
    return np.array(generated_motion_all_batches_raw_output)


def get_dance_len_lst(dances):
    len_lst = []
    for dance_idx, dance in enumerate(dances):
        # Proportional to length, but with a minimum to ensure all dances have a chance
        length = max(1, len(dance) // 100) # Original used 100, then 10. Using //100 for more fine-grained, min 1.
        len_lst.extend([dance_idx] * length)
    return len_lst

def load_dances(dance_folder):
    dance_files = os.listdir(dance_folder)
    dances = []
    print('Loading motion files for synthesis seed...')
    for dance_file in sorted(dance_files): # Sort for consistency
        if dance_file.endswith(".npy"):
            print("load " + dance_file)
            try:
                dance = np.load(os.path.join(dance_folder, dance_file))
                if dance.ndim == 2 and dance.shape[1] == EULER_FRAME_CHANNELS: # Basic check
                    dances.append(dance)
                    print("frame number: " + str(dance.shape[0]))
                else:
                    print(f"Skipping {dance_file}: incorrect shape or dimensions. Expected (?, {EULER_FRAME_CHANNELS})")
            except Exception as e:
                print(f"Could not load {dance_file}: {e}")
    if not dances:
        print(f"No valid .npy files (with {EULER_FRAME_CHANNELS} channels) found in {dance_folder}")
    return dances
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesize Euler angle motion using a trained acLSTM model.")
    parser.add_argument('--read_weight_path', type=str, required=True, help='Path to the trained model weights (.weight file)')
    parser.add_argument('--dances_folder', type=str, required=True, help='Path to folder containing seed motion .npy files (Euler data)')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to folder to save generated BVH files')
    parser.add_argument('--standard_bvh_reference', type=str, required=True, help='Path to the standard.bvh reference file for BVH reconstruction')
    
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Frame rate of the dance data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to generate')
    parser.add_argument('--initial_seq_len', type=int, default=15, help='Length of initial seed motion sequence (frames from .npy)')
    parser.add_argument('--generate_frames_number', type=int, default=400, help='Number of frames to synthesize')
    
    parser.add_argument('--in_frame_size', type=int, default=EULER_FRAME_CHANNELS, help='Input frame size for model')
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE_DEFAULT, help='Hidden size of LSTM layers')
    # out_frame_size typically equals in_frame_size for these types of models
    parser.add_argument('--out_frame_size', type=int, default=EULER_FRAME_CHANNELS, help='Output frame size for model')
    parser.add_argument('--quantitative_comparison_len', type=int, default=20, help='Num frames for quantitative MSE against ground truth')

    args = parser.parse_args()

    if args.initial_seq_len < 2:
        print("Error: --initial_seq_len must be at least 2 to calculate frame differences for the model seed.")
        exit()

    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)
    if not os.path.exists(args.standard_bvh_reference):
        print(f"Error: Standard BVH reference file not found at {args.standard_bvh_reference}")
        exit()
    if not os.path.exists(args.read_weight_path):
        print(f"Error: Trained weight file not found at {args.read_weight_path}")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # torch.cuda.set_device(0) # Assuming CUDA device 0 - Handled by device object

    dances_list = load_dances(args.dances_folder) 
    if not dances_list:
        print(f"Error: No dance files (.npy) found or loaded from {args.dances_folder}. Exiting.")
        exit()
        
    dance_len_proportional_idx_list = get_dance_len_lst(dances_list)
    if not dance_len_proportional_idx_list:
        print(f"Error: Could not create dance selection list. Check dance data in {args.dances_folder}.")
        exit()

    # This implies the model was trained on 30fps dynamics, and the .npy files might be at args.dance_frame_rate.
    # Let's define the model's operational FPS clearly. Assume it's 30 FPS based on training script logic.
    MODEL_TARGET_FPS = 30.0
    speed = args.dance_frame_rate / MODEL_TARGET_FPS 
    # Example: salsa npy is 60fps (args.dance_frame_rate=60). speed = 60/30 = 2.0, meaning take every 2nd frame for a 30fps seed.

    # Output BVH frame time should also be based on MODEL_TARGET_FPS
    output_bvh_frame_time = 1.0 / MODEL_TARGET_FPS

    seed_batch_list = []
    ground_truth_quantitative_batch_list = []
    full_ground_truth_for_bvh_batch_list = []
    
    # Total frames needed from the source .npy file for one item in the batch
    # Seed: args.initial_seq_len
    # GT for BVH: args.generate_frames_number (starts after seed)
    # GT for Quant: args.quantitative_comparison_len (starts after seed, typically <= generate_frames_number)
    # So, from the start of the seed, we need initial_seq_len + max(generate_frames_number, quantitative_comparison_len) frames.
    # The way it was structured before: start_id_raw + initial_seq_len*speed for seed end,
    # then generate_frames_number*speed for GT. This seems correct.
    # min_required_source_frames should ensure we can extract the longest necessary contiguous block.
    max_gt_needed_after_seed = max(args.generate_frames_number, args.quantitative_comparison_len) 
    min_required_source_frames = int((args.initial_seq_len + max_gt_needed_after_seed) * speed) + 20 # Safety margin

    for b_idx in range(args.batch_size):
        selected_dance_original_frames = None
        dance_id_for_log = -1
        attempt_count = 0
        max_attempts = len(dances_list) * 3 

        while attempt_count < max_attempts:
            # random.choice is simpler for weighted random selection if dance_len_proportional_idx_list is pre-calculated
            dance_actual_idx = random.choice(dance_len_proportional_idx_list)
            selected_dance_original_frames = dances_list[dance_actual_idx].copy()
            dance_id_for_log = dance_actual_idx # Log the actual index in dances_list
            
            if selected_dance_original_frames.shape[0] >= min_required_source_frames:
                break 
            attempt_count += 1
        
        if selected_dance_original_frames is None or selected_dance_original_frames.shape[0] < min_required_source_frames:
            print(f"Warning: Could not find a dance long enough ({min_required_source_frames} frames needed) after {max_attempts} attempts.")
            print("Trying to use the longest available dance if any meets criteria, or the first one as a last resort.")
            # Fallback: try longest, then first if it's long enough
            dances_list.sort(key=lambda d: len(d), reverse=True) # Sort by length
            if dances_list and dances_list[0].shape[0] >= min_required_source_frames:
                selected_dance_original_frames = dances_list[0].copy()
                dance_id_for_log = "longest_fallback" # Special identifier
                print(f"Using longest dance (length {selected_dance_original_frames.shape[0]}) as fallback.")
            elif dances_list: # If even longest is too short, take first if any exist
                selected_dance_original_frames = dances_list[0].copy()
                dance_id_for_log = "first_fallback"
                print(f"Warning: Using first available dance (length {selected_dance_original_frames.shape[0]}) as it's too short. Results may be poor.")
                if selected_dance_original_frames.shape[0] < int(args.initial_seq_len * speed) :
                    print(f"Error: Fallback dance is even shorter than initial_seq_len. Cannot proceed for batch item {b_idx}.")
                    continue # Skip this batch item or exit
            else: # No dances loaded
                 print(f"Error: No dances available to select from for batch item {b_idx}. Exiting.")
                 exit()

        dance_len_frames = selected_dance_original_frames.shape[0]
        
        # Adjusted to ensure enough frames for initial_seq_len + max_gt_needed_after_seed
        # max_start_id_raw calculation logic needs to be robust.
        # We need to be able to pick `args.initial_seq_len` frames for seed,
        # AND `max_gt_needed_after_seed` frames immediately following the seed portion from the original dance.
        # So, the total length of segment we need from `selected_dance_original_frames` is `args.initial_seq_len + max_gt_needed_after_seed`.
        # If speed != 1, these are counts of frames in the *original data rate*.
        total_frames_to_extract_at_speed = int((args.initial_seq_len + max_gt_needed_after_seed) * speed)
        
        if dance_len_frames < total_frames_to_extract_at_speed : # Check if dance is long enough at all
             max_start_id_raw = 0 # Not enough frames, must start at 0 and will be truncated/padded later
             print(f"Warning Batch {b_idx}: Dance {dance_id_for_log} (len {dance_len_frames}) is shorter than total needed with speed ({total_frames_to_extract_at_speed}). Will use from start.")
        else:
             max_start_id_raw = dance_len_frames - total_frames_to_extract_at_speed

        start_id_raw = random.randint(0, max_start_id_raw) if max_start_id_raw > 0 else 0


        print(f"Batch {b_idx}: Using dance {dance_id_for_log} (length {dance_len_frames} frames), seed starting raw frame {start_id_raw}")

        current_seed_seq = []
        # Extract seed sequence (args.initial_seq_len frames)
        for i in range(args.initial_seq_len):
            frame_index = int(i * speed + start_id_raw)
            safe_frame_index = min(frame_index, dance_len_frames - 1)
            current_seed_seq.append(selected_dance_original_frames[safe_frame_index])
        seed_batch_list.append(np.array(current_seed_seq))

        # Extract ground truth for BVH (args.generate_frames_number frames, starts AFTER seed portion)
        current_full_gt_for_bvh = []
        gt_bvh_start_offset_in_original = int(args.initial_seq_len * speed) # Frame index in original data where GT for BVH begins
        for i in range(args.generate_frames_number):
            frame_index = int(gt_bvh_start_offset_in_original + i * speed + start_id_raw)
            safe_frame_index = min(frame_index, dance_len_frames - 1)
            current_full_gt_for_bvh.append(selected_dance_original_frames[safe_frame_index])
        full_ground_truth_for_bvh_batch_list.append(np.array(current_full_gt_for_bvh))
        
        # Extract ground truth for quantitative comparison (args.quantitative_comparison_len frames, starts AFTER seed portion)
        if args.quantitative_comparison_len > 0:
            current_quant_gt = []
            gt_quant_start_offset_in_original = int(args.initial_seq_len * speed) # Same start as BVH GT portion
            for i in range(args.quantitative_comparison_len):
                frame_index = int(gt_quant_start_offset_in_original + i * speed + start_id_raw)
                safe_frame_index = min(frame_index, dance_len_frames - 1)
                current_quant_gt.append(selected_dance_original_frames[safe_frame_index])
            
            # This list will be padded/truncated later if necessary before np.array conversion
            ground_truth_quantitative_batch_list.append(np.array(current_quant_gt))
        elif not ground_truth_quantitative_batch_list and b_idx == 0: # only print warning once if quant_len is 0
            print(f"Quantitative comparison length is 0. No quantitative GT will be extracted.")


    if not seed_batch_list:
        print("Error: No seed sequences were prepared. Exiting.")
        exit()

    seed_batch_np = np.array(seed_batch_list)
    
    ground_truth_quantitative_np = None
    if ground_truth_quantitative_batch_list:
        # Pad/truncate shorter sequences for np.array conversion for quantitative GT
        target_len_quant = args.quantitative_comparison_len
        padded_gt_quantitative_list = []
        for s_arr in ground_truth_quantitative_batch_list:
            if s_arr.ndim == 2 and s_arr.shape[0] > 0: # Valid array
                if s_arr.shape[0] == target_len_quant:
                    padded_gt_quantitative_list.append(s_arr)
                elif s_arr.shape[0] > target_len_quant: # Truncate
                    padded_gt_quantitative_list.append(s_arr[:target_len_quant, :])
                else: # Pad
                    pad_len = target_len_quant - s_arr.shape[0]
                    padded_s = np.pad(s_arr, ((0, pad_len), (0, 0)), mode='edge')
                    padded_gt_quantitative_list.append(padded_s)
                    print(f"Warning: Padded a quantitative GT sequence from {s_arr.shape[0]} to {target_len_quant} frames.")
            elif s_arr.ndim == 1 and s_arr.shape[0] == 0 and target_len_quant > 0: # Empty array from too short source for any quant frames
                 print(f"Warning: An empty quantitative GT sequence was found. Padding with zeros to target length {target_len_quant}.")
                 padded_gt_quantitative_list.append(np.zeros((target_len_quant, seed_batch_np.shape[2])))
            # else: skip if not valid array and target_len_quant is 0

        if padded_gt_quantitative_list:
            try:
                ground_truth_quantitative_np = np.array(padded_gt_quantitative_list)
            except ValueError as e:
                print(f"Error creating numpy array for quantitative ground truth after padding: {e}")
                print("This might happen if sequences still have inconsistent shapes despite padding attempts.")
                # Fallback or error handling
                ground_truth_quantitative_np = None 
        else: # if all quant gt were empty or invalid, or quant_len was 0
            ground_truth_quantitative_np = None


    full_ground_truth_for_bvh_np = None
    if full_ground_truth_for_bvh_batch_list:
        # Pad/truncate sequences for full GT BVH to ensure consistent length (args.generate_frames_number)
        target_len_bvh = args.generate_frames_number
        processed_full_gt_bvh_list = []
        for s_arr in full_ground_truth_for_bvh_batch_list:
            if s_arr.ndim == 2 and s_arr.shape[0] > 0:
                if s_arr.shape[0] == target_len_bvh:
                    processed_full_gt_bvh_list.append(s_arr)
                elif s_arr.shape[0] > target_len_bvh: # Truncate
                    processed_full_gt_bvh_list.append(s_arr[:target_len_bvh, :])
                else: # Pad
                    pad_len = target_len_bvh - s_arr.shape[0]
                    padded_s = np.pad(s_arr, ((0, pad_len), (0, 0)), mode='edge')
                    processed_full_gt_bvh_list.append(padded_s)
                    print(f"Warning: Padded a full GT for BVH sequence from {s_arr.shape[0]} to {target_len_bvh} frames.")
            elif s_arr.ndim == 1 and s_arr.shape[0] == 0 and target_len_bvh > 0: # Empty array from too short source
                print(f"Warning: An empty full GT for BVH sequence was found. Padding with zeros to target length {target_len_bvh}.")
                processed_full_gt_bvh_list.append(np.zeros((target_len_bvh, seed_batch_np.shape[2])))
        
        if processed_full_gt_bvh_list:
            try:
                full_ground_truth_for_bvh_np = np.array(processed_full_gt_bvh_list)
            except ValueError as e:
                print(f"Error creating numpy array for full ground truth BVH after padding: {e}")
                full_ground_truth_for_bvh_np = None


    model_to_use = acLSTM(args.in_frame_size, args.hidden_size, args.out_frame_size)
    model_to_use.load_state_dict(torch.load(args.read_weight_path, map_location=device)) # Ensure weights load to correct device
    model_to_use.to(device) # Move model to device
    model_to_use.eval()

    generated_motion = generate_seq(seed_batch_np, args.generate_frames_number, model_to_use, 
                                    args.write_bvh_motion_folder, args.standard_bvh_reference, 
                                    args.in_frame_size,
                                    ground_truth_quantitative_np, args.quantitative_comparison_len,
                                    full_ground_truth_for_bvh_np)
    
    if generated_motion.size > 0: # Check if generation was successful
        print(f"Euler angle motion synthesis complete. BVH files saved to: {args.write_bvh_motion_folder}")
    else:
        print(f"Euler angle motion synthesis may have failed or was interrupted.")
