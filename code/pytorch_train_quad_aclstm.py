import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import argparse
import json # Added for metadata

# Import model and necessary functions from synthesize_quad_motion_old
from synthesize_quad_motion import acLSTM, load_full_dance_data, generate_and_evaluate_seq, get_dance_len_lst
import read_bvh # For write_frames in the training loop if not fully relying on synthesize script
import read_bvh_hierarchy # For loading skeleton if needed directly
import rotation_conversions as rc # Added for rotation conversions
import torch.nn.functional as F # Added for F.normalize if needed by copied function

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU) device for pytorch_train_quad_aclstm_old.py.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for pytorch_train_quad_aclstm_old.py.")
else:
    device = torch.device("cpu")
    print("Using CPU device for pytorch_train_quad_aclstm_old.py.")
# --- End Device Setup ---

# Global for metadata content, loaded in main
METADATA_QUAD = None

# Constants for the Auto-Conditioned LSTM (can be overridden by args or metadata)
Condition_num=5         # number of auto-conditioned steps
Groundtruth_num=5       # number of groundtruth steps

# --- Helper function adapted from synthesize_quad_motion.py ---
# This function converts a sequence of quaternion frames to Euler angle frames suitable for BVH.
def _convert_quad_seq_to_euler_bvh_for_train_script(quad_sequence_np, skeleton_info_ref, ordered_bones, joint_conv_map, num_trans_ch, translation_scale_factor, rc_module):
    num_frames = quad_sequence_np.shape[0]
    if not ordered_bones:
        # print("Warning (_convert_quad_seq_to_euler_bvh_for_train_script): No ordered_bones for rotation.")
        euler_frame_size = num_trans_ch
    else:
        euler_frame_size = num_trans_ch + len(ordered_bones) * 3
    
    euler_sequence_list = []

    for frame_idx in range(num_frames):
        quad_frame = quad_sequence_np[frame_idx]
        euler_frame_data = []
        if num_trans_ch > 0:
            # Translation data in quad_frame is already scaled down if preprocessing did that.
            # For BVH, we need original scale.
            scaled_down_translations = quad_frame[:num_trans_ch]
            if translation_scale_factor != 0: # Should not be 0
                original_scale_translations = [t / translation_scale_factor for t in scaled_down_translations]
            else: # Fallback, should not happen
                original_scale_translations = scaled_down_translations
            euler_frame_data.extend(original_scale_translations)
        
        current_quad_offset = num_trans_ch

        if ordered_bones:
            for bone_name in ordered_bones:
                if current_quad_offset + 4 > len(quad_frame):
                    # print(f"Warning (_convert_quad_seq_to_euler_bvh_for_train_script): Frame {frame_idx}, bone {bone_name}: Not enough data for quaternion. Padding Euler.")
                    euler_frame_data.extend([0.0, 0.0, 0.0])
                    continue

                quaternion_values = torch.tensor(quad_frame[current_quad_offset : current_quad_offset + 4], dtype=torch.float32)
                # Normalize quaternion before conversion to be safe
                quaternion_values = F.normalize(quaternion_values, p=2, dim=-1)


                convention = joint_conv_map.get(bone_name)
                if not convention:
                    # print(f"Warning (_convert_quad_seq_to_euler_bvh_for_train_script): No convention for bone {bone_name}. Using ZYX fallback.")
                    convention = "ZYX" 

                rot_matrix = rc_module.quaternion_to_matrix(quaternion_values.unsqueeze(0)) # Add batch dim
                euler_angles_rad = rc_module.matrix_to_euler_angles(rot_matrix, convention).squeeze(0) # Remove batch dim
                euler_angles_deg = (euler_angles_rad * (180.0 / torch.pi)).tolist() # Use torch.pi
                euler_frame_data.extend(euler_angles_deg)
                current_quad_offset += 4
        
        # Ensure frame data matches expected Euler frame size, pad if necessary
        if len(euler_frame_data) != euler_frame_size:
            # print(f"Warning (_convert_quad_seq_to_euler_bvh_for_train_script): Frame {frame_idx} data length {len(euler_frame_data)} != expected {euler_frame_size}. Padding.")
            padding = [0.0] * (euler_frame_size - len(euler_frame_data))
            euler_frame_data.extend(padding)
        
        euler_sequence_list.append(euler_frame_data)
    
    if not euler_sequence_list:
        return np.array([], dtype=np.float32)
    return np.array(euler_sequence_list, dtype=np.float32)
# --- End of helper function ---

def train_one_iteration(real_seq_np, model, optimizer, iteration, args,
                        num_translation_channels, current_quat_frame_size, # From metadata
                        # For visualization/debug BVH saving, if done directly here:
                        # skeleton_for_bvh, ordered_bones_for_bvh, joint_to_convention_map_for_bvh
                        ):
    
    batch_size = real_seq_np.shape[0]
    # seq_len_from_data = real_seq_np.shape[1] # This is model_seq_len + 1 (for diff target)
    model_seq_len = args.seq_len # Length of sequence processed by model.forward

    # 1. Prepare input data for the model: Hip XZ differences, Hip Y absolute, Quats absolute
    # real_seq_np is (batch, seq_len_from_data_loader, quat_frame_size)
    # We need one more frame in real_seq_np than model_seq_len to calculate differences for the target.
    # So, real_seq_np should have shape (batch, model_seq_len + 1, quat_frame_size)
    
    processed_input_seq_np = np.zeros((batch_size, model_seq_len, current_quat_frame_size), dtype=np.float32)
    target_seq_np = np.zeros((batch_size, model_seq_len, current_quat_frame_size), dtype=np.float32)

    for b in range(batch_size):
        # `current_dance_segment` has `model_seq_len + 1` frames
        current_dance_segment = real_seq_np[b] 

        # Input sequence for model (model_seq_len frames)
        # Frame 0: absolute X, Y, Z and absolute Quaternions (copied directly)
        processed_input_seq_np[b, 0, :] = current_dance_segment[0, :]
        
        # Frames 1 to model_seq_len-1: dX, Y_abs, dZ, Quats_abs
        # Hip Y (index 1 if num_translation_channels > 1) and Quaternions are copied directly
        if num_translation_channels > 1:
            processed_input_seq_np[b, 1:, 1] = current_dance_segment[1:model_seq_len, 1]
        if current_quat_frame_size > num_translation_channels:
            processed_input_seq_np[b, 1:, num_translation_channels:] = current_dance_segment[1:model_seq_len, num_translation_channels:]

        # Calculate dX and dZ for input frames 1 to model_seq_len-1
        if num_translation_channels >= 1: # dX at index 0
            processed_input_seq_np[b, 1:, 0] = current_dance_segment[1:model_seq_len, 0] - current_dance_segment[0:model_seq_len-1, 0]
        if num_translation_channels >= 3: # dZ at index 2
            processed_input_seq_np[b, 1:, 2] = current_dance_segment[1:model_seq_len, 2] - current_dance_segment[0:model_seq_len-1, 2]

        # Target sequence for loss (model_seq_len frames)
        # Target[t] is what model_input[t] should predict.
        # Target frame 0: dX, Y_abs, dZ, Quats_abs (derived from current_dance_segment[1] and [0])
        # Hip Y and Quats for target[0]
        if num_translation_channels > 1:
            target_seq_np[b, 0, 1] = current_dance_segment[1, 1]
        if current_quat_frame_size > num_translation_channels:
            target_seq_np[b, 0, num_translation_channels:] = current_dance_segment[1, num_translation_channels:]
        # dX, dZ for target[0]
        if num_translation_channels >= 1:
            target_seq_np[b, 0, 0] = current_dance_segment[1, 0] - current_dance_segment[0, 0]
        if num_translation_channels >= 3:
            target_seq_np[b, 0, 2] = current_dance_segment[1, 2] - current_dance_segment[0, 2]

        # Target frames 1 to model_seq_len-1: dX, Y_abs, dZ, Quats_abs
        # (derived from current_dance_segment[t+1] and current_dance_segment[t])
        # Hip Y and Quats for target[1:]
        if num_translation_channels > 1:
            target_seq_np[b, 1:, 1] = current_dance_segment[2:model_seq_len+1, 1]
        if current_quat_frame_size > num_translation_channels:
            target_seq_np[b, 1:, num_translation_channels:] = current_dance_segment[2:model_seq_len+1, num_translation_channels:]
        # dX, dZ for target[1:]
        if num_translation_channels >= 1:
            target_seq_np[b, 1:, 0] = current_dance_segment[2:model_seq_len+1, 0] - current_dance_segment[1:model_seq_len, 0]
        if num_translation_channels >= 3:
            target_seq_np[b, 1:, 2] = current_dance_segment[2:model_seq_len+1, 2] - current_dance_segment[1:model_seq_len, 2]

    model_input_tensor = Variable(torch.from_numpy(processed_input_seq_np).float()).to(device)
    # Target for loss calculation is also in the "dX, Y_abs, dZ, Quats_abs" format.
    target_loss_tensor = Variable(torch.from_numpy(target_seq_np).float()).to(device)

    # The acLSTM.forward in synthesize_quad_motion_old.py is for generation (primes with initial_seq, then generates).
    # For training, we need a different forward pass if we use auto-conditioning with ground truth.
    # The original pytorch_train_pos_aclstm.py had a model.forward that took condition_lst.
    # Let's adapt the acLSTM model from synthesize_quad_motion_old to have such a training forward pass.
    # For now, assuming model.forward is the generation one, and we need a different way to calculate loss for training.
    # OR, the model.forward from synthesize script can be used if initial_seq is the input and we compare its output to target_loss_tensor
    # This requires that model.forward(initial_seq, generate_frames_number=0) effectively does a one-step-ahead prediction over initial_seq.
    # The current model.forward in synthesize_quad_motion_old.py *generates* `generate_frames_number` new frames *after* processing `initial_seq`.
    # This is not what we want for training loss calculation over the `model_seq_len` input.

    # We need a training-specific forward pass in the acLSTM model or here.
    # Let's assume acLSTM needs a `forward_train` method or adapt existing one.
    # For now, let's replicate the logic of the original pos training script's forward loop for auto-conditioning.

    (vec_h, vec_c) = model.init_hidden(batch_size) # Already on device
    
    # Loop for training with auto-conditioning based on Condition_num and Groundtruth_num
    # Condition_lst determines if model uses its own output or groundtruth as input for next step
    condition_lst = model.get_condition_lst(Condition_num, Groundtruth_num, model_seq_len) 
    
    predicted_frames_list = []
    current_input_to_lstm = model_input_tensor[:, 0, :] # Start with the first frame of our prepared input

    for i in range(model_seq_len):
        if i > 0: # For subsequent frames
            if condition_lst[i-1] == 1: # If previous step was groundtruth-forced for LSTM input decision
                # Use actual processed groundtruth from model_input_tensor as input for LSTM
                current_input_to_lstm = model_input_tensor[:, i, :]
            # Else (condition_lst[i-1] == 0), current_input_to_lstm remains the model's previous output_frame
        
        (output_frame, vec_h, vec_c) = model.forward_lstm(current_input_to_lstm, vec_h, vec_c)
        predicted_frames_list.append(output_frame.unsqueeze(1)) # Store (batch, 1, frame_size)
        current_input_to_lstm = output_frame # Next input is current output if not groundtruth-forced

    predict_seq_tensor = torch.cat(predicted_frames_list, dim=1) # (batch, model_seq_len, frame_size)
    
    optimizer.zero_grad()
    loss_hip, loss_quad_ql, total_loss = model.calculate_quaternion_loss_metric(predict_seq_tensor, target_loss_tensor, num_translation_channels)
    
    # The calculate_quaternion_loss_metric returns individual scalar losses, we need a combined tensor loss for backward()
    # Let's assume the 'total_loss' from that function is what we want to backpropagate.
    # However, that function returns Python floats. We need to define the combined loss here as a tensor.

    # Re-calculate loss as tensor for backprop (using the components from the metric function for consistency)
    pred_hip = predict_seq_tensor[..., :num_translation_channels]
    target_hip = target_loss_tensor[..., :num_translation_channels]
    mse_loss_fn = nn.MSELoss()
    loss_hip_tensor = mse_loss_fn(pred_hip, target_hip)

    pred_quads_flat = predict_seq_tensor[..., num_translation_channels:].reshape(-1, 4)
    target_quats_flat = target_loss_tensor[..., num_translation_channels:].reshape(-1, 4)
    
    loss_quad_ql_tensor = torch.tensor(0.0, device=device)
    if pred_quads_flat.shape[0] > 0:
        pred_q_norm = torch.nn.functional.normalize(pred_quads_flat, p=2, dim=-1)
        target_q_norm = torch.nn.functional.normalize(target_quats_flat, p=2, dim=-1)
        dot_product = torch.sum(pred_q_norm * target_q_norm, dim=-1)
        dot_product_abs = torch.abs(dot_product)
        epsilon = 1e-7
        dot_product_clamped = torch.clamp(dot_product_abs, -1.0 + epsilon, 1.0 - epsilon)
        angle_error = 2.0 * torch.acos(dot_product_clamped)
        loss_quad_ql_tensor = torch.mean(angle_error)

    # Define combined loss for backpropagation (can be weighted)    
    # These weights should be configurable or match instruction if specified
    weight_hip_loss = 1.0 
    weight_quad_loss = 1.0 # As per original paper for positional, angle loss often weighted higher
                           # The instructions mention QL = 2 * acos(|angle|), so an implicit weight of 2 is in the formula.
                           # For now, let's use 1.0 here and rely on the loss formulation.
    
    combined_loss_tensor = (weight_hip_loss * loss_hip_tensor) + (weight_quad_loss * loss_quad_ql_tensor)
    combined_loss_tensor.backward()
    optimizer.step()
    
    if args.print_loss_iter > 0 and iteration % args.print_loss_iter == 0:
        print(f"Iter {iteration:06d}/{args.total_iterations} | Total Loss: {combined_loss_tensor.item():.4f} (Hip: {loss_hip_tensor.item():.4f}, QuatQL: {loss_quad_ql_tensor.item():.4f})")
    
    return combined_loss_tensor.item() # Return scalar total loss for tracking


def generate_and_save_test_animation(model, dances_for_init, args, iteration,
                                     skeleton_for_bvh, ordered_bones_list, joint_to_convention_map, 
                                     num_translation_channels, current_quat_frame_size,
                                     translation_scaling_factor):
    model.eval() # Set model to evaluation mode
    
    test_batch_size = 1 # Generate one sample animation
    if not dances_for_init:
        print("No dances provided for generating test animation.")
        return

    # Prepare initial sequence from a random dance
    dance_idx = random.randint(0, len(dances_for_init) - 1)
    test_dance = dances_for_init[dance_idx] # This is the full dance data (already processed quat format)

    # Determine the length of the segment for generation and ground truth saving
    # The generated part will be args.num_generate_frames_for_synth
    # The initial priming part is args.num_initial_frames_for_synth
    # Total length for comparison / GT saving:
    gt_comparison_length = args.num_initial_frames_for_synth + args.num_generate_frames_for_synth
    
    # min_len_for_test now refers to the length needed from test_dance for this operation
    min_len_for_test_gt_and_synth = gt_comparison_length 
    # The generate_and_evaluate_seq also uses num_eval_frames_for_synth for its internal eval against GT.
    # Ensure dance is long enough for the *longest* segment we might need from it.
    # The call to generate_and_evaluate_seq uses num_initial_frames_for_synth + num_eval_frames_for_synth for its GT lookup.
    # For saving GT bvh, we are interested in num_initial_frames_for_synth + num_generate_frames_for_synth.
    # We should ensure test_dance is long enough for the longer of these two requirements starting from start_idx.
    
    min_len_needed_from_dance = max(
        args.num_initial_frames_for_synth + args.num_generate_frames_for_synth, # For our GT BVH
        args.num_initial_frames_for_synth + args.num_eval_frames_for_synth    # For generate_and_evaluate_seq's internal eval
    )

    if test_dance.shape[0] < min_len_needed_from_dance:
        print(f"Skipping test animation: Dance {dance_idx} too short ({test_dance.shape[0]} frames) for {min_len_needed_from_dance} required for GT saving/evaluation.")
        model.train() # Set back to training mode
        return

    max_start_idx = test_dance.shape[0] - min_len_needed_from_dance
    start_idx = random.randint(0, max_start_idx) if max_start_idx >=0 else 0
    
    initial_seq_np = test_dance[start_idx : start_idx + args.num_initial_frames_for_synth, :].reshape(1, args.num_initial_frames_for_synth, current_quat_frame_size)
    # Ground truth for comparison (if num_eval_frames_for_synth > 0)
    gt_data_for_eval_np = np.array([test_dance]) # generate_and_evaluate_seq expects a list/array of full dances for GT
    initial_seq_start_indices = [start_idx]

    output_folder_iter = os.path.join(args.write_bvh_motion_folder, f"iter_{iteration:06d}")
    if not os.path.exists(output_folder_iter):
        os.makedirs(output_folder_iter)

    # --- Save Ground Truth BVH ---
    # Extract the ground truth segment corresponding to initial + generated frames
    gt_segment_to_save_np = test_dance[start_idx : start_idx + gt_comparison_length, :]
    
    if gt_segment_to_save_np.size > 0:
        # Convert this segment to Euler for BVH
        # Note: The `skeleton_for_bvh` is the raw hierarchy, `ordered_bones_list` and `joint_to_convention_map` are from metadata.
        # The conversion function needs these, plus num_translation_channels and scaling_factor.
        gt_euler_frames_for_bvh = _convert_quad_seq_to_euler_bvh_for_train_script(
            quad_sequence_np=gt_segment_to_save_np,
            skeleton_info_ref=skeleton_for_bvh, # Reference, though primarily uses ordered_bones
            ordered_bones=ordered_bones_list,
            joint_conv_map=joint_to_convention_map,
            num_trans_ch=num_translation_channels,
            translation_scale_factor=translation_scaling_factor,
            rc_module=rc
        )

        if gt_euler_frames_for_bvh.size > 0:
            gt_bvh_filename = os.path.join(output_folder_iter, f"gt_iter_{iteration:06d}.bvh")
            try:
                frame_time = 1.0 / args.dance_frame_rate_for_synth if args.dance_frame_rate_for_synth > 0 else (1.0/60.0) # Default if not set
                read_bvh.write_frames(args.standard_bvh_file, gt_bvh_filename, gt_euler_frames_for_bvh, frame_time_override=frame_time)
                print(f"  Ground Truth Animation iter {iteration}: Saved to {gt_bvh_filename}")
            except Exception as e:
                print(f"  Error saving Ground Truth BVH for iter {iteration}: {e}")
        else:
            print(f"  Warning: Ground Truth Euler sequence for iter {iteration} is empty after conversion. Skipping GT BVH save.")
    else:
        print(f"  Warning: Ground Truth segment for iter {iteration} is empty. Skipping GT BVH save.")
    # --- End of Save Ground Truth BVH ---

    with torch.no_grad():
        _, (loss_h, loss_q, loss_t) = generate_and_evaluate_seq(
            initial_seq_np,
            gt_data_for_eval_np, # Ground truth dance(s)
            initial_seq_start_indices,
            args.num_generate_frames_for_synth, 
            model, 
            output_folder_iter, # Save in iteration-specific subfolder
            skeleton_info=skeleton_for_bvh, 
            ordered_bones_list=ordered_bones_list, 
            joint_to_convention_map=joint_to_convention_map,
            num_translation_channels=num_translation_channels,
            standard_bvh_file_for_header=args.standard_bvh_file, # From args
            translation_scaling_factor=translation_scaling_factor, # Pass scaling factor
            num_eval_frames=args.num_eval_frames_for_synth
        )
    print(f"  Test Animation iter {iteration}: Saved to {output_folder_iter}. Eval Losses: H={loss_h:.3f} Q={loss_q:.3f} T={loss_t:.3f}")
    model.train() # Set model back to training mode


def main():
    global METADATA_QUAD # Allow modification of global
    parser = argparse.ArgumentParser(description="Train Auto-Conditioned LSTM for Quaternion-based character motion.")
    # Data and Paths
    parser.add_argument('--dances_folder', type=str, required=True, help='Path to .npy QUATERNION motion files.')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata_quad.json from preprocessing.')
    parser.add_argument('--write_weight_folder', type=str, default="./weights_quad/", help='Folder to save model weights.')
    parser.add_argument('--write_bvh_motion_folder', type=str, default="./train_output_bvh_quad/", help='Folder to save generated BVH during training.')
    parser.add_argument('--read_weight_path', type=str, default=None, help='Path to pre-trained model weights to continue training.')
    parser.add_argument('--standard_bvh_file', type=str, default='../train_data_bvh/standard.bvh', help='Path to standard BVH for skeleton info during saving.')

    # Training Hyperparameters
    parser.add_argument('--total_iterations', type=int, default=50000, help='Total training iterations.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for model input during training (model_seq_len).')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
    # parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (ignored, using auto device detection).') # Device auto-detected

    # Model Architecture (hidden_size is fixed, frame_size from metadata)
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size of LSTM layers.')
    # in_frame_size / out_frame_size will be determined by metadata
    
    # Logging and Saving
    parser.add_argument('--print_loss_iter', type=int, default=20, help='Print loss every N iterations. 0 to disable.')
    parser.add_argument('--save_model_iter', type=int, default=1000, help='Save model weights every N iterations.')
    parser.add_argument('--save_bvh_iter', type=int, default=1000, help='Save test BVH animation every N iterations.')

    # Synthesis parameters for intermediate BVH generation
    parser.add_argument('--num_initial_frames_for_synth', type=int, default=20, help='Initial frames for synth during training.')
    parser.add_argument('--num_generate_frames_for_synth', type=int, default=100, help='Frames to generate for synth during training.')
    parser.add_argument('--num_eval_frames_for_synth', type=int, default=20, help='Frames for quant. eval for synth during training.')
    parser.add_argument('--dance_frame_rate_for_synth', type=int, default=60, help='Frame rate of source data (used for sampling initial seq).')

    args = parser.parse_args()

    # Create output directories
    if not os.path.exists(args.write_weight_folder): os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder): os.makedirs(args.write_bvh_motion_folder)
    if not os.path.exists(args.standard_bvh_file): print(f"Error: Standard BVH {args.standard_bvh_file} not found!"); return
    if not os.path.exists(args.metadata_path): print(f"Error: Metadata {args.metadata_path} not found!"); return

    # Load Metadata
    try:
        with open(args.metadata_path, 'r') as f:
            METADATA_QUAD = json.load(f)
        current_quat_frame_size = METADATA_QUAD['actual_quat_frame_size']
        ordered_bones_list = METADATA_QUAD['ordered_bones_with_rotations']
        joint_to_convention_map = METADATA_QUAD['joint_to_convention_map']
        num_translation_channels = METADATA_QUAD['num_translation_channels']
        translation_scaling_factor = METADATA_QUAD.get('translation_scaling_factor', 1.0) # Load with default
        print(f"Loaded metadata: FrameSize={current_quat_frame_size}, NumTrans={num_translation_channels}, Bones={len(ordered_bones_list)}, ScalingFactor={translation_scaling_factor}")
    except Exception as e:
        print(f"Error loading metadata from {args.metadata_path}: {e}"); return

    # Initialize model
    model = acLSTM(in_frame_size=current_quat_frame_size, 
                   hidden_size=args.hidden_size, 
                   out_frame_size=current_quat_frame_size)
    if args.read_weight_path and os.path.exists(args.read_weight_path):
        try:
            model.load_state_dict(torch.load(args.read_weight_path, map_location=device))
            print(f"Loaded pre-trained weights from: {args.read_weight_path}")
        except Exception as e:
            print(f"Could not load weights from {args.read_weight_path}: {e}. Training from scratch.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load training data (using the loader from synthesize_quad_motion_old for consistency)
    # load_full_dance_data loads all dances, up to a limit if specified (here, load all for flexible batching)
    # It filters out metadata.json automatically.
    all_dances = load_full_dance_data(args.dances_folder, num_dances_to_load=float('inf')) 
    if not all_dances:
        print(f"Error: No dance data loaded from {args.dances_folder}. Check path and .npy files."); return
    
    # Create a list of dance indices proportional to their length for sampling
    # The get_dance_len_lst from synthesize script has a fixed length=10, may need adjustment for diverse dance lengths
    dance_len_indices = get_dance_len_lst(all_dances) 
    if not dance_len_indices:
        print("Error: Could not create dance length index list. Ensure dances were loaded."); return

    # Load raw skeleton hierarchy once for BVH writing reference if needed by generate_and_evaluate_seq
    skeleton_for_bvh = None
    if args.standard_bvh_file and os.path.exists(args.standard_bvh_file):
        # The first return value is the skeleton dictionary, second is ordered names (we use metadata's ordered_bones_list)
        skeleton_for_bvh, _ = read_bvh_hierarchy.read_bvh_hierarchy(args.standard_bvh_file)
        if not skeleton_for_bvh:
            print(f"Warning: Could not load skeleton from {args.standard_bvh_file} for BVH writing reference.")
    else: # Ensure skeleton_for_bvh is None if file doesn't exist, to avoid issues in conversion fn
        skeleton_for_bvh = None

    model.train()
    # Training loop
    # The sequence length for data loading needs to be model_seq_len + 1 
    # because target[t] depends on original_frame[t+1] and original_frame[t]
    seq_len_for_data_loader = args.seq_len + 1

    for iteration in range(args.total_iterations):
        current_batch_list = []
        for _ in range(args.batch_size):
            chosen_dance_idx = random.choice(dance_len_indices)
            dance_data = all_dances[chosen_dance_idx]
            
            # Ensure dance is long enough for seq_len_for_data_loader
            # The synthesis script has speed adjustment, not directly used here; assume 1-to-1 frame sampling for training batches
            if dance_data.shape[0] < seq_len_for_data_loader:
                # print(f"Skipping dance {chosen_dance_idx} - too short for seq_len {seq_len_for_data_loader}")
                continue # Skip this dance, try another for the batch
            
            max_start = dance_data.shape[0] - seq_len_for_data_loader
            start_frame = random.randint(0, max_start) if max_start >= 0 else 0
            
            # Validate frame size of the chosen segment
            segment = dance_data[start_frame : start_frame + seq_len_for_data_loader, :]
            if segment.shape[1] != current_quat_frame_size:
                print(f"Critical Error: Dance {chosen_dance_idx} segment has frame size {segment.shape[1]}, expected {current_quat_frame_size}. Skipping batch.")
                # This indicates a fundamental data mismatch, should ideally not happen if preprocessing is correct.
                # Breaking out of inner loop, the outer loop will try a new batch.
                current_batch_list = [] # Clear partially filled batch
                break 
            current_batch_list.append(segment)
        
        if not current_batch_list or len(current_batch_list) < args.batch_size:
            # print(f"Warning: Batch for iter {iteration} has {len(current_batch_list)} samples, less than requested {args.batch_size}. Continuing.")
            if not current_batch_list: continue # Skip iteration if batch is empty

        real_seq_batch_np = np.array(current_batch_list, dtype=np.float32)

        train_one_iteration(real_seq_batch_np, model, optimizer, iteration, args, 
                            num_translation_channels, current_quat_frame_size)

        if args.save_model_iter > 0 and iteration % args.save_model_iter == 0 and iteration > 0:
            weight_path = os.path.join(args.write_weight_folder, f"{iteration:07d}.weight")
            torch.save(model.state_dict(), weight_path)
            print(f"Saved model weights to {weight_path}")

        if args.save_bvh_iter > 0 and iteration % args.save_bvh_iter == 0 and iteration > 0:
            generate_and_save_test_animation(model, all_dances, args, iteration, 
                                             skeleton_for_bvh, ordered_bones_list, joint_to_convention_map,
                                             num_translation_channels, current_quat_frame_size,
                                             translation_scaling_factor)

    print("Training finished.")
    final_weight_path = os.path.join(args.write_weight_folder, "final.weight")
    torch.save(model.state_dict(), final_weight_path)
    print(f"Saved final model weights to {final_weight_path}")

if __name__ == '__main__':
    main()

# end of file