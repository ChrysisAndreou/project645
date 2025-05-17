import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import torch.nn.functional as F # For normalization
import read_bvh_hierarchy # Added import
import rotation_conversions as rc # Changed import
import json # Added for metadata

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU) device for synthesize_quad_motion.py.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device for synthesize_quad_motion.py.")
else:
    device = torch.device("cpu")
    print("Using CPU device for synthesize_quad_motion.py.")
# --- End Device Setup ---

# For raw BVH-derived quaternion data: Hip X pos is 0, Y pos is 1, Z pos is 2.
# Quaternions for rotations start from index 3.

# Unused globals from positional, can be removed if generate_and_evaluate_seq is sole path
# Hip_index = read_bvh.joint_index['hip'] 
# Seq_len=100
# Hidden_size = 1024 # Model parameters are passed via args or init
# Joints_num =  57
# Condition_num=5 # Not used in synthesis model's forward
# Groundtruth_num=5 # Not used in synthesis model's forward
# In_frame_size = Joints_num*3 # Set by args

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).to(device))
        return  ([h0,h1,h2], [c0,c1,c2])
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):
        # Ensure inputs to LSTMCell are on the correct device
        in_frame_dev = in_frame.to(device)
        vec_h_dev = [h.to(device) for h in vec_h]
        vec_c_dev = [c.to(device) for c in vec_c]

        vec_h0,vec_c0 = self.lstm1(in_frame_dev, (vec_h_dev[0],vec_c_dev[0]))
        vec_h1,vec_c1 = self.lstm2(vec_h0, (vec_h_dev[1],vec_c_dev[1])) # Pass previous h (vec_h0) as input
        vec_h2,vec_c2 = self.lstm3(vec_h1, (vec_h_dev[2],vec_c_dev[2])) # Pass previous h (vec_h1) as input
     
        out_frame = self.decoder(vec_h2) #out b*frame_size
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num)) # Assuming 100 is a large enough base
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor initial_seq: b*(initial_seq_len*frame_size)
    #out cuda tensor out_seq  b* ( (intial_seq_len + generate_frame_number) *frame_size)
    def forward(self, initial_seq, generate_frames_number):
        initial_seq = initial_seq.to(device) # Ensure initial_seq is on device
        batch=initial_seq.size()[0]
        
        (vec_h, vec_c) = self.init_hidden(batch) # vec_h, vec_c are already on device
        
        out_seq_list = [] 
        current_frame = initial_seq[:,0] # Start with the first frame of initial_seq
        
        # Process initial sequence to set LSTM state
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            (current_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            # current_frame is now the prediction based on the last frame of initial_seq.
        
        # Start generating from the last predicted frame
        for i in range(generate_frames_number):
            in_frame=current_frame # Use the previously generated frame as input
            (current_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq_list.append(current_frame.unsqueeze(1)) 
            
        if not out_seq_list:
            return torch.FloatTensor([]).to(device) if batch == 0 else torch.FloatTensor(np.zeros((batch, 0, self.out_frame_size))).to(device)

        return torch.cat(out_seq_list, dim=1) 

    def calculate_quaternion_loss_metric(self, pred_seq, gt_seq, num_translation_channels):
        # pred_seq, gt_seq are [batch, num_eval_frames, frame_size]
        pred_seq = pred_seq.to(device) # Ensure on device
        gt_seq = gt_seq.to(device)     # Ensure on device

        batch_size, num_eval_frames, frame_size = pred_seq.shape

        hip_pos_pred = pred_seq[..., :num_translation_channels]
        hip_pos_gt = gt_seq[..., :num_translation_channels]
        quad_pred_flat = pred_seq[..., num_translation_channels:]
        quad_gt_flat = gt_seq[..., num_translation_channels:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        num_quad_channels = quad_pred_flat.shape[-1]
        if num_quad_channels == 0 : 
             return loss_hip.item(), 0.0, loss_hip.item()
        if num_quad_channels % 4 != 0:
            print(f"Warning: Quaternion channels ({num_quad_channels}) not divisible by 4.")
            return loss_hip.item(), -1.0, loss_hip.item() 
            
        num_quads = num_quad_channels // 4
        quad_pred = quad_pred_flat.reshape(batch_size, num_eval_frames, num_quads, 4)
        quad_gt = quad_gt_flat.reshape(batch_size, num_eval_frames, num_quads, 4)

        quad_pred_normalized = F.normalize(quad_pred, p=2, dim=-1)
        # Normalize gt_quad as well, in case they are not perfectly unit from data
        quad_gt_normalized = F.normalize(quad_gt, p=2, dim=-1) 

        dot_product = torch.sum(quad_pred_normalized * quad_gt_normalized, dim=-1)
        dot_product_abs = torch.abs(dot_product)
        epsilon = 1e-7
        dot_product_clamped = torch.clamp(dot_product_abs, -1.0 + epsilon, 1.0 - epsilon)
        angle_error = 2.0 * torch.acos(dot_product_clamped)
        loss_quad_ql = torch.mean(angle_error)
        
        total_eval_loss = loss_hip + loss_quad_ql # Consider weighting these losses
        return loss_hip.item(), loss_quad_ql.item(), total_eval_loss.item()

# This older generate_seq function has issues (e.g., Hip_index usage) and is superseded by
# generate_and_evaluate_seq. It should ideally be removed or significantly refactored if kept.
# For now, we focus on generate_and_evaluate_seq.
# numpy array inital_seq_np: batch*seq_len*frame_size
# return numpy b*generate_frames_number*frame_data
# def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):
#
#     #set hip_x and hip_z as the difference from the future frame to current frame
#     dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
#     initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
# # Hip_index was from positional, direct indexing is better for channel-based data
# #    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
# #    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
# # Corrected hip diff assignment (assuming hip is first 3 channels, X=0, Z=2)
#     if initial_seq_np.shape[2] > 2: # Check if frame has enough channels for hip
#         initial_seq_dif_hip_x_z_np[:,:,0]=dif[:,:,0] # Hip X diff
#         initial_seq_dif_hip_x_z_np[:,:,2]=dif[:,:,2] # Hip Z diff
#     else:
#         print("Warning in old generate_seq: Frame size too small for hip diff processing.")
#     
#     
#     initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
#  
#     predict_seq = model.forward(initial_seq, generate_frames_number)
#     
#     batch=initial_seq_np.shape[0]
#    
# # Determine frame size from model if possible, or must be known.
# # current_In_frame_size = model.in_frame_size # This is what the model expects for its internal decoder output
#     current_In_frame_size = initial_seq_np.shape[2] # Assuming output matches input structure before conversion

#     for b in range(batch):
#         
#         out_seq_frames=np.array(predict_seq[b].data.cpu().tolist()).reshape(-1, current_In_frame_size)
#         last_x=0.0
#         last_z=0.0
# # This hip reconstruction also assumes X=0, Z=2 for hip within the frame data
#         for frame_idx in range(out_seq_frames.shape[0]):
#             if out_seq_frames.shape[1] > 2:
#                 out_seq_frames[frame_idx,0]=out_seq_frames[frame_idx,0]+last_x
#                 last_x=out_seq_frames[frame_idx,0]
#                 
#                 out_seq_frames[frame_idx,2]=out_seq_frames[frame_idx,2]+last_z
#                 last_z=out_seq_frames[frame_idx,2]
#             else:
#                 print("Warning in old generate_seq: Frame size too small for hip reconstruction.")
#         # This call to write_traindata_to_bvh is problematic as it doesn't provide standard_bvh_file
#         # and out_seq_frames is still quaternion data.
#         read_bvh.write_traindata_to_bvh(save_dance_folder+"out"+"%02d"%b+".bvh", out_seq_frames)
#     return np.array(predict_seq.data.cpu().tolist()).reshape(batch, -1, current_In_frame_size)


# Helper function to get Euler convention string for a joint (same as in training script)
# ... existing code ...

#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        length=len(dance)/100 # Original logic, seems okay for weighting
        length=10 # This overrides, maybe keep original or make it configurable
        if(length<1):
            length=1              
        len_lst=len_lst+[length]
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length): # Cast length to int if it's float
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
#def load_dances(dance_folder): # This function is removed
#    dance_files=os.listdir(dance_folder)
#    # Filter for .npy files and exclude metadata.json
#    npy_files = [f for f in dance_files if f.endswith(".npy") and not f.endswith("metadata_quad_old.json")]
#
#    dances=[]
#    for dance_file in npy_files: # Use filtered list
#        print ("load "+dance_file)
#        # Add allow_pickle=True if your .npy files were saved with it and contain object arrays
#        try:
#            dance=np.load(os.path.join(dance_folder, dance_file), allow_pickle=True) 
#            print ("frame number: "+ str(dance.shape[0]))
#            dances=dances+[dance]
#        except Exception as e:
#            print(f"Could not load {dance_file}: {e}")
#    return dances
    
# dances: [dance1, dance2, dance3,....]
#def test(dance_batch_np, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path, # This function is removed
#         write_bvh_motion_folder, in_frame_size=171, hidden_size=1024, out_frame_size=171):
#    
#    torch.cuda.set_device(0)
#
#    # Initialize and load pretrained weights
#    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
#    model.load_state_dict(torch.load(read_weight_path))
#    # Move model to GPU
#    model.cuda()
#
#    dance_len_lst=get_dance_len_lst(dances)
#    random_range=len(dance_len_lst)
#    
#    speed=frame_rate/30 # we train the network with frame rate of 30
#    
#    dance_batch=[]
#    # Get 15 step sequence initial motion
#    for b in range(batch):
#        #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
#        dance_id = dance_len_lst[np.random.randint(0,random_range)]
#        dance=dances[dance_id].copy()
#        dance_len = dance.shape[0]
#            
#        start_id=random.randint(10, int(dance_len-initial_seq_len*speed-10))#the first and last several frames are sometimes noisy. 
#        sample_seq=[]
#        for i in range(initial_seq_len):
#            sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
#        
#        dance_batch=dance_batch+[sample_seq]
#            
#    dance_batch_np=np.array(dance_batch)
#
#    # Generate the next steps
#    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)

def generate_and_evaluate_seq(initial_seq_np, full_dance_data_np, initial_seq_start_indices, 
                                generate_frames_number, model, save_dance_folder, 
                                skeleton_info, ordered_bones_list, joint_to_convention_map,
                                num_translation_channels, standard_bvh_file_for_header, 
                                translation_scaling_factor, # Added for scaling
                                num_eval_frames=20):
    batch_size = initial_seq_np.shape[0]
    initial_seq_len = initial_seq_np.shape[1]
    # current_frame_size = initial_seq_np.shape[2] # This should match model.in_frame_size

    temp_initial_seq_for_lstm = initial_seq_np.copy()
    # Hip X (index 0) and Z (index 2) velocity calculation, if num_translation_channels allows
    if initial_seq_len > 1:
        if num_translation_channels >= 1: # Check for X
            base_hip_x = temp_initial_seq_for_lstm[:, 0, 0].copy()
        if num_translation_channels >= 3: # Check for Z
            base_hip_z = temp_initial_seq_for_lstm[:, 0, 2].copy()
        
        for i in range(1, initial_seq_len):
            if num_translation_channels >= 1:
                current_hip_x = temp_initial_seq_for_lstm[:, i, 0].copy()
                temp_initial_seq_for_lstm[:, i, 0] = current_hip_x - base_hip_x
                base_hip_x = current_hip_x
            if num_translation_channels >= 3:
                current_hip_z = temp_initial_seq_for_lstm[:, i, 2].copy()
                temp_initial_seq_for_lstm[:, i, 2] = current_hip_z - base_hip_z
                base_hip_z = current_hip_z
    
    initial_seq_for_lstm_cuda  = torch.autograd.Variable(torch.FloatTensor(temp_initial_seq_for_lstm.tolist())).to(device)

    generated_seq_raw_cuda = model.forward(initial_seq_for_lstm_cuda, generate_frames_number)
    if generated_seq_raw_cuda.numel() == 0: 
        print("Warning: model.forward returned empty tensor. Skipping saving and evaluation.")
        return None, (0,0,0) # Return empty and zero losses
    generated_seq_raw_np = generated_seq_raw_cuda.data.cpu().numpy()

    generated_seq_abs_hip_np = generated_seq_raw_np.copy()
    # Restore absolute hip X and Z positions
    if num_translation_channels >= 1:
        last_abs_hip_x = initial_seq_np[:, -1, 0].copy()
    if num_translation_channels >= 3:
        last_abs_hip_z = initial_seq_np[:, -1, 2].copy()

    for i in range(generate_frames_number):
        if num_translation_channels >= 1:
            current_pred_delta_x = generated_seq_abs_hip_np[:, i, 0]
            generated_seq_abs_hip_np[:, i, 0] = last_abs_hip_x + current_pred_delta_x
            last_abs_hip_x = generated_seq_abs_hip_np[:, i, 0]
        if num_translation_channels >= 3:
            current_pred_delta_z = generated_seq_abs_hip_np[:, i, 2]
            generated_seq_abs_hip_np[:, i, 2] = last_abs_hip_z + current_pred_delta_z
            last_abs_hip_z = generated_seq_abs_hip_np[:, i, 2]
        # Hip Y and quaternions are already absolute from network output

    # --- Function to convert a sequence from quad to euler for BVH ---
    def convert_quad_seq_to_euler_bvh(quad_sequence_np, _skeleton_info, _ordered_bones, _joint_conv_map, _num_trans_ch, _translation_scale_factor, rc_module):
        num_frames = quad_sequence_np.shape[0]
        if not _ordered_bones: 
            print("Warning: No ordered_bones for rotation, BVH will only contain hip if anything.")
            euler_frame_size = _num_trans_ch 
        else:
            euler_frame_size = _num_trans_ch + len(_ordered_bones) * 3
        
        euler_sequence_list = []

        for frame_idx in range(num_frames):
            quad_frame = quad_sequence_np[frame_idx]
            euler_frame_data = []
            if _num_trans_ch > 0:
                # Scale up translations before adding to BVH frame data
                scaled_down_translations = quad_frame[:_num_trans_ch]
                if _translation_scale_factor != 0:
                    original_scale_translations = [t / _translation_scale_factor for t in scaled_down_translations]
                else: # Avoid division by zero if factor is somehow 0 (should not happen)
                    original_scale_translations = scaled_down_translations
                euler_frame_data.extend(original_scale_translations) 
            
            current_quad_offset = _num_trans_ch

            if _ordered_bones: 
                for bone_name in _ordered_bones:
                    if current_quad_offset + 4 > len(quad_frame):
                        print(f"Warning: Frame {frame_idx}, bone {bone_name}: Not enough data for quaternion. Padding Euler.")
                        euler_frame_data.extend([0.0, 0.0, 0.0]) 
                        continue 

                    quaternion_values = torch.tensor(quad_frame[current_quad_offset : current_quad_offset + 4], dtype=torch.float32)
                    convention = _joint_conv_map.get(bone_name)
                    if not convention:
                        print(f"Warning: No convention for bone {bone_name} in map. Using ZYX fallback.")
                        convention = "ZYX" # Fallback, or could pad with zeros / raise error

                    rot_matrix = rc_module.quaternion_to_matrix(quaternion_values.unsqueeze(0))
                    euler_angles_rad = rc_module.matrix_to_euler_angles(rot_matrix, convention).squeeze(0)
                    euler_angles_deg = (euler_angles_rad * (180.0 / torch.pi)).tolist() # Use torch.pi
                    euler_frame_data.extend(euler_angles_deg)
                    current_quad_offset += 4
            
            if len(euler_frame_data) != euler_frame_size:
                padding = [0.0] * (euler_frame_size - len(euler_frame_data))
                euler_frame_data.extend(padding)
            
            euler_sequence_list.append(euler_frame_data)
        
        if not euler_sequence_list:
            return np.array([], dtype=np.float32) 
        return np.array(euler_sequence_list, dtype=np.float32)
    # --- End of conversion function ---

    for b_idx in range(batch_size):
        full_quad_output_seq_np = np.concatenate((initial_seq_np[b_idx], generated_seq_abs_hip_np[b_idx]), axis=0)
        
        if not os.path.exists(save_dance_folder):
            os.makedirs(save_dance_folder)

        # skeleton_info here is the full skeleton dict, ordered_bones_list and joint_to_convention_map are from metadata
        if skeleton_info and ordered_bones_list and joint_to_convention_map and standard_bvh_file_for_header:            
            full_euler_output_seq_bvh = convert_quad_seq_to_euler_bvh(
                full_quad_output_seq_np, 
                skeleton_info, # Pass the raw skeleton
                ordered_bones_list, 
                joint_to_convention_map,
                num_translation_channels,
                translation_scaling_factor, # Pass scaling factor
                rc
            )
            
            if full_euler_output_seq_bvh.size > 0:
                # write_traindata_to_bvh expects (standard_bvh_ref, output_path, motion_data_frames)
                read_bvh.write_frames(standard_bvh_file_for_header, # Use the argument here
                                      os.path.join(save_dance_folder, f"generated_quad_to_euler_{b_idx:02d}.bvh"), 
                                      full_euler_output_seq_bvh)
                print(f"Saved generated Quad (converted to Euler) motion for batch {b_idx} to generated_quad_to_euler_{b_idx:02d}.bvh")
            else:
                print(f"Warning: Euler sequence for batch {b_idx} is empty after conversion. Skipping BVH save.")
        else:
            print(f"Warning: Missing info for BVH conversion (skeleton, ordered_bones, convention_map, or header path) for batch {b_idx}. Saving raw quaternion data.")
            np.save(os.path.join(save_dance_folder, f"generated_quad_raw_{b_idx:02d}.npy"), full_quad_output_seq_np)

    loss_hip, loss_quad_ql, total_loss = (0.0, 0.0, 0.0) # Default losses
    if num_eval_frames > 0 and generated_seq_abs_hip_np.shape[1] >= num_eval_frames:
        eval_pred_seq_abs_hip_np = generated_seq_abs_hip_np[:, :num_eval_frames, :]
        eval_gt_seq_list = []
        current_frame_size_from_data = initial_seq_np.shape[2] # Get frame size from actual data

        for b_idx in range(batch_size):
            gt_start_frame_idx = initial_seq_start_indices[b_idx] + initial_seq_len
            gt_end_frame_idx = gt_start_frame_idx + num_eval_frames
            
            # Ensure full_dance_data_np[b_idx] exists and has enough frames
            if b_idx < len(full_dance_data_np) and full_dance_data_np[b_idx] is not None:
                current_dance_data = full_dance_data_np[b_idx]
                if gt_end_frame_idx <= current_dance_data.shape[0]:
                    eval_gt_seq_list.append(current_dance_data[gt_start_frame_idx:gt_end_frame_idx, :])
                else: # Handle cases where ground truth is shorter than needed
                    available_gt = current_dance_data[gt_start_frame_idx:, :] if gt_start_frame_idx < current_dance_data.shape[0] else np.array([])
                    padding_needed = num_eval_frames - available_gt.shape[0]
                    # Use current_frame_size_from_data for padding
                    padding_array = np.zeros((padding_needed, current_frame_size_from_data)) if padding_needed > 0 else np.array([]) 
                    
                    if available_gt.size > 0 and padding_array.size > 0:
                        eval_gt_seq_list.append(np.concatenate((available_gt, padding_array), axis=0))
                    elif available_gt.size > 0:
                        eval_gt_seq_list.append(available_gt) # Should not happen if padding_needed > 0
                    elif padding_array.size > 0 : # Only padding needed
                         eval_gt_seq_list.append(padding_array)
                    else: # No data, no padding (num_eval_frames was 0 or available_gt covered it)
                         eval_gt_seq_list.append(np.zeros((num_eval_frames, current_frame_size_from_data)))
            else: # If a specific dance is missing in full_dance_data_np
                print(f"Warning: Ground truth dance data missing for batch index {b_idx}. Using zeros for evaluation.")
                eval_gt_seq_list.append(np.zeros((num_eval_frames, current_frame_size_from_data)))

        if len(eval_gt_seq_list) == batch_size:
            eval_gt_seq_np = np.array(eval_gt_seq_list)
            # Ensure eval_gt_seq_np has the correct frame size if padding occurred differently
            if eval_gt_seq_np.shape[-1] != eval_pred_seq_abs_hip_np.shape[-1]:
                 print(f"Warning: Mismatch in frame size for evaluation. GT: {eval_gt_seq_np.shape[-1]}, Pred: {eval_pred_seq_abs_hip_np.shape[-1]}. Using zeros for GT.")
                 eval_gt_seq_np = np.zeros_like(eval_pred_seq_abs_hip_np)


            loss_hip, loss_quad_ql, total_loss = model.calculate_quaternion_loss_metric(
                torch.FloatTensor(eval_pred_seq_abs_hip_np).to(device), 
                torch.FloatTensor(eval_gt_seq_np).to(device),
                num_translation_channels
            )
            print(f"---- Quantitative Evaluation (first {num_eval_frames} generated frames) ----")
            print(f"Hip MSE Loss: {loss_hip:.4f}")
            print(f"Quaternion QL Loss: {loss_quad_ql:.4f}")
            print(f"Total Combined Eval Loss: {total_loss:.4f}")
            print("----------------------------------------------------------")
    return generated_seq_abs_hip_np, (loss_hip, loss_quad_ql, total_loss)


def load_full_dance_data(dance_folder, num_dances_to_load):
    # Exclude metadata file from being loaded as a dance
    dance_files = sorted([f for f in os.listdir(dance_folder) if f.endswith(".npy") and not f.endswith("metadata_quad.json")])
    dances = []
    print(f'Loading full motion files for ground truth from {dance_folder}...')
    for i, dance_file in enumerate(dance_files):
        if i >= num_dances_to_load: break
        dance_path = os.path.join(dance_folder, dance_file)
        try: 
            # Add allow_pickle=True if your .npy files might contain object arrays
            dances.append(np.load(dance_path, allow_pickle=True))
        except Exception as e: print(f"Could not load {dance_path}: {e}")
    print(f"{len(dances)} full motion files loaded.")
    return dances

def main():
    parser = argparse.ArgumentParser(description="Synthesize Quaternion based character motion.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained .weight model.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to .npy motion files (Quaternion format).')
    parser.add_argument('--output_folder', type=str, default="./synthesized_quad_motions/", help='Folder to save generated BVH/NPY files.')
    parser.add_argument('--num_initial_frames', type=int, default=20, help='Initial frames to feed model.')
    parser.add_argument('--num_generate_frames', type=int, default=400, help='Frames to generate.')
    parser.add_argument('--num_eval_frames', type=int, default=20, help='Generated frames for quantitative evaluation.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to generate.')
    # in_frame_size will be loaded from metadata
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size of LSTM layers in the model.')
    # parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (ignored, using auto device detection).') # Commented out as device is auto-detected
    parser.add_argument('--standard_bvh_file', type=str, default='../train_data_bvh/standard.bvh', help='Path to the standard BVH file for skeleton hierarchy and BVH writing.')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata_quad.json file from preprocessing.')


    args = parser.parse_args()

    # GPU device is set globally. torch.cuda.set_device(args.gpu_id) is not needed if using global `device`.
    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)
    if not os.path.exists(args.model_path): print(f"Error: Model {args.model_path} not found."); return
    if not os.path.exists(args.data_folder): print(f"Error: Data folder {args.data_folder} not found."); return
    if not os.path.exists(args.metadata_path): print(f"Error: Metadata file {args.metadata_path} not found."); return
    if not os.path.exists(args.standard_bvh_file): print(f"Error: Standard BVH file {args.standard_bvh_file} not found."); return


    # Load metadata
    try:
        with open(args.metadata_path, 'r') as f:
            metadata = json.load(f)
        loaded_quat_frame_size = metadata['actual_quat_frame_size']
        ordered_bones_from_meta = metadata['ordered_bones_with_rotations']
        joint_to_convention_map_from_meta = metadata['joint_to_convention_map']
        num_translation_channels_from_meta = metadata['num_translation_channels']
        translation_scaling_factor_from_meta = metadata.get('translation_scaling_factor', 1.0) # Load with default
        print(f"Loaded metadata: Frame Size={loaded_quat_frame_size}, NumTransChannels={num_translation_channels_from_meta}, ScalingFactor={translation_scaling_factor_from_meta}")
    except Exception as e:
        print(f"Error loading metadata from {args.metadata_path}: {e}"); return

    # Initialize model with frame size from metadata
    model = acLSTM(in_frame_size=loaded_quat_frame_size, hidden_size=args.hidden_size, out_frame_size=loaded_quat_frame_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device)) # Ensure map_location for device flexibility
    model.to(device) # Move model to device
    model.eval()

    # Load skeleton information (raw skeleton dict for reference, if needed by conversion, though primarily using metadata)
    skeleton_for_bvh_write = None
    if args.standard_bvh_file and os.path.exists(args.standard_bvh_file):
        print(f"Loading raw skeleton from: {args.standard_bvh_file} for BVH writer reference.")
        # This skeleton is primarily for read_bvh.write_frames if it needs the hierarchy,
        # but conversions will use metadata's ordered_bones and conventions.
        skeleton_for_bvh_write, _ = read_bvh_hierarchy.read_bvh_hierarchy(args.standard_bvh_file)
        if not skeleton_for_bvh_write:
            print(f"Warning: Could not load full skeleton from {args.standard_bvh_file} for BVH writing reference, but proceeding with metadata.")
    else:
        print(f"Warning: Standard BVH file '{args.standard_bvh_file}' not found. BVH saving might be impacted if write_frames relies heavily on full hierarchy beyond metadata.")

    # Load dance data for initial sequences and ground truth
    full_dances = load_full_dance_data(args.data_folder, args.batch_size)
    if not full_dances or len(full_dances) < args.batch_size:
        print(f"Error: Need at least {args.batch_size} dances in {args.data_folder}, found {len(full_dances)}. (Ensure metadata_quad_old.json is not counted as a dance)."); return

    initial_seq_batch_list, initial_seq_start_indices_list = [], []
    actual_batch_size = 0 # In case some dances are too short

    for i in range(args.batch_size):
        if i >= len(full_dances): # Should be caught by previous check but good for safety
            print(f"Warning: Requested batch size {args.batch_size} but only {len(full_dances)} dances available. Reducing batch size.")
            break 
        dance = full_dances[i]
        # Validate frame size of loaded dance against metadata
        if dance.shape[1] != loaded_quat_frame_size:
            print(f"Error: Dance {i} has frame size {dance.shape[1]}, but metadata expects {loaded_quat_frame_size}. Skipping this dance.")
            continue

        min_len_needed = args.num_initial_frames + args.num_eval_frames # For GT comparison
        if dance.shape[0] < min_len_needed: # If comparing to GT for eval_frames
             # If no eval, then just num_initial_frames
            if args.num_eval_frames > 0 and dance.shape[0] < args.num_initial_frames + args.num_eval_frames :
                 print(f"Warning: Dance {i} too short ({dance.shape[0]} frames) for {args.num_initial_frames} initial + {args.num_eval_frames} eval frames. Skipping.")
                 continue
            elif dance.shape[0] < args.num_initial_frames:
                 print(f"Warning: Dance {i} too short ({dance.shape[0]} frames) for {args.num_initial_frames} initial frames. Skipping.")
                 continue

        # Max start index calculation needs to ensure we can get initial_seq + eval_frames for GT
        max_start_idx_for_gt_compare = dance.shape[0] - (args.num_initial_frames + args.num_eval_frames)
        max_start_idx_for_initial_only = dance.shape[0] - args.num_initial_frames
        
        # If we do evaluation, we must be able to pick a start_idx that allows for initial + eval from GT
        # Otherwise, we just need to pick a start_idx for the initial sequence.
        max_start_idx = max_start_idx_for_gt_compare if args.num_eval_frames > 0 else max_start_idx_for_initial_only

        if max_start_idx < 0 : # Should be caught by length check above, but defensive
            print(f"Warning: Dance {i} max_start_idx ({max_start_idx}) is less than 0. Skipping.")
            continue

        start_idx = random.randint(0, max_start_idx) # Random start
        initial_seq_batch_list.append(dance[start_idx : start_idx + args.num_initial_frames, :])
        initial_seq_start_indices_list.append(start_idx)
        actual_batch_size +=1
    
    if not initial_seq_batch_list: print("Error: No initial sequences created. Check dance file lengths and frame sizes."); return
    
    # Adjust full_dances_np for evaluation to match the dances actually used for initial sequences
    relevant_full_dances_list = []
    for i in range(actual_batch_size): # Use actual_batch_size
        # This assumes full_dances indices align with the ones chosen for initial_seq_batch_list
        # A safer way would be to store which original dance index was used for each item in initial_seq_batch_list
        # For now, assuming direct correspondence up to actual_batch_size if dances weren't skipped randomly but sequentially.
        # If dances were skipped, initial_seq_start_indices_list corresponds to the *filtered* list of dances that were long enough.
        # This part needs care. Let's re-construct relevant_full_dances_np based on dances that formed initial_seq_batch_list.
        # However, the original code implies `full_dances` maps directly to the batch.
        # So if initial_seq_batch_list has `actual_batch_size` items, we take the first `actual_batch_size` from `full_dances`.
         if i < len(full_dances): # Ensure we don't go out of bounds if actual_batch_size < args.batch_size
            relevant_full_dances_list.append(full_dances[i])
         else: # Should not happen if actual_batch_size is derived correctly
            relevant_full_dances_list.append(np.zeros((1, loaded_quat_frame_size))) # Placeholder if something went wrong

    initial_seq_batch_np = np.array(initial_seq_batch_list)
    # relevant_full_dances_np should correspond to the dances from which initial sequences were successfully drawn
    # This needs to be robust if some dances from `full_dances` were skipped.
    # The simplest is to pass the subset of `full_dances` that corresponds to `initial_seq_batch_np`.
    # If `initial_seq_batch_list` was formed by taking dances 0 to `actual_batch_size-1` from `full_dances`, then:
    relevant_full_dances_np_for_eval = np.array(relevant_full_dances_list)


    with torch.no_grad():
        _, (loss_h, loss_q, loss_t) = generate_and_evaluate_seq(
                                    initial_seq_batch_np, 
                                    relevant_full_dances_np_for_eval, # Pass the potentially smaller, aligned GT array
                                    initial_seq_start_indices_list, 
                                    args.num_generate_frames, model, 
                                    args.output_folder,
                                    skeleton_info=skeleton_for_bvh_write, # Pass loaded raw skeleton
                                    ordered_bones_list=ordered_bones_from_meta, # Pass from metadata
                                    joint_to_convention_map=joint_to_convention_map_from_meta, # Pass from metadata
                                    num_translation_channels=num_translation_channels_from_meta, # Pass from metadata
                                    standard_bvh_file_for_header=args.standard_bvh_file, 
                                    translation_scaling_factor=translation_scaling_factor_from_meta, # Pass scaling factor
                                    num_eval_frames=args.num_eval_frames)
    print(f"Motion synthesis finished. Output in {args.output_folder}")
    print(f"Final Evaluation Losses: Hip MSE={loss_h:.4f}, Quad QL={loss_q:.4f}, Total={loss_t:.4f}")

if __name__ == '__main__':
    main()