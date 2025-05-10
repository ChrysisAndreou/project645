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
from code import rotation_conversions as rc # Added import

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
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor initial_seq: b*(initial_seq_len*frame_size)
    #out cuda tensor out_seq  b* ( (intial_seq_len + generate_frame_number) *frame_size)
    def forward(self, initial_seq, generate_frames_number):
        
        batch=initial_seq.size()[0]
        
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        # out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda()) # Original code for cat in loop
        out_seq_list = [] # Store frames as a list of tensors for efficient concatenation

        # current_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda()) # Original
        current_frame = initial_seq[:,0] # Start with the first frame of initial_seq for the loop logic
        
        # Process initial sequence to set LSTM state
        # The loop for initial_seq feeds frames and updates state, last out_frame is prediction after init_seq[-1]
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            (current_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            # We don't add these to out_seq_list during priming phase.
            # current_frame is now the prediction based on the last frame of initial_seq.
        
        # Start generating from the last predicted frame (which is now in current_frame)
        for i in range(generate_frames_number):
            in_frame=current_frame # Use the previously generated frame as input
            (current_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq_list.append(current_frame.unsqueeze(1)) # Add (batch, 1, frame_size) to list
            
        if not out_seq_list:
             # This case should ideally not be hit if generate_frames_number > 0
            return torch.FloatTensor([]).cuda() if batch == 0 else torch.FloatTensor(np.zeros((batch, 0, self.out_frame_size))).cuda()

        # Concatenate along seq_len dim -> (batch, gen_frames, frame_size)
        return torch.cat(out_seq_list, dim=1) 

    # This is unused in favor of calculate_quaternion_loss_metric
    # def calculate_loss(self, out_seq, groundtruth_seq):
    # ... (content of calculate_loss commented out) ...
    #     return loss

    def calculate_quaternion_loss_metric(self, pred_seq, gt_seq):
        # pred_seq, gt_seq are [batch, num_eval_frames, frame_size]
        pred_seq = pred_seq.to(gt_seq.device)
        batch_size, num_eval_frames, frame_size = pred_seq.shape

        hip_pos_pred = pred_seq[..., :3]
        hip_pos_gt = gt_seq[..., :3]
        quad_pred_flat = pred_seq[..., 3:]
        quad_gt_flat = gt_seq[..., 3:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        num_quad_channels = quad_pred_flat.shape[-1]
        if num_quad_channels == 0 : # No quaternion data
             return loss_hip.item(), 0.0, loss_hip.item()
        if num_quad_channels % 4 != 0:
            print(f"Warning: Quaternion channels ({num_quad_channels}) not divisible by 4.")
            return loss_hip.item(), -1.0, loss_hip.item() # Indicate error
            
        num_quads = num_quad_channels // 4
        quad_pred = quad_pred_flat.reshape(batch_size, num_eval_frames, num_quads, 4)
        quad_gt = quad_gt_flat.reshape(batch_size, num_eval_frames, num_quads, 4)

        quad_pred_normalized = F.normalize(quad_pred, p=2, dim=-1)
        dot_product = torch.sum(quad_pred_normalized * quad_gt, dim=-1)
        dot_product_abs = torch.abs(dot_product)
        epsilon = 1e-7
        dot_product_clamped = torch.clamp(dot_product_abs, -1.0 + epsilon, 1.0 - epsilon)
        angle_error = 2.0 * torch.acos(dot_product_clamped)
        loss_quad_ql = torch.mean(angle_error)
        
        total_eval_loss = loss_hip + loss_quad_ql
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
        length=len(dance)/100
        length=10
        if(length<1):
            length=1              
        len_lst=len_lst+[length]
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    for dance_file in dance_files:
        print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file)
        print ("frame number: "+ str(dance.shape[0]))
        dances=dances+[dance]
    return dances
    
# dances: [dance1, dance2, dance3,....]
def test(dance_batch_np, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
         write_bvh_motion_folder, in_frame_size=171, hidden_size=1024, out_frame_size=171):
    
    torch.cuda.set_device(0)

    # Initialize and load pretrained weights
    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
    model.load_state_dict(torch.load(read_weight_path))
    # Move model to GPU
    model.cuda()

    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    dance_batch=[]
    # Get 15 step sequence initial motion
    for b in range(batch):
        #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
        dance_id = dance_len_lst[np.random.randint(0,random_range)]
        dance=dances[dance_id].copy()
        dance_len = dance.shape[0]
            
        start_id=random.randint(10, int(dance_len-initial_seq_len*speed-10))#the first and last several frames are sometimes noisy. 
        sample_seq=[]
        for i in range(initial_seq_len):
            sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
        
        dance_batch=dance_batch+[sample_seq]
            
    dance_batch_np=np.array(dance_batch)

    # Generate the next steps
    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)

def generate_and_evaluate_seq(initial_seq_np, full_dance_data_np, initial_seq_start_indices, 
                                generate_frames_number, model, save_dance_folder, 
                                skeleton_info, ordered_bones_list, standard_bvh_file_for_header, # New args
                                num_eval_frames=20):
    batch_size = initial_seq_np.shape[0]
    initial_seq_len = initial_seq_np.shape[1]
    current_frame_size = initial_seq_np.shape[2]

    temp_initial_seq_for_lstm = initial_seq_np.copy()
    if initial_seq_len > 1:
        base_hip_x = temp_initial_seq_for_lstm[:, 0, 0].copy()
        base_hip_z = temp_initial_seq_for_lstm[:, 0, 2].copy()
        for i in range(1, initial_seq_len):
            current_hip_x = temp_initial_seq_for_lstm[:, i, 0].copy()
            current_hip_z = temp_initial_seq_for_lstm[:, i, 2].copy()
            temp_initial_seq_for_lstm[:, i, 0] = current_hip_x - base_hip_x
            temp_initial_seq_for_lstm[:, i, 2] = current_hip_z - base_hip_z
            base_hip_x = current_hip_x
            base_hip_z = current_hip_z
    initial_seq_for_lstm_cuda  = torch.autograd.Variable(torch.FloatTensor(temp_initial_seq_for_lstm.tolist()).cuda())

    generated_seq_raw_cuda = model.forward(initial_seq_for_lstm_cuda, generate_frames_number)
    if generated_seq_raw_cuda.numel() == 0: # Handle empty tensor if nothing generated
        print("Warning: model.forward returned empty tensor. Skipping saving and evaluation.")
        return None
    generated_seq_raw_np = generated_seq_raw_cuda.data.cpu().numpy()

    generated_seq_abs_hip_np = generated_seq_raw_np.copy()
    last_abs_hip_x = initial_seq_np[:, -1, 0].copy()
    last_abs_hip_z = initial_seq_np[:, -1, 2].copy()

    for i in range(generate_frames_number):
        current_pred_delta_x = generated_seq_abs_hip_np[:, i, 0]
        current_pred_delta_z = generated_seq_abs_hip_np[:, i, 2]
        generated_seq_abs_hip_np[:, i, 0] = last_abs_hip_x + current_pred_delta_x
        generated_seq_abs_hip_np[:, i, 2] = last_abs_hip_z + current_pred_delta_z
        last_abs_hip_x = generated_seq_abs_hip_np[:, i, 0]
        last_abs_hip_z = generated_seq_abs_hip_np[:, i, 2]
        # Hip Y and quaternions are already absolute from network output

    # --- Function to convert a sequence from quad to euler for BVH (similar to training script) ---
    def convert_quad_seq_to_euler_bvh(quad_sequence_np, skeleton, ordered_bones, rc_module):
        num_frames = quad_sequence_np.shape[0]
        if not ordered_bones: # If no rotating bones, output might just be hip
            print("Warning: No ordered_bones for rotation, BVH will only contain hip if anything.")
            # Handle this case: perhaps return only hip data or an empty array if frame size is unexpected
            # For now, assume this implies an issue and proceed cautiously, or return minimal data
            euler_frame_size = 3 # Only hip
        else:
            euler_frame_size = 3 + len(ordered_bones) * 3
        
        euler_sequence_list = []

        for frame_idx in range(num_frames):
            quad_frame = quad_sequence_np[frame_idx]
            euler_frame_data = []
            euler_frame_data.extend(quad_frame[:3]) # Hip translation
            current_quad_offset = 3

            if ordered_bones: # Only process bones if list is not empty
                for bone_name in ordered_bones:
                    if current_quad_offset + 4 > len(quad_frame):
                        print(f"Warning: Frame {frame_idx}, bone {bone_name}: Not enough data for quaternion. Padding Euler.")
                        euler_frame_data.extend([0.0, 0.0, 0.0]) # Pad Euler angles
                        continue # Skip to next bone, don't increment offset if data was missing

                    quaternion_values = torch.tensor(quad_frame[current_quad_offset : current_quad_offset + 4], dtype=torch.float32)
                    convention = get_euler_convention_for_joint(bone_name, skeleton)
                    rot_matrix = rc_module.quaternion_to_matrix(quaternion_values.unsqueeze(0))
                    euler_angles_rad = rc_module.matrix_to_euler_angles(rot_matrix, convention).squeeze(0)
                    euler_angles_deg = (euler_angles_rad * (180.0 / torch.pi)).tolist()
                    euler_frame_data.extend(euler_angles_deg)
                    current_quad_offset += 4
            
            # Ensure frame data matches expected euler_frame_size (padding if necessary due to earlier skips/warnings)
            if len(euler_frame_data) != euler_frame_size:
                # This might happen if ordered_bones was empty, or if a bone was skipped
                # Pad with zeros to meet the expected euler_frame_size for consistency
                padding = [0.0] * (euler_frame_size - len(euler_frame_data))
                euler_frame_data.extend(padding)
            
            euler_sequence_list.append(euler_frame_data)
        
        if not euler_sequence_list:
            return np.array([], dtype=np.float32) # Return empty if nothing processed
        return np.array(euler_sequence_list, dtype=np.float32)
    # --- End of conversion function ---

    # Save full generated sequences (initial_absolute + generated_absolute_hip)
    for b_idx in range(batch_size):
        # Concatenate the original initial sequence (absolute) with the generated part (absolute hip)
        # Both initial_seq_np[b_idx] and generated_seq_abs_hip_np[b_idx] are in Quad format here.
        full_quad_output_seq_np = np.concatenate((initial_seq_np[b_idx], generated_seq_abs_hip_np[b_idx]), axis=0)
        
        if not os.path.exists(save_dance_folder):
            os.makedirs(save_dance_folder)

        if skeleton_info and ordered_bones_list and standard_bvh_file_for_header:            
            # Convert the ENTIRE sequence (initial_seed_quad + generated_quad) to Euler for BVH
            full_euler_output_seq_bvh = convert_quad_seq_to_euler_bvh(full_quad_output_seq_np, skeleton_info, ordered_bones_list, rc)
            
            if full_euler_output_seq_bvh.size > 0:
                read_bvh.write_traindata_to_bvh(standard_bvh_file_for_header, 
                                                os.path.join(save_dance_folder, f"generated_quad_to_euler_{b_idx:02d}.bvh"), 
                                                full_euler_output_seq_bvh)
                print(f"Saved generated Quad (converted to Euler) motion for batch {b_idx} to generated_quad_to_euler_{b_idx:02d}.bvh")
            else:
                print(f"Warning: Euler sequence for batch {b_idx} is empty after conversion. Skipping BVH save.")
        else:
            # Fallback: Save raw quaternion data as .npy if skeleton info is missing for BVH conversion
            print(f"Warning: Skeleton info missing for batch {b_idx}. Saving raw quaternion data as .npy instead of BVH.")
            np.save(os.path.join(save_dance_folder, f"generated_quad_raw_{b_idx:02d}.npy"), full_quad_output_seq_np)

    if num_eval_frames > 0 and generated_seq_abs_hip_np.shape[1] >= num_eval_frames:
        eval_pred_seq_abs_hip_np = generated_seq_abs_hip_np[:, :num_eval_frames, :]
        eval_gt_seq_list = []
        for b_idx in range(batch_size):
            gt_start_frame_idx = initial_seq_start_indices[b_idx] + initial_seq_len
            gt_end_frame_idx = gt_start_frame_idx + num_eval_frames
            if gt_end_frame_idx <= full_dance_data_np[b_idx].shape[0]:
                eval_gt_seq_list.append(full_dance_data_np[b_idx][gt_start_frame_idx:gt_end_frame_idx, :])
            else:
                available_gt = full_dance_data_np[b_idx][gt_start_frame_idx:, :] if gt_start_frame_idx < full_dance_data_np[b_idx].shape[0] else np.array([])
                padding_needed = num_eval_frames - available_gt.shape[0]
                padding_array = np.zeros((padding_needed, current_frame_size)) if padding_needed > 0 else np.array([])
                eval_gt_seq_list.append(np.concatenate((available_gt, padding_array), axis=0) if available_gt.size > 0 or padding_array.size > 0 else np.zeros((num_eval_frames, current_frame_size)))
        
        if len(eval_gt_seq_list) == batch_size:
            eval_gt_seq_np = np.array(eval_gt_seq_list)
            loss_hip, loss_quad_ql, total_loss = model.calculate_quaternion_loss_metric(
                torch.FloatTensor(eval_pred_seq_abs_hip_np).cuda(), 
                torch.FloatTensor(eval_gt_seq_np).cuda()
            )
            print(f"---- Quantitative Evaluation (first {num_eval_frames} generated frames) ----")
            print(f"Hip MSE Loss: {loss_hip:.4f}")
            print(f"Quaternion QL Loss: {loss_quad_ql:.4f}")
            print(f"Total Combined Eval Loss: {total_loss:.4f}")
            print("----------------------------------------------------------")
    return generated_seq_abs_hip_np

def load_full_dance_data(dance_folder, num_dances_to_load):
    dance_files = sorted([f for f in os.listdir(dance_folder) if f.endswith(".npy")]) # Sort for consistency
    dances = []
    print(f'Loading full motion files for ground truth from {dance_folder}...')
    for i, dance_file in enumerate(dance_files):
        if i >= num_dances_to_load: break
        dance_path = os.path.join(dance_folder, dance_file)
        try: dances.append(np.load(dance_path))
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
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to generate (must be <= number of files in data_folder).')
    parser.add_argument('--in_frame_size', type=int, required=True, help='Input frame size (channels) for the model (e.g., 83 for quad: 3 hip + 20 joints*4 quat).')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size of LSTM layers in the model.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--standard_bvh_file', type=str, default='train_data_bvh/standard.bvh', help='Path to the standard BVH file for skeleton hierarchy.')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)
    if not os.path.exists(args.model_path): print(f"Error: Model {args.model_path} not found."); return
    if not os.path.exists(args.data_folder): print(f"Error: Data folder {args.data_folder} not found."); return

    model = acLSTM(in_frame_size=args.in_frame_size, hidden_size=args.hidden_size, out_frame_size=args.in_frame_size)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    # Load skeleton information
    skeleton = None
    ordered_bones_with_rotations = []
    if args.standard_bvh_file and os.path.exists(args.standard_bvh_file):
        print(f"Loading skeleton from: {args.standard_bvh_file}")
        skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(args.standard_bvh_file)
        if skeleton:
            hip_name = 'hip' 
            temp_ordered_bones = []
            if hip_name in skeleton and any('rotation' in ch.lower() for ch in skeleton[hip_name].get('channels',[])):
                temp_ordered_bones.append(hip_name)
            for bone in non_end_bones:
                if bone != hip_name and bone in skeleton and any('rotation' in ch.lower() for ch in skeleton[bone].get('channels',[])):
                    temp_ordered_bones.append(bone)
            ordered_bones_with_rotations = temp_ordered_bones
            print(f"Order of bones for rotation processing: {ordered_bones_with_rotations}")
            if not ordered_bones_with_rotations:
                print("Warning: No bones with rotations found in skeleton. BVH saving might be incorrect or incomplete.")
        else:
            print(f"Warning: Could not load skeleton from {args.standard_bvh_file}. BVH saving will be incorrect.")
            skeleton = None # Ensure it's None
            ordered_bones_with_rotations = []
    else:
        print(f"Warning: Standard BVH file '{args.standard_bvh_file}' not found or not provided. BVH saving will be incorrect.")
        skeleton = None
        ordered_bones_with_rotations = []

    # Load dance data for initial sequences and ground truth
    full_dances = load_full_dance_data(args.data_folder, args.batch_size)
    if not full_dances or len(full_dances) < args.batch_size:
        print(f"Error: Need at least {args.batch_size} dances in {args.data_folder}, found {len(full_dances)}."); return

    initial_seq_batch_list, initial_seq_start_indices_list = [], []
    for i in range(args.batch_size):
        dance = full_dances[i]
        min_len_needed = args.num_initial_frames + args.num_eval_frames
        if dance.shape[0] < min_len_needed:
            print(f"Warning: Dance {i} too short ({dance.shape[0]} frames) for {min_len_needed} total frames. Skipping."); continue
        max_start_idx = dance.shape[0] - min_len_needed
        start_idx = random.randint(0, max_start_idx)
        initial_seq_batch_list.append(dance[start_idx : start_idx + args.num_initial_frames, :])
        initial_seq_start_indices_list.append(start_idx)
    
    if not initial_seq_batch_list: print("Error: No initial sequences created."); return
    current_batch_size = len(initial_seq_batch_list)
    initial_seq_batch_np = np.array(initial_seq_batch_list)
    relevant_full_dances_np = np.array([full_dances[i] for i, start_idx in enumerate(initial_seq_start_indices_list) if i < current_batch_size])

    with torch.no_grad():
        generate_and_evaluate_seq(initial_seq_batch_np, relevant_full_dances_np[:current_batch_size], 
                                    initial_seq_start_indices_list, args.num_generate_frames, model, 
                                    args.output_folder,
                                    skeleton_info=skeleton, # Pass loaded skeleton
                                    ordered_bones_list=ordered_bones_with_rotations, # Pass ordered bones
                                    standard_bvh_file_for_header=args.standard_bvh_file, # Pass path for BVH writing
                                    num_eval_frames=args.num_eval_frames)
    print(f"Motion synthesis finished. Output in {args.output_folder}")

if __name__ == '__main__':
    main()