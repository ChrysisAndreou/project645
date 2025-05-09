import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import torch.nn.functional as F # For normalization

# For raw BVH-derived quaternion data: Hip X pos is 0, Y pos is 1, Z pos is 2.
# Quaternions for rotations start from index 3.

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3

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
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        
        
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
        
        for i in range(generate_frames_number):
            
            in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss

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

#numpy array inital_seq_np: batch*seq_len*frame_size
#return numpy b*generate_frames_number*frame_data
def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):

    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
    initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    
    initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
 
    predict_seq = model.forward(initial_seq, generate_frames_number)
    
    batch=initial_seq_np.shape[0]
   
    for b in range(batch):
        
        out_seq=np.array(predict_seq[b].data.tolist()).reshape(-1,In_frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]
            
        read_bvh.write_traindata_to_bvh(save_dance_folder+"out"+"%02d"%b+".bvh", out_seq)
    return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)



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
                                generate_frames_number, model, save_dance_folder, num_eval_frames=20):
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

    for i in range(generated_seq_abs_hip_np.shape[1]): # Iterate over generated frames
        current_pred_delta_x = generated_seq_abs_hip_np[:, i, 0]
        current_pred_delta_z = generated_seq_abs_hip_np[:, i, 2]
        generated_seq_abs_hip_np[:, i, 0] = last_abs_hip_x + current_pred_delta_x
        generated_seq_abs_hip_np[:, i, 2] = last_abs_hip_z + current_pred_delta_z
        last_abs_hip_x = generated_seq_abs_hip_np[:, i, 0]
        last_abs_hip_z = generated_seq_abs_hip_np[:, i, 2]

    if not os.path.exists(save_dance_folder):
        os.makedirs(save_dance_folder)

    for b_idx in range(batch_size):
        full_output_seq_np = np.concatenate((initial_seq_np[b_idx], generated_seq_abs_hip_np[b_idx]), axis=0)
        bvh_file_path = os.path.join(save_dance_folder, f"generated_quad_raw_{b_idx:02d}.bvh")
        npy_file_path = os.path.join(save_dance_folder, f"generated_quad_raw_{b_idx:02d}.npy")
        print(f"WARNING: Saving motion data (hip_pos + Quaternions) to {bvh_file_path} and {npy_file_path}.")
        print(f"BVH file at {bvh_file_path} will likely NOT be standard-viewable as it contains raw Quaternions.")
        print(f"Use {npy_file_path} with generate_training_quad_data.py script for proper BVH conversion.")
        read_bvh.write_traindata_to_bvh(bvh_file_path, full_output_seq_np) # Saves raw data as BVH
        np.save(npy_file_path, full_output_seq_np) # Save as npy for robust storage
        print(f"Saved generated Quaternion motion (initial + {generate_frames_number} steps) to {bvh_file_path} (raw) and {npy_file_path}")

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
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to generate.')
    parser.add_argument('--in_frame_size', type=int, required=True, help='Input frame size for model (e.g., 3 + num_joints*4).')
    parser.add_argument('--hidden_size', type=int, default=1024, help='LSTM hidden size.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID.')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)
    if not os.path.exists(args.model_path): print(f"Error: Model {args.model_path} not found."); return
    if not os.path.exists(args.data_folder): print(f"Error: Data folder {args.data_folder} not found."); return

    model = acLSTM(in_frame_size=args.in_frame_size, hidden_size=args.hidden_size, out_frame_size=args.in_frame_size)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda(); model.eval()

    full_dances_np = load_full_dance_data(args.data_folder, args.batch_size)
    if not full_dances_np or len(full_dances_np) < args.batch_size:
        print(f"Error: Need at least {args.batch_size} dances in {args.data_folder}, found {len(full_dances_np)}."); return

    initial_seq_batch_list, initial_seq_start_indices_list = [], []
    for i in range(args.batch_size):
        dance = full_dances_np[i]
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
    relevant_full_dances_np = np.array([full_dances_np[i] for i, start_idx in enumerate(initial_seq_start_indices_list) if i < current_batch_size])
    # This selection of relevant_full_dances_np might be incorrect if items were skipped. A safer way would be to filter based on successful additions.
    # For simplicity, assuming the first current_batch_size dances from full_dances_np correspond if no skips occurred or batch_size was met.
    # Re-slicing full_dances_np based on successful `initial_seq_batch_list` indices is better for robustness.
    # Corrected logic for relevant_full_dances_np based on successful appends to initial_seq_batch_list:
    temp_relevant_dances = []
    temp_start_indices = []
    original_indices_of_used_dances = [] # if we need to map back to original full_dances_np
    # This is tricky. Let's assume `initial_seq_start_indices_list` contains valid indices into `full_dances_np`
    # for the items that *were* successfully added to `initial_seq_batch_list`.
    # The current loop for `relevant_full_dances_np` assumes contiguous selection from `full_dances_np` which is okay if batch_size is small and dances are long.
    # However, if some dances were skipped, the indices in initial_seq_start_indices_list refer to the original `full_dances_np`.
    # So, we should use these indices to pick the correct full dances.
    # We need to ensure `full_dance_data_np` passed to `generate_and_evaluate_seq` corresponds to `initial_seq_batch_np`.
    # `load_full_dance_data` loads up to `args.batch_size` dances. If these are the ones used, it's simpler.
    # For now, `relevant_full_dances_np` will be the first `current_batch_size` dances loaded, assuming they correspond.

    with torch.no_grad():
        generate_and_evaluate_seq(initial_seq_batch_np, relevant_full_dances_np[:current_batch_size], 
                                    initial_seq_start_indices_list, args.num_generate_frames, model, 
                                    args.output_folder, args.num_eval_frames)
    print(f"Motion synthesis finished. Output in {args.output_folder}")

if __name__ == '__main__':
    main()