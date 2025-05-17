import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse

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


#numpy array inital_seq_np: batch*seq_len*frame_size
#return numpy b*generate_frames_number*frame_data
def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder, 
                 dances=None, dance_info=None, quantitative_comparison_len=20):

    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
    initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    
    initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
 
    predict_seq = model.forward(initial_seq, generate_frames_number)
    
    batch=initial_seq_np.shape[0]
   
    for b in range(batch):
        
        # Using .cpu() before .tolist() for safety, as predict_seq is on CUDA.
        # predict_seq contains (initial_seq_len-1) frames from seed + generate_frames_number new frames
        current_predict_seq_np = np.array(predict_seq[b].data.cpu().tolist()).reshape(-1,In_frame_size)
        
        # Perform hip accumulation
        # Make a copy to modify, as this out_seq is used for generation and saving
        out_seq_accumulated = current_predict_seq_np.copy()
        
        # Initialize last_x and last_z from the global hip position of the first frame of the original seed sequence.
        # initial_seq_np[b, 0] is the first frame of the seed for batch item 'b'.
        # Hip_index*3 is the index for hip X, Hip_index*3+2 is for hip Z.
        last_x = initial_seq_np[b, 0, Hip_index*3]
        last_z = initial_seq_np[b, 0, Hip_index*3+2]

        for frame_idx in range(out_seq_accumulated.shape[0]): 
            # Accumulate Hip X
            current_delta_x = out_seq_accumulated[frame_idx, Hip_index*3] # This is the predicted delta_x
            out_seq_accumulated[frame_idx, Hip_index*3] = current_delta_x + last_x
            last_x = out_seq_accumulated[frame_idx, Hip_index*3]
            
            # Accumulate Hip Z
            current_delta_z = out_seq_accumulated[frame_idx, Hip_index*3+2] # This is the predicted delta_z
            out_seq_accumulated[frame_idx, Hip_index*3+2] = current_delta_z + last_z
            last_z = out_seq_accumulated[frame_idx, Hip_index*3+2]
            
            # Hip Y (at Hip_index*3+1) is predicted as an absolute value by the model,
            # so it doesn't need explicit accumulation like X and Z. It's already in out_seq_accumulated.

        # Save the generated motion with accumulated hip positions
        read_bvh.write_traindata_to_bvh(save_dance_folder+"out"+"%02d"%b+".bvh", out_seq_accumulated)

        # Quantitative evaluation for the first batch item if data is available
        if b == 0 and dances is not None and dance_info is not None and quantitative_comparison_len > 0:
            dance_id, seed_start_id_in_dance = dance_info[b]
            
            if dance_id < len(dances):
                selected_full_dance = dances[dance_id]
                seed_len = initial_seq_np.shape[1] # Number of frames in the input seed initial_seq_np[b]
                
                # Ground truth frames for comparison (frames *after* the seed)
                gt_comp_start_frame = seed_start_id_in_dance + seed_len
                gt_comp_end_frame = gt_comp_start_frame + quantitative_comparison_len
                
                if gt_comp_end_frame <= selected_full_dance.shape[0]:
                    gt_frames_for_eval = selected_full_dance[gt_comp_start_frame:gt_comp_end_frame]
                    
                    # Predicted frames (purely generated part, after seed reconstruction)
                    # out_seq_accumulated has (seed_len - 1) reconstructed seed frames + generated_frames_number new frames
                    pred_frames_start_idx = seed_len - 1 
                    pred_frames_end_idx = pred_frames_start_idx + quantitative_comparison_len

                    if pred_frames_end_idx <= out_seq_accumulated.shape[0] and gt_frames_for_eval.shape[0] == quantitative_comparison_len:
                        pred_frames_for_eval = out_seq_accumulated[pred_frames_start_idx:pred_frames_end_idx]
                        
                        mse = np.mean((gt_frames_for_eval - pred_frames_for_eval)**2)
                        print(f"Batch {b} - Quantitative Evaluation - MSE for first {quantitative_comparison_len} generated frames: {mse:.8f}")
                    else:
                        print(f"Batch {b} - Quantitative Evaluation: Not enough predicted frames ({out_seq_accumulated.shape[0] - pred_frames_start_idx}) or GT frames ({gt_frames_for_eval.shape[0]}) to compare {quantitative_comparison_len} frames.")
                else:
                    print(f"Batch {b} - Quantitative Evaluation: Not enough frames in ground truth dance (ID: {dance_id}, Length: {selected_full_dance.shape[0]}) to get {quantitative_comparison_len} frames after seed (ends at {seed_start_id_in_dance + seed_len}).")

        # Save ground truth BVH for the first batch item for comparison
        if b == 0 and dances is not None and dance_info is not None:
            dance_id, start_id = dance_info[b] # start_id is the beginning of the seed in the full dance
            if dance_id < len(dances):
                dance = dances[dance_id].copy()
                # Save ground truth for seed + generated part
                gt_total_len = initial_seq_np.shape[1] + generate_frames_number
                end_id_for_gt_bvh = min(start_id + gt_total_len, dance.shape[0])
                
                if start_id < end_id_for_gt_bvh : # Ensure there's a valid sequence to save
                    gt_seq_to_save = dance[start_id:end_id_for_gt_bvh]
                    read_bvh.write_traindata_to_bvh(save_dance_folder+"gt"+"%02d"%b+".bvh", gt_seq_to_save)
                    print(f"Batch {b} - Saved ground truth sequence (length {gt_seq_to_save.shape[0]}) to {save_dance_folder}gt{b:02d}.bvh")
                else:
                    print(f"Batch {b} - Could not save ground truth: Invalid range or empty sequence (start: {start_id}, end: {end_id_for_gt_bvh}, dance len: {dance.shape[0]})")

    # Return the raw predicted sequence (before hip accumulation) from the model, on CPU
    return np.array(predict_seq.data.cpu().tolist()).reshape(batch, -1, In_frame_size)


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
        # original.py uses: dance=np.load(dance_folder+dance_file)
        # This relies on dance_folder ending with '/' if it's a directory,
        # or paths being structured for simple concatenation.
        # Colab notebook seems to provide paths like "folder_path/" so this should be fine.
        dance=np.load(dance_folder+dance_file)
        print ("frame number: "+ str(dance.shape[0]))
        dances=dances+[dance] # original.py uses list concatenation
    return dances
    
# dances: [dance1, dance2, dance3,....]
# The original test function from synthesize_pos_motion.py is being removed.
# Its logic is incorporated into __main__ if it matches original.py's test logic,
# or superseded by original.py's test logic.
# Current test function (lines 230-240) is deleted.

# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesize motion using a trained acLSTM model (Original Style).")
    parser.add_argument('--read_weight_path', type=str, required=True, help='Path to the trained model weights (.weight file)')
    parser.add_argument('--dances_folder', type=str, required=True, help='Path to the folder containing seed motion .npy files')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to the folder to save generated BVH files')
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Frame rate of the dance data')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to generate in a batch (original.py default was 5)')
    parser.add_argument('--initial_seq_len', type=int, default=15, help='Length of the initial seed motion sequence (frames in original data)')
    parser.add_argument('--generate_frames_number', type=int, default=400, help='Number of frames to synthesize')
    parser.add_argument('--in_frame_size', type=int, default=In_frame_size, help='Input frame size for the model')
    parser.add_argument('--hidden_size', type=int, default=Hidden_size, help='Hidden size of the LSTM layers in the model')
    parser.add_argument('--out_frame_size', type=int, default=In_frame_size, help='Output frame size for the model')
    # quantitative_comparison_len is kept for argparse compatibility with Colab, but not used by original.py's synthesis logic
    parser.add_argument('--quantitative_comparison_len', type=int, default=20, help='Number of initial generated frames to compare (INFO: this arg is not used by the adapted original synthesis logic)')

    args = parser.parse_args()

    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    # Logic adapted from original.py's test function and its main execution context
    torch.cuda.set_device(0) # As in original.py

    model = acLSTM(args.in_frame_size, args.hidden_size, args.out_frame_size)
    model.load_state_dict(torch.load(args.read_weight_path))
    model.cuda()
    # No model.eval(), as per original.py's synthesis part

    dances_list = load_dances(args.dances_folder) 

    if not dances_list:
        print(f"Error: No dance files found in {args.dances_folder} or failed to load. Exiting.")
        exit()
        
    dance_len_lst = get_dance_len_lst(dances_list) # This function is identical
    
    if not dance_len_lst:
        print(f"Error: dance_len_lst is empty (no dances loaded or all dances have zero effective length). Cannot select dances from {args.dances_folder}. Exiting.")
        exit()
    random_range = len(dance_len_lst)
    
    speed = args.dance_frame_rate / 30.0 # original.py uses 30.0

    dance_batch_list = []
    dance_info_list = [] # To store [dance_id, start_id] for each batch item
    
    for b_idx in range(args.batch_size):
        # Randomly pick up one dance based on its length proportion (from get_dance_len_lst)
        dance_list_idx_picker = np.random.randint(0, random_range) # Index into dance_len_lst
        actual_dance_idx = dance_len_lst[dance_list_idx_picker] # Index into dances_list
        
        selected_dance_frames = dances_list[actual_dance_idx].copy()
        num_frames_in_selected_dance = selected_dance_frames.shape[0]
            
        # Determine start_id. original.py's logic can error if dance is too short.
        # int(num_frames_in_selected_dance - args.initial_seq_len * speed - 10)
        # random.randint(10, upper_bound_for_start_id)
        upper_bound_for_start_id = int(num_frames_in_selected_dance - args.initial_seq_len * speed - 10)
        
        if upper_bound_for_start_id < 10 :
            print(f"Error for batch item {b_idx}: Dance {actual_dance_idx} (length {num_frames_in_selected_dance}) is too short. "
                  f"Calculated upper bound for start_id ({upper_bound_for_start_id}) is less than lower bound (10). "
                  f"Original script would error here. Skipping this batch item.")
            continue # Skip this item for the batch

        start_id_raw = random.randint(10, upper_bound_for_start_id)
        
        print(f"Batch {b_idx}: Using dance {actual_dance_idx} (length {num_frames_in_selected_dance} frames), effective start frame {start_id_raw}")
        dance_info_list.append([actual_dance_idx, start_id_raw]) # Store info for this batch item

        current_seed_seq_frames = []
        try:
            for i in range(args.initial_seq_len):
                # original.py's frame indexing: selected_dance_frames[int(i * speed + start_id_raw)]
                # This can cause IndexError if out of bounds. This is preserved.
                idx_to_sample = int(i * speed + start_id_raw)
                frame_data = selected_dance_frames[idx_to_sample]
                current_seed_seq_frames = current_seed_seq_frames + [frame_data] # original.py uses list concatenation
        except IndexError as e:
            print(f"Error for batch item {b_idx} during seed sequence assembly: {e}. "
                  f"Index {idx_to_sample} out of bounds for dance {actual_dance_idx} (length {num_frames_in_selected_dance}). "
                  f"Original script would error here. Skipping this batch item.")
            continue # Skip this item

        if len(current_seed_seq_frames) == args.initial_seq_len:
            dance_batch_list.append(np.array(current_seed_seq_frames))
        else:
            print(f"Warning for batch item {b_idx}: Seed sequence not fully assembled (got {len(current_seed_seq_frames)} frames, expected {args.initial_seq_len}). Skipping.")


    if not dance_batch_list:
        print("Error: No valid seed sequences could be prepared for the batch. Exiting.")
        exit()

    seed_batch_np = np.array(dance_batch_list)
    
    # Call the 'original.py' style generate_seq
    generate_seq(seed_batch_np, args.generate_frames_number, model, args.write_bvh_motion_folder,
                 dances=dances_list, dance_info=dance_info_list, 
                 quantitative_comparison_len=args.quantitative_comparison_len)
    
    print(f"Positional synthesis (Original-style) complete. BVH files saved to: {args.write_bvh_motion_folder}")