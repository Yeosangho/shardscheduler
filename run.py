import subprocess
import os, csv
import math
import argparse
import datetime
import traceback 

import torch 
import torch.nn as nn
import torch.distributed as dist

from algo import read_profile_info

def make_bucket_list(alpha, beta, comp_ops, total_param_num, world_size):
    bucket_size_list = []
    min_bucket_size = 100000
    possible_overlappable_parameter_num = 0
    #case 1 : consider RS/AG cases
    for comp in comp_ops :

        if(comp.overlappable_time - alpha > 0):

            bucket_size = (comp.overlappable_time - alpha)/beta * ((world_size+1)/world_size)
            if(min_bucket_size < bucket_size ):
                print(bucket_size)
                bucket_size_list.append(math.ceil(bucket_size))
        else:
            print("latency is over!!")
    #case 2 : consider all reduce cases 
    for comp in comp_ops : 
        if(comp.overlappable_time - alpha > 0):
            bucket_size = (comp.overlappable_time - alpha)/(beta*2) * ((world_size+1)/world_size)
            possible_overlappable_parameter_num += math.ceil(bucket_size /4)
            if(min_bucket_size < bucket_size ):
                print(bucket_size)
                bucket_size_list.append(math.ceil(bucket_size))                
        else:
            print("latency is over!! AR")
    bucket_size_list = sorted(bucket_size_list) 
    #max_overlappable_param_num = math.ceil(bucket_size_list[-1] / 4)
    max_overlappable_param_size = math.ceil(bucket_size_list[-1] *((world_size+1)/world_size))
    #case 3 : consider residual parameters which can not overlapped with computation
    residual_param_num = total_param_num - possible_overlappable_parameter_num
    residual_param_size = residual_param_num*4 *((world_size+1)/world_size)
    idx = 1
    while True :
        buffer_for_residual = math.ceil(residual_param_size/idx)
        print(max_overlappable_param_size)
        if(buffer_for_residual > max_overlappable_param_size):
            bucket_size_list.append(buffer_for_residual)
        else:
            break
        idx += 1

    return sorted(bucket_size_list)
	

def make_static_bucket_list(step=1000000):
    bucket_size_list = []
    for i in range(1, 101):
        bucket_size_list.append(i*step)
    return bucket_size_list

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
os.environ['MASTER_PORT'] = os.environ['HANDLER_PORT']
parser = argparse.ArgumentParser()
parser.add_argument('--python_path',  default='', type=str)

args = parser.parse_args()
#profile
#try:
#    a = subprocess.check_output([args.python_path, 'profile.py', '--rank', str(args.rank)])      
#except subprocess.CalledProcessError as e:
#    print(e.output)


    
#for comm in ordered_comm_ops:
#    print(comm)

#3.1 AG -> RS has same priority -> select AG if current layer ops is in forward ops, else if select RS  


#set_bucket size list 
comp_ops = []
param_nums = {}
forward_ops = []
backward_ops = []
layer_bench_file_name = 'profile_data/layer_bench_resnet50_cas_v100_4_node2.csv'
alpha, beta, total_comp_times, total_backward_times, total_forward_times, total_param_num, total_layer_num = read_profile_info(comp_ops, forward_ops, backward_ops, param_nums, layer_bench_file_name)

bucket_list = make_bucket_list(alpha, beta, comp_ops, total_param_num, world_size)
print(bucket_list)
bucket_list = make_static_bucket_list()


##training
#dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)
#proc_exec = True
#target_mem = 0.485
#flag_tensor = torch.ones((1))
#sdp_ratio = 0.0
#fsdp_ratio = 0.0
#dp_ratio = 1.0
#bucket_size = bucket_list[0] / (1024*1024)
#bucket_idx = 0 
#now = datetime.datetime.now()
#dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
#proc = None
#out = None
#while True :
#    try:
#        proc = subprocess.Popen([args.python_path, 'main.py',  
#        	'--sdp_ratio',  str(sdp_ratio),
#        	'--fsdp_ratio', str(fsdp_ratio),
#        	'--dp_ratio', str(dp_ratio),
#        	'--bucket_size', str(bucket_size),
#        	'--target_memory', str(target_mem),
#        	'--exp_tag', dt_string
#        	], stdout=subprocess.PIPE)   
#        print(f'waiting proc')
#        out = proc.communicate()
#        if(int(proc.returncode) == 1 ):
#            raise Exception()
#        flag_tensor = torch.ones((1))
#
#        dist.all_reduce(flag_tensor)
#        #process result
#        print(f"process result {flag_tensor}")
#
#         
#    except Exception as e:
#        
#        flag_tensor = torch.zeros((1))
#
#        dist.all_reduce(flag_tensor)
#        print(f"process result {flag_tensor}")      
#        if(flag_tensor.item() != 0):
#            with open(f'log_handler_{dt_string}.txt', 'a') as f:
#                f.write(str(e))
#                f.write(str(out[1]))
#        else:
#            with open(f'log_handler2_{dt_string}.txt', 'a') as f:
#                f.write(str(e))
#                f.write(str(out[1]))
#       
#    finally:
#        if(flag_tensor.item() == 0):
#            sdp_ratio += 0.05
#            dp_ratio -= 0.05
#
#            sdp_ratio = round(sdp_ratio, 2)
#            dp_ratio = round(dp_ratio, 2)
#            if(sdp_ratio > 1.0):
#                os._exit(0)
#        elif(flag_tensor.item() == world_size):
#            bucket_idx += 1
#            if(bucket_idx >= len(bucket_list)):
#                os._exit(0)
#            bucket_size = bucket_list[bucket_idx] / (1024 * 1024)
#        else:
#            print("retry same case!")
#        print(proc.pid)
#        try:
#            os.kill(proc.pid, 9)
#            print("remain trial is removing!")
#        except:
#            print("trial already removed!")