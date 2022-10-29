import subprocess
import os, csv
import math
import argparse
import datetime

import torch 
import torch.nn as nn
import torch.distributed as dist

from algo import read_profile_info

def make_bucket_list(alpha, beta, comp_ops, total_param_num):
    bucket_size_list = []
    min_bucket_size = 100000
    possible_overlappable_parameter_num = 0
    #case 1 : consider RS/AG cases
    for comp in comp_ops : 
        if(comp.overlappable_time - alpha > 0):

            bucket_size = (comp.overlappable_time - alpha)/beta
            if(min_bucket_size < bucket_size ):
                bucket_size_list.append(math.ceil(bucket_size))
    #case 2 : consider all reduce cases 
    for comp in comp_ops : 
        if(comp.overlappable_time - alpha > 0):
            bucket_size = (comp.overlappable_time - alpha)/(beta*2)
            possible_overlappable_parameter_num += math.ceil(bucket_size /4)
            if(min_bucket_size < bucket_size ):
                bucket_size_list.append(math.ceil(bucket_size))                

    bucket_size_list = sorted(bucket_size_list)
    max_overlappable_param_num = math.ceil(bucket_size_list[-1] / 4)

    #case 3 : consider residual parameters which can not overlapped with computation
    residual_param_num = total_param_num - possible_overlappable_parameter_num
    idx = 1
    while True :
        buffer_for_residual = math.ceil(residual_param_num / idx)
        if(buffer_for_residual > max_overlappable_param_num):
            bucket_size_list.append(buffer_for_residual)
        else:
            break
        idx += 1

    return sorted(bucket_size_list)
	

def make_static_bucket_list(step=1000000):
    bucket_size_list = []
    for i in range(1, 51):
        bucket_size_list.append(i*step)
    return bucket_size_list

parser = argparse.ArgumentParser()
parser.add_argument('--rank', dest='rank', default=0, type=int)
parser.add_argument('--python_path',  default='', type=str)

args = parser.parse_args()
os.environ['MASTER_ADDR'] = '210.107.197.219'
os.environ['MASTER_PORT'] = '30001'
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
layer_bench_file_name = 'layer_bench.csv'
alpha, beta, total_comp_times, total_backward_times, total_forward_times, total_param_num, total_layer_num = read_profile_info(comp_ops, forward_ops, backward_ops, param_nums, layer_bench_file_name)

bucket_list = make_bucket_list(alpha, beta, comp_ops, total_param_num)
print(bucket_list)
bucket_list = make_static_bucket_list()


#training
world_size = 2
dist.init_process_group(backend='gloo', world_size=world_size, rank=args.rank)
proc_exec = True
target_mem = 0.4
flag_tensor = torch.ones((1))
sdp_ratio = 1.0
fsdp_ratio = 0.0
dp_ratio = 0.0
bucket_size = bucket_list[0] / (1024*1024)
bucket_idx = 0 
now = datetime.datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

while True :
    try:
        print(f"start proc {target_mem}")
        print(flag_tensor)
        a = subprocess.check_output([args.python_path, 'main.py', 
        	'--rank', str(args.rank), 
        	'--sdp_ratio',  str(sdp_ratio),
        	'--fsdp_ratio', str(fsdp_ratio),
        	'--dp_ratio', str(dp_ratio),
        	'--bucket_size', str(bucket_size),
        	'--target_memory', str(target_mem),
        	'--exp_tag', dt_string
        	])      
        print(f'end proc')
        flag_tensor = torch.ones((1))

        dist.all_reduce(flag_tensor)
        #process result
        print(f"process result {flag_tensor}")
        #mem error is not occured !! -> no more sharding!! + it can increase bucket size more!!
        bucket_idx += 1
        if(bucket_idx >= len(bucket_list)):
        	os._exit(0)
        bucket_size = bucket_list[bucket_idx] / (1024 * 1024)
         
    except subprocess.CalledProcessError as e:
        #print(e.output)
        flag_tensor = torch.zeros((1))

        dist.all_reduce(flag_tensor)
        print(f"process result {flag_tensor}")      
        #mem error occured !!! -> more sharding!!!
        if(flag_tensor.item() == 0):
            fsdp_ratio += 0.05
            sdp_ratio -= 0.05
            if(fsdp_ratio > 1.0):
            	os._exit(0)
        else:
            bucket_idx += 1
            if(bucket_idx >= len(bucket_list)):
                os._exit(0)
            bucket_size = bucket_list[bucket_idx] / (1024 * 1024)