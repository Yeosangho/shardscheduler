import subprocess
import os, csv
import math
import argparse
import datetime

import torch 
import torch.nn as nn
import torch.distributed as dist

from algo import read_profile_info

def make_bucket_list(alpha, beta, comp_ops):
	bucket_size_list = []
	min_bucket_size = 50000
	for comp in comp_ops : 
		if(comp.overlappable_time - alpha > 0):
			bucket_size = (comp.overlappable_time - alpha)/beta
			if(min_bucket_size < bucket_size ):
				bucket_size_list.append(math.ceil(bucket_size))

	return sorted(bucket_size_list)
	

def make_static_bucket_list(step=100000):
    bucket_size_list = []
    for i in range(1, 101):
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
alpha, beta, total_comp_times, total_backward_times, total_forward_times = read_profile_info(comp_ops, forward_ops, backward_ops, param_nums, layer_bench_file_name)

#bucket_list = make_bucket_list(alpha, beta, comp_ops)
bucket_list = make_static_bucket_list()
#print(bucket_list)

#training
dist.init_process_group(backend='gloo', world_size=2, rank=args.rank)
proc_exec = True
target_mem = 0.485
flag_tensor = torch.ones((1))
sdp_ratio = 0.0
fsdp_ratio = 0.0
dp_ratio = 1.0
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
        sdp_ratio += 0.1
        dp_ratio -= 0.1
        if(sdp_ratio > 1.0):
        	os._exit(0)
