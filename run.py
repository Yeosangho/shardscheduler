import subprocess
import os, csv
from collections import OrderedDict
import argparse
import functools
import torch 
import torch.nn as nn
import torch.distributed as dist




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








#training
#dist.init_process_group(backend='gloo', world_size=2, rank=args.rank)
#proc_exec = True
#target_mem = 7.5
#flag_tensor = torch.ones((1))
#
#while proc_exec :
#    try:
#        print(f"start proc {target_mem}")
#        print(flag_tensor)
#        a = subprocess.check_output([args.python_path, 'main.py', '--rank', str(args.rank), '--target_memory', str(target_mem)])      
#        print(f'end proc')
#        flag_tensor = torch.ones((1))
#
#        dist.all_reduce(flag_tensor)
#        #exit()
#    except subprocess.CalledProcessError as e:
#        #print(e.output)
#        flag_tensor = torch.zeros((1))
#
#        dist.all_reduce(flag_tensor)        
#        target_mem += 0.1
#        for line in e.output.splitlines():
#            print(line)
#        proc_exec = True