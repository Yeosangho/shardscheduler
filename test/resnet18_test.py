import torch 
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


from torchvision import datasets
from torchvision import transforms
import torchvision.models as models

#from torch.profiler import profile, record_function, ProfilerActivity

import os, sys
import time, datetime
from multiprocessing import Process, log_to_stderr
import csv
from gossip_module.utils import flatten_tensors, flatten_tensors_grad, unflatten_tensors, unflatten_tensors_grad
from fsdp_custom import FullyShardedDataParallel as FSDP
from sdp_custom import ShardedDataParallel as SDP
from dp_custom import DataParallel_Custom as DP

from auto_wrap_custom import enable_wrap, auto_wrap, wrap
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch_scheduler import ShardScheduler
import threading
import argparse
from queue import Queue
import timeit
import numpy as np
import copy
import gc
import csv
from test_cases import *
from algo import schedule


def run(health_check_main_proc, health_check_scheduler_thread, group, world_size, rank, trainer):
	thread_name = threading.current_thread().name

	#for i in range(world_size):
	#	if(i != rank):
	#		handle = dist.broadcast(tensor_one, group=group, src=i, async_op=True)
	#		queue.append(handle)
	#handle = queue.pop(0)		
	#while not handle.is_completed() :
	#	queue.append(handle)
	#	time.sleep(0.5)
	tensor_one = torch.ones(1)
	queue = []		
	handle = dist.broadcast(tensor_one, group=groups[f"{rank}:{(rank-1)%world_size}"], src=(rank-1)%world_size, async_op=True)
	#handle.wait()
	while not handle.is_completed() :
		print(f"wait src from {(rank-1)%world_size}")
		if health_check_main_proc.locked() or health_check_scheduler_thread.locked():
			handle_me = dist.broadcast(tensor_one, group=groups[f"{(rank-1)%world_size}:{rank}"], src=rank, async_op=True)
			while not handle_me.is_completed() :
				print(f"broadcast {rank}")
				time.sleep(0.5)
			break
		time.sleep(0.5)	
	print("!!!!!!!!!!! run with exception")
	health_check_main_proc.acquire()
	health_check_scheduler_thread.acquire()
	trainer.train_continue = False
	trainer.optimizer.train_continue = False
	#trainer.release_all_lock()
	os._exit(1)

	print("1")
	#trainer.optimizer.stop()	

	print("2")
	#print("!!!")
	#raise RuntimeError("An error")
	#RuntimeError("An error")
	#raise RuntimeError("other worker should be stopped")

	#sys.exit()	
	#print("3")
	#raise Exception("An error in thread '{}'.".format(thread_name))

def module_check(module):
	#if (len(list(module.children())) == 0 ):
		#print(module)
		#print(module.data.size())
	total_wrapped_layers = 0
	for name, child in module.named_children():
		count = module_check(child)
		total_wrapped_layers += count
	if (len(list(module.children())) == 0 ):
		return 1
	else :
		return total_wrapped_layers
	return total_wrapped_layers


class Trainer:
	def __init__(self, world_size, rank, adaptive_shard_ratio, ):

		#world_size = int(os.environ["WORLD_SIZE"])
		self.world_size = world_size
		print(f'world_size : {world_size}')
		ngpus_per_node = torch.cuda.device_count()
		#self.shard = shard
		#rank = int(os.environ['SLURM_PROCID'])
		self.rank = rank
		
		print(f'rank : {rank}')


		self.device = torch.device("cuda:"  + str(rank%ngpus_per_node))
		torch.cuda.set_device(rank%ngpus_per_node)
		print("cuda:"  + str(rank%ngpus_per_node))
		self.process_groups = []
		world_list = [x for x in range(world_size) ]

		#self.process_groups = []
		#world_list = [x for x in range(world_size) ]
		#for i in range(self.thread_num):
		#    ng = dist.new_group(world_list, backend='gloo')
		#    self.process_groups.append(ng) 

		self.batch_size = 16
		self.image_size = 42
		self.classification_num = 1000
		#self.model = models.resnet101()
		print(f"before init model  {torch.cuda.memory_allocated() / 1024 /1024}") 
		self.model = ResNet(Bottleneck, [3, 4, 6, 3]) #it means "resnet18 model"
		self.model.cuda()

	def benchmark_step(self):

		start = 0
		#for batch_idx, (data, target) in enumerate(self.train_loader):
		for i in range(10):
			data = torch.rand((3,3,32,32))
			data = data.cuda()
			output = self.model(data)

	


	
if __name__ == '__main__':
	os.environ['MASTER_ADDR'] = '210.107.197.219'
	os.environ['MASTER_PORT'] = '30005'
	os.environ["NCCL_SOCKET_IFNAME"]="eno1,eth0"
	parser = argparse.ArgumentParser()
	parser.add_argument('--rank', dest='rank', default=0, type=int)
	parser.add_argument('--target_memory', default=7.0, type=float)
	parser.add_argument('--sdp_ratio', default=0, type=float)
	parser.add_argument('--fsdp_ratio', default=0, type=float)
	parser.add_argument('--dp_ratio', default=0, type=float)


	args = parser.parse_args()
		#try :
	world_size = 2
	rank = args.rank
	adaptive_shard_ratio = {}
	adaptive_shard_ratio['dp'] = args.dp_ratio
	adaptive_shard_ratio['sdp'] = args.sdp_ratio
	adaptive_shard_ratio['fsdp'] = args.fsdp_ratio
	#shard = args.shard
	#mixed_precision = args.mixed_precision 	
	#world_size = int(os.environ["WORLD_SIZE"])
	#rank = int(os.environ['SLURM_PROCID'])	
	
	torch.cuda.empty_cache()
	gc.collect()
	total_memory = torch.cuda.get_device_properties(0).total_memory
	target_memory = args.target_memory  *1024 * 1024 * 1024
	fraction = target_memory  /  total_memory
	torch.cuda.set_per_process_memory_fraction(fraction, 0)    	
	parameter_num = int(2.0 * 1024 * 1024 * 1024 / 4)

	dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
	#run(world_size, rank)
	#comm_stream = torch.cuda.Stream()

	#group = dist.new_group(timeout=datetime.timedelta(seconds=5),)
	trainer = Trainer(world_size, rank, adaptive_shard_ratio) 


		
	trainer.benchmark_step()