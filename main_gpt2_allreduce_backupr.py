
import os, sys
import time, datetime
from multiprocessing import Process, log_to_stderr
import csv
import copy
import traceback
import threading
import argparse
from queue import Queue
import timeit
import numpy as np
import pandas as pd
import random
import gc
import csv
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch 
import torch.distributed as dist

import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch.profiler import profile, record_function, ProfilerActivity


from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup



from auto_wrap_custom import enable_wrap, auto_wrap, wrap
from torch_scheduler import ShardScheduler, get_param_num_by_buffer_size
from test_cases import make_schedule_from_json
from algo import schedule
from logger import write_trial
from memory_hook import register_memory_profiling_hooks


def is_any_completed(handles):

    for handle in handles:
        if handle.is_completed():
            return True
    return False 


def run(health_check_main_proc, health_check_scheduler_thread, health_check_thread_ready, groups, world_size, rank, trial_info):
	thread_name = threading.current_thread().name

	tensor_one = torch.ones(1)
	queue = []
	handles = []
	for i in range(world_size):
		if(i!=rank):		
			handle = dist.broadcast(tensor_one, group=groups[f"{i}"], src=i, async_op=True)
			handles.append(handle)
			#handle.wait()
	while not is_any_completed(handles) :
		
		print(f"wait src from {(rank-1)%world_size}")
		if health_check_main_proc.locked() or health_check_scheduler_thread.locked():
			handle_me = dist.broadcast(tensor_one, group=groups[f"{rank}"], src=rank, async_op=True)
			while not handle_me.is_completed() :
				print(f"broadcast {rank}")
				time.sleep(0.5)
			break
		time.sleep(0.5)
		if health_check_thread_ready.locked():
			health_check_thread_ready.release()	
	print("!!!!!!!!!!! run with exception")

	trial_info["time"] = -1
	write_trial(trial_info)	
	os._exit(1)


def module_check(module):

	total_wrapped_layers = 0
	for name, child in module.named_children():
		count = module_check(child)
		total_wrapped_layers += count
	if (len(list(module.children())) == 0 ):
		return 1
	else :
		return total_wrapped_layers
	return total_wrapped_layers

def get_reviews(review_path="/scratch/hpc72a03/review_dataset/Reviews.csv"):
	df = pd.read_csv (review_path)  
	df = df[:600]
	print(df)
	print(len(df))
	df.dropna(inplace=True)
	reviews = df.Text.copy() 
	return reviews

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 

class Trainer:
	def __init__(self, world_size, rank,  bucket_size, count, adaptive_shard_ratio,  health_check_scheduler_thread, health_check_main_proc, health_check_thread_ready, trial_info, thread):
		self.health_check_scheduler_thread = health_check_scheduler_thread
		self.health_check_main_proc = health_check_main_proc
		self.health_check_thread_ready = health_check_thread_ready
		self.train_continue = True 
		self.trial_info = trial_info
		self.world_size = world_size
		self.bucket_size = bucket_size
		print(f'world_size : {world_size}')

		self.rank = rank
		
		print(f'rank : {rank}')


		self.process_groups = []
		world_list = [x for x in range(world_size) ]


		while health_check_thread_ready.locked() :
			print("main")
			time.sleep(0.5)

		self.batch_size = 2
		#self.model = models.resnet101()



		self._locks = {}
		self._conditions = {} 

		self._rs_locks = {}
		self._ag_locks = {}
		self._ar_locks = {}
		self._ag_fsdp_locks = {}

		self._rs_conditions = {}
		self._ag_conditions = {}
		self._ar_conditions = {}
		self._ag_fsdp_conditions = {} 


		self._forward_locks = {}
		self._backward_locks = {}

		self._forward_conditions = {}
		self._backward_conditions = {}

		#check lazy init
		self._lazy_init_locks = {}
		self._lazy_init_conditions = {}

		self._partition_counts = {}
		self._scheduled_comms = []
		self._schedule_comm_init = []
		self._done_counts = {}

		self.model_parameter_names = {}
		

		self.datasets = []
		self.target = None
		self.data_index = 0
		self.profile_target_layer = []


		print(f"before init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 
		
		self.reviews = get_reviews()
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
		self.train_dataset = GPT2Dataset(self.reviews, self.tokenizer)
		train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=False)
		self.train_loader = torch.utils.data.DataLoader(
		    self.train_dataset , batch_size=self.batch_size, sampler=train_sampler, shuffle=False, num_workers=2)
		print(f"after init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 

		print(f"before init model  {torch.cuda.memory_allocated() / 1024 /1024}") 
		self.configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
		self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.configuration)
		self.model.resize_token_embeddings(len(self.tokenizer))
		self.model.cuda()
		print(f"after init model  {torch.cuda.memory_allocated() / 1024 /1024}") 


		#summary(self.model, ( 3, 32, 32))
		self.profiled_memory_utilization = []

		ngpus_per_node = torch.cuda.device_count()

		device_id = rank%ngpus_per_node		
		self.comm_stream = torch.cuda.Stream(device_id)

		self.wrap_params = dict( mixed_precision=False, flatten_parameters=True, 

								locks=self._locks,
								health_check_main_proc=self.health_check_main_proc, 

								conditions=self._conditions, 

								profile_layer = self.profile_target_layer,

								memory_record=self.profiled_memory_utilization,
								
								model_parameter_names=self.model_parameter_names
								)
		self.memory_hook_params = dict(memory_record=self.profiled_memory_utilization)
					


		self.sharded_module = None
		self.optimizer = None
		self.criterion = None
		self.partition_threshold = 20000
		
		len_module = module_check(self.model) 

		if(count <= len_module):
			dp_num = len_module - count
			sdp_num = count
			fsdp_num = 0
		else:
			dp_num = 0
			sdp_num = 2*len_module - count
			fsdp_num = count - len_module

		fsdp_num =  int(len_module * adaptive_shard_ratio['fsdp'])
		sdp_num =  int(len_module * adaptive_shard_ratio['sdp'])
		dp_num = int(len_module) -fsdp_num - sdp_num
		adaptive_sdp = {}
		adaptive_sdp['FSDP'] = fsdp_num
		adaptive_sdp['DP'] = dp_num
		adaptive_sdp['SDP'] = sdp_num 
		#register_memory_profiling_hooks(self.model, "")
		'''
		with enable_wrap(**self.wrap_params):
			self.sharded_module = auto_wrap(adaptive_sdp, self.model)
			print(len(list(self.sharded_module.named_parameters())))
			self.adaptive_sdp_modules = {}
			self.adaptive_sdp_modules['FSDP'] = 0 
			self.adaptive_sdp_modules['SDP'] = 0
			self.adaptive_sdp_modules['DP'] = 0

			for n, p in self.sharded_module.named_parameters():
				print(n)
				if('_fsdp_wrapped_module' in n):
					self.adaptive_sdp_modules['FSDP'] += 1
				elif('_sdp_wrapped_module' in n):
					self.adaptive_sdp_modules['SDP'] += 1
				elif('_dp_wrapped_module' in n):
					self.adaptive_sdp_modules['DP'] += 1


			for n, p in self.sharded_module.named_parameters():
				#print(n)
				self._partition_counts[p] = (p.numel() // self.partition_threshold) + 1
				self._done_counts[p] = 0

				self._rs_locks[p] = threading.Lock()
				self._ag_locks[p] = threading.Lock()
				self._ar_locks[p] = threading.Lock()
				self._ag_fsdp_locks[p] = threading.Lock()

				self._forward_locks[p] = threading.Lock()
				self._backward_locks[p] = threading.Lock()

				self._forward_locks[p].acquire()

				self._backward_locks[p].acquire()

				self._rs_conditions[p] = threading.Condition(threading.Lock())
				self._ag_conditions[p] = threading.Condition(threading.Lock())
				self._ar_conditions[p] = threading.Condition(threading.Lock())
				self._ag_fsdp_conditions[p] = threading.Condition(threading.Lock())

				self._forward_conditions[p] = threading.Condition(threading.Lock())
				self._backward_conditions[p] = threading.Condition(threading.Lock())

				self._lazy_init_locks[p] = threading.Lock()
				self._lazy_init_conditions[p] = threading.Condition(threading.Lock())

				self.model_parameter_names[p] = n

			self._locks["FW"] 	    = self._forward_locks
			self._locks["BW"]	    = self._backward_locks 
			self._locks["AG"] 		= self._ag_locks
			self._locks["AR"]		= self._ar_locks
			self._locks["FWTOBW"]   = threading.Lock()
			self._locks["BWTOFW"]   = threading.Lock()
			self._locks["BWTOFW"].acquire()


			self._conditions["FW"]        = self._forward_conditions    
			self._conditions["BW"]        = self._backward_conditions    
			self._conditions["AG"]        = self._ag_conditions  
			self._conditions["AR"]		  = self._ar_conditions
			self._conditions["FWTOBW"]   = threading.Condition(threading.Lock())
			self._conditions["BWTOFW"]   = threading.Condition(threading.Lock())



		params_list = list(self.sharded_module.parameters())
		self.profile_target_layer.append(params_list[20])
		max_param_num = get_param_num_by_buffer_size(self.world_size, self.bucket_size)
		schedule(self.adaptive_sdp_modules, max_param_num, \
			layer_bench_file_name='profile_data/layer_bench_gpt2_cas_v100_4_2.csv', net_bench_file_name='profile_data/net_bench_cas_v100_4_2.csv')
		dist.barrier()

		make_schedule_from_json(params_list, self._schedule_comm_init, self._scheduled_comms, self._locks, self.adaptive_sdp_modules)

		dist.barrier()
		print(f"before init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
		'''
		#self.optimizer = torch.optim.SGD(self.sharded_module.parameters() , lr=0.001, momentum=0.9, nesterov=True)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		#self.optimizer = ShardScheduler(self.sharded_module, self.sharded_module.named_parameters(), self.world_size, self.rank, self.optimizer,
		#                                self.partition_threshold, self._done_counts, self._partition_counts,
		#								self.health_check_scheduler_thread,
		#								self.health_check_thread_ready,
		#								self.trial_info,
		#								thread,
		#								self._locks,
#
		#								self._conditions,
#
		#								self.profile_target_layer, 
		#								self.bucket_size,
		#								10**6, self.comm_stream, self._schedule_comm_init, self._scheduled_comms)
		print(f"after init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
		
		self.criterion = nn.CrossEntropyLoss()

		self.scaler = GradScaler()
		print("end inittialization trainer")

	def benchmark_step(self):

		count = 0
		start = time.time()
		self.model.train()
		for batch_idx, batch in tqdm(self.train_loader):

			b_input_ids = batch[0].cuda()
			b_labels = batch[0].cuda()
			b_masks = batch[1].cuda()





			output = self.model( b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )


			loss = output[0] 
	#	
			loss.backward()
			print("!!!")
			start_time = time.time()
			for n,p in self.model.named_parameters():

				grad_clone = torch.clone(p.grad.data)
				dist.all_reduce(grad_clone, async_op=False)
	
				p.grad.data.copy_(grad_clone)

			self.optimizer.step()
			print("comm time @@@@@@@@@@@@@@@@@@@@")
			f = open("allreduce time.txt", "a+")
			print(f"{time.time() - start_time}\n", file=f)
			f.close()
			if(not self.train_continue):
				break
			count += 1
		execution_time = time.time() -start
		trial_info["time"] = time.time() - start
		write_trial(trial_info)			
		os._exit(0)


	def release_all_lock(self):
		for n, p in self.sharded_module.named_parameters():
			self._release_lock(self._locks['AG'][p], self._conditions['AG'][p])      
		for n, p in self.sharded_module.named_parameters():
			self._release_lock(self._locks['AR'][p], self._conditions['AR'][p])
		for n, p in self.sharded_module.named_parameters():
			self._release_lock(self._locks['FW'][p], self._conditions['FW'][p])     	  
		for n, p in self.sharded_module.named_parameters():
			self._release_lock(self._locks['BW'][p], self._conditions['BW'][p])     	
		if self._locks['BWTOFW'].locked():   
			self._release_lock(self._locks['BWTOFW'], self._conditions['BWTOFW'])	
		if self._locks['FWTOBW'].locked():   
			self._release_lock(self._locks['FWTOBW'], self._conditions['FWTOBW'])			

	def _acquire_lock(self, lock):
		lock.acquire()    	
	def _wait_lock(self, lock, condition):
	    if lock.locked():
	        None
	    else :
	        with condition :
	            condition.wait()
	
	def _release_lock(self, lock, condition):
		if lock.locked():
			lock.release()
		with condition :
			condition.notify_all()	
	def wait_unlock(self, lock, condition):
	    if not lock.locked():
	        None 
	    else :
	        with condition :
	            condition.wait()	
	


	
if __name__ == '__main__':
	world_size = int(os.environ["WORLD_SIZE"])
	rank = int(os.environ["SLURM_PROCID"])
	os.environ['MASTER_PORT'] = os.environ['TRAINER_PORT']
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--target_memory', default=7.0, type=float)
	parser.add_argument('--sdp_ratio', default=0, type=float)
	parser.add_argument('--fsdp_ratio', default=0, type=float)
	parser.add_argument('--dp_ratio', default=0, type=float)
	parser.add_argument('--bucket_size', default=20, type=float)
	parser.add_argument('--exp_tag', type=str)

	args = parser.parse_args()
		#try :
	bucket_size = args.bucket_size
	adaptive_shard_ratio = {}
	adaptive_shard_ratio['dp'] = args.dp_ratio
	adaptive_shard_ratio['sdp'] = args.sdp_ratio
	adaptive_shard_ratio['fsdp'] = args.fsdp_ratio

	exp_tag = ''
	if(args.exp_tag is None):
		now = datetime.datetime.now()
		dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
		exp_tag = dt_string
	else:
		exp_tag = args.exp_tag

	torch.cuda.empty_cache()
	gc.collect()
	
	ngpus_per_node = torch.cuda.device_count()

	device_id = rank%ngpus_per_node
	total_memory = torch.cuda.get_device_properties(device_id).total_memory
	target_memory = args.target_memory  *1024 * 1024 * 1024
	fraction = target_memory  /  total_memory

	torch.cuda.set_device(device_id)	
	torch.cuda.set_per_process_memory_fraction(fraction, device_id)    	
	parameter_num = int(2.0 * 1024 * 1024 * 1024 / 4)
	count = 0
	dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
	
	health_check_main_proc = threading.Lock()
	health_check_scheduler_thread = threading.Lock()
	health_check_thread_ready = threading.Lock()
	#health_check_thread_ready.acquire()

	trial_info = {}
	trial_info["exp_tag"] = exp_tag
	trial_info["bucket_size"] = bucket_size
	trial_info['sdp'] = adaptive_shard_ratio['sdp']
	trial_info['dp'] = adaptive_shard_ratio['dp']
	trial_info["fsdp"] = adaptive_shard_ratio['fsdp']

	try :

		groups = {}
		for i in range(world_size):
			for j in range(world_size):
				if(i != j):
					proc_list = list(range(world_size))
					group = dist.new_group(proc_list, backend='gloo')
					groups[f'{i}'] = group

		thread = threading.Thread(target=run, args=(health_check_main_proc, health_check_scheduler_thread, health_check_thread_ready, groups, world_size, rank, trial_info))
		thread.daemon = True
		#thread.start()	

		trainer = Trainer(world_size, rank, bucket_size, count, adaptive_shard_ratio, health_check_scheduler_thread, health_check_main_proc, health_check_thread_ready, trial_info, thread) 
			
		trainer.benchmark_step()

	except RuntimeError as error :
		print("line 550 in main.py")
		print(traceback.format_exc())
		with open(f'log_{exp_tag}.txt', 'a') as f:
			f.write(str(error))
			f.write(traceback.format_exc())
		health_check_main_proc.acquire()
		trial_info["time"] = -1
		write_trial(trial_info)
		thread.join()