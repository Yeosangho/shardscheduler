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

from torch.profiler import profile, record_function, ProfilerActivity

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
	def __init__(self, world_size, rank,  count, health_check_scheduler_thread, health_check_main_proc):
		self.health_check_scheduler_thread = health_check_scheduler_thread
		self.health_check_main_proc = health_check_main_proc
		self.train_continue = True 
		torch.backends.cudnn.benchmark = True
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

		print(f"after init model  {torch.cuda.memory_allocated() / 1024 /1024}") 

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
		
		#for _ in range(100):
		#    data = torch.rand(self.batch_size, 3, 80, 80)
		#    self.target = torch.LongTensor(self.batch_size).random_() % 1000
		#    data, self.target = data.cuda(), self.target.cuda()
		#    self.datasets.append(data)

		#self.train_dataset = datasets.MNIST(root='data', 
		#                               train=True, 
		#                               transform=transforms.ToTensor(),
		#                               download=True)
#
		#self.train_loader = DataLoader(dataset=self.train_dataset, 
		#                          batch_size=128, 
		#                          shuffle=True)	
		self.train_dataset = datasets.CIFAR10(
		    root='cifar10-data', train=True, download=True, transform=transforms.ToTensor())
		self.train_loader = torch.utils.data.DataLoader(
		    self.train_dataset , batch_size=32, shuffle=True, num_workers=2)
		print(f"after init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 

		#summary(self.model, ( 3, 32, 32))
		self.profiled_memory_utilization = []

		print("111")
		self.comm_stream = torch.cuda.Stream()
		print("222")

		self.wrap_params = dict( mixed_precision=False, flatten_parameters=True, 
								done_counts=self._done_counts, partition_counts=self._partition_counts, 

								locks=self._locks,
								health_check_main_proc=self.health_check_main_proc, 

								conditions=self._conditions, 

								profile_layer = self.profile_target_layer,

								init_schedule=self._schedule_comm_init, 

								schedule=self._scheduled_comms,

								memory_record=self.profiled_memory_utilization, comm_stream=self.comm_stream,
								
								model_parameter_names=self.model_parameter_names
								)

					

		print("333")

		#self.fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=False, flatten_parameters=False,  memory_record=self.profiled_memory_utilization)
		self.sharded_module = None
		self.optimizer = None
		self.criterion = None
		self.partition_threshold = 20000
		
		len_module = module_check(self.model) 
		print("444")

		if(count <= len_module):
			dp_num = len_module - count
			sdp_num = count
			fsdp_num = 0
		else:
			dp_num = 0
			sdp_num = 2*len_module - count
			fsdp_num = count - len_module

		fsdp_num = 2
		sdp_num =  int(len_module) - 4
		dp_num = 2
		adaptive_sdp = {}
		adaptive_sdp['FSDP'] = fsdp_num
		adaptive_sdp['DP'] = dp_num
		adaptive_sdp['SDP'] = sdp_num

		schedule(adaptive_sdp)
		
		with enable_wrap(**self.wrap_params):
			self.sharded_module = auto_wrap(adaptive_sdp, self.model)
			print(len(list(self.sharded_module.named_parameters())))
			adaptive_sdp_modules = {}
			adaptive_sdp_modules['FSDP'] = 0 
			adaptive_sdp_modules['SDP'] = 0
			adaptive_sdp_modules['DP'] = 0

			for n, p in self.sharded_module.named_parameters():
				print(n)
				if('_fsdp_wrapped_module' in n):
					adaptive_sdp_modules['FSDP'] += 1
				elif('_sdp_wrapped_module' in n):
					adaptive_sdp_modules['SDP'] += 1
				elif('_dp_wrapped_module' in n):
					adaptive_sdp_modules['DP'] += 1


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
				#self._rs_locks[p].acquire()
				#self._ag_fsdp_locks[p].acquire()
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
			#self._locks["AGFSDP"]   = self._ag_fsdp_locks
			#self._locks["RS"]       = self._rs_locks

			self._conditions["FW"]        = self._forward_conditions    
			self._conditions["BW"]        = self._backward_conditions    
			self._conditions["AG"]        = self._ag_conditions  
			self._conditions["AR"]		  = self._ar_conditions
			self._conditions["FWTOBW"]   = threading.Condition(threading.Lock())
			self._conditions["BWTOFW"]   = threading.Condition(threading.Lock())

			#self._conditions["AGFSDP"]    = self._ag_fsdp_conditions       
			#self._conditions["RS"]        = self._rs_conditions      

		params_list = list(self.sharded_module.parameters())

		self.profile_target_layer.append(params_list[20])
		#make_schedules_adaptive_sdp_auto(params_list, self._schedule_comm_init, self._scheduled_comms, self._locks, adaptive_sdp_modules)
		make_schedule_from_json(params_list, self._schedule_comm_init, self._scheduled_comms, self._locks, adaptive_sdp_modules)
		#make_schedule_wfbp_sdp(params_list, self._schedule_comm_init, self._scheduled_comms, self._locks)
		os._exit(1)
		print(f"before init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
		#self.optimizer = torch.optim.SGD(self.sharded_module.parameters() , lr=0.001, momentum=0.9, nesterov=True)
		self.optimizer = torch.optim.Adam(self.sharded_module.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		self.optimizer = ShardScheduler(self.sharded_module, self.sharded_module.named_parameters(), self.world_size, self.rank, self.optimizer,
		                                self.partition_threshold, self._done_counts, self._partition_counts,
										self.health_check_scheduler_thread,
										self._locks,

										self._conditions,

										self.profile_target_layer, 

										10**6, self.comm_stream, self._schedule_comm_init, self._scheduled_comms)
		print(f"after init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
		
		self.criterion = nn.CrossEntropyLoss()

		#if(wftp == True):
		#	self._register_hooks()
		self.scaler = GradScaler()
		print("end inittialization trainer")

	def benchmark_step(self):
		print("bench 1")
		with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
			print("bench 2")

			with record_function("model_training"):
				#data = self.datasets[self.data_index%len(self.datasets)]
				count = 0
				print("bench 3")
				start = 0
				for batch_idx, (data, target) in enumerate(self.train_loader):
					if(count == 5):
						start = time.time()
					self.data_index += 1
					data = data.cuda()
					print("bench 4")
					print(f"target : {target.shape} {target.type()}")
					print(f"data : {data.shape}")

					target = target.cuda()

					print(f"before forward  {torch.cuda.memory_allocated() / 1024 /1024}") 
					if self._locks['BWTOFW'].locked():   
						self._release_lock(self._locks['BWTOFW'], self._conditions['BWTOFW'])				
					output = self.sharded_module(data)

					if self._locks['FWTOBW'].locked():   
						self._release_lock(self._locks['FWTOBW'], self._conditions['FWTOBW'])

					print(f"after forward  {torch.cuda.memory_allocated() / 1024 /1024}") 
					print(output.sum())
					loss = self.criterion(output, target)
					print(loss)
					print(f"before backward  {torch.cuda.memory_allocated() / 1024 /1024}") 
	#		
					loss.backward()
					if self._locks['BWTOFW'].locked():   
						self._release_lock(self._locks['BWTOFW'], self._conditions['BWTOFW'])

					print(f"after backward  {torch.cuda.memory_allocated() / 1024 /1024}") 
					if(not self.train_continue):
						break
					count += 1
					if(count == 10):
						break
			#torch.cuda.synchronize()
			print(time.time() -start)
			print("1111")
			if(self.health_check_scheduler_thread.locked()):
				raise RuntimeError("Thread Runtime Error!")
		#self.release_all_lock()
		#self.optimizer.train_continue = False
		#self.optimizer.stop()
		prof.export_chrome_trace("trace_algo.json")


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
	def _wait_unlock(self, lock, condition):
	    if not lock.locked():
	        None 
	    else :
	        with condition :
	            condition.wait()	
	


	
if __name__ == '__main__':
	os.environ['MASTER_ADDR'] = '210.107.197.219'
	os.environ['MASTER_PORT'] = '30004'
	parser = argparse.ArgumentParser()
	parser.add_argument('--rank', dest='rank', default=0, type=int)
	parser.add_argument('--target_memory', default=7.0, type=float)

	args = parser.parse_args()
		#try :
	world_size = 2
	rank = args.rank
	#shard = args.shard
	#mixed_precision = args.mixed_precision 	
	#world_size = int(os.environ["WORLD_SIZE"])
	#rank = int(os.environ['SLURM_PROCID'])	
	
	torch.cuda.empty_cache()
	gc.collect()
	total_memory = torch.cuda.get_device_properties(0).total_memory
	target_memory = args.target_memory  *1024 * 1024 * 1024
	fraction = target_memory  /  total_memory
	print(fraction)
	torch.cuda.set_per_process_memory_fraction(fraction, 0)    	
	parameter_num = int(2.0 * 1024 * 1024 * 1024 / 4)
	mem_tensor = torch.zeros(parameter_num).cuda()
	count = 0
	dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

	#dist.barrier()
	torch.cuda.max_split_size_mb  = 512
	a = torch.cuda.memory_allocated(0)
	print(f"!!!!!!!!!!!!!!!! {torch.cuda.memory_reserved()}")
	#print(torch.cuda.memory_stats())	
	health_check_main_proc = threading.Lock()
	health_check_scheduler_thread = threading.Lock()
	try :
		#run(world_size, rank)
		#comm_stream = torch.cuda.Stream()

		#group = dist.new_group(timeout=datetime.timedelta(seconds=5),)
		trainer = Trainer(world_size, rank,count, health_check_scheduler_thread, health_check_main_proc) 

		def custom_hook(args):
			# report the failure
			print("custom hook")
			#trainer = None 
			#gc.collect()

			trainer.train_continue = False
			trainer.release_all_lock()
			trainer.optimizer.train_continue = False
			print("1")
			trainer.optimizer.stop()	
			print("2")
#
			#print("!!!")
			#raise RuntimeError("An error")
			#RuntimeError("An error")
			#raise RuntimeError("other worker should be stopped")

			sys.exit("1")
			#exit()			
		groups = {}
		for i in range(world_size):
			for j in range(world_size):
				if(i != j):
					group = dist.new_group([i,j], backend='gloo')
					groups[f'{i}:{j}'] = group
		#run(comm_stream, group, world_size, rank)
		thread = threading.Thread(target=run, args=(health_check_main_proc, health_check_scheduler_thread, groups, world_size, rank, trainer))
		#thread.excepthook = custom_hook

		thread.daemon = True
		thread.start()	
		print("1")	
		#if(rank == 0):
		#	tensor_one = torch.ones(1)
		#	handle = dist.broadcast(tensor_one, group=group, src=rank, async_op=True)
		#	while not handle.is_completed() :
		#		print(f"broadcast src from {rank}")
		#		time.sleep(0.5)		
		#thread = threading.Thread(target=run2, args=(comm_stream, group, world_size, rank))
		#thread.start()					
		print("2")
		#time.sleep(1)
		trainer.benchmark_step()	
	except RuntimeError as error :
		print(f"RuntimeError {error}")
		#print(error)
		#with open('test.txt', encoding="utf-8") as f:
		#	f.write(error)
		#tensor_one = torch.ones(1)
		health_check_main_proc.acquire()
		#RuntimeError("An error")
		#raise RuntimeError("Source of RUntime Error")
		#handle = dist.broadcast(tensor_one, group=group, src=rank, async_op=True)
		#print("wait broadcast")
		#while not handle.is_completed() :
		#	print(f"broadcast src from {rank}")

		sys.exit(1)


			
			
	        #print(torch.cuda.mem_get_info(0)[0])	

	#for x in range(1):
	#	time = timeit.timeit(trainer.benchmark_step, number=5)
	#	img_sec = 32 * 10 / time
	#	print('Iter #%d: %.1f img/sec per'  % (x, img_sec))
	# 	img_secs.append(img_sec)
	

	
	#with open(f'memory_utilization.csv', 'w', newline='') as f:
	#	writer = csv.writer(f)
	#	for i in range(len(trainer.profiled_memory_utilization)):
	#		writer.writerow([trainer.profiled_memory_utilization[i]])


	#img_sec_mean = np.mean(img_secs)
	#img_sec_conf = 1.96 * np.std(img_secs)
	#print('Img/sec : %.1f +-%.1f' % ( img_sec_mean, img_sec_conf))
	#print('Total img/sec on %d (s): %.1f +-%.1f' % (world_size, world_size * img_sec_mean, world_size * img_sec_conf))
