
from custom_logger import customlogging
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

from tqdm import tqdm
import torch 
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity

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
from schedule_converter import make_schedule_from_json
from algo import schedule
from logger import write_trial
from memory_hook import register_memory_profiling_hooks
from dp_custom import DataParallel_Custom as DP
from ar_bucketer import ARBucketer 
from comm_mixin import CommMixin

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

class Trainer(CommMixin):
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

        self.batch_size = 32
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
        self._scheduled_comms = {}
        self._scheduled_comms["FWTOBW"] = {}
        self._scheduled_comms["FW"] = {}
        self._scheduled_comms["BW"] = {}
        self._scheduled_comms["BWTOFW"] = {}		
        self._schedule_comm_init = {}
        self._schedule_comm_init["FWTOBW"] = {}
        self._schedule_comm_init["FW"] = {}
        self._schedule_comm_init["BW"] = {}
        self._schedule_comm_init["BWTOFW"] = {}
        self._done_counts = {}

        self.model_parameter_names = {}
        self.synced_param_num_dict = {}

        self.datasets = []
        self.target = None
        self.data_index = 0
        self.profile_target_layer = []
        self.optim_dict = {}
        proc_list = list(range(world_size))
        self.group = dist.new_group(ranks=proc_list)
        max_param_num = get_param_num_by_buffer_size(self.world_size, self.bucket_size)
        self.bucketer = ARBucketer(max_param_num, self.world_size)

        print(f"before init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 

        self.train_dataset = datasets.CIFAR10(
            root='cifar10-data', train=True, download=False, transform=transforms.ToTensor())
        train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=False)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset , batch_size=self.batch_size, sampler=train_sampler, shuffle=False, num_workers=2)
        print(f"after init dataset  {torch.cuda.memory_allocated() / 1024 /1024}") 

        print(f"before init model  {torch.cuda.memory_allocated() / 1024 /1024}") 
        self.model = ResNet(Bottleneck,  [3, 4, 6, 3]) #it means "resnet50 model"
        self.model.cuda()
        print(f"after init model  {torch.cuda.memory_allocated() / 1024 /1024}") 


        #summary(self.model, ( 3, 32, 32))
        self.profiled_memory_utilization = []

        ngpus_per_node = torch.cuda.device_count()

        device_id = rank%ngpus_per_node		
        self.comm_stream = torch.cuda.Stream(device_id)
        self.post_ar_stream = torch.cuda.Stream(device_id)

        self.wrap_params = dict( mixed_precision=False, flatten_parameters=True, 

                                locks=self._locks,
                                health_check_main_proc=self.health_check_main_proc, 

                                conditions=self._conditions, 

                                profile_layer = self.profile_target_layer,

                                memory_record=self.profiled_memory_utilization,
                                
                                model_parameter_names=self.model_parameter_names,
                                comm_stream=self.comm_stream,
                                optim_dict=self.optim_dict,
                                init_comm_schedule=self._schedule_comm_init,
                                comm_schedule =self._scheduled_comms, 
                                param_name_dict=self.model_parameter_names,
                                bucketer=self.bucketer,
                                synced_param_num_dict=self.synced_param_num_dict
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
        with enable_wrap(**self.wrap_params):
            self.sharded_module = auto_wrap(adaptive_sdp, self.model)
            #self.sharded_module = DP(self.sharded_module)
            #self.sharded_module._lazy_init()
            
            self.adaptive_sdp_modules = {}
            self.adaptive_sdp_modules['FSDP'] = 0 
            self.adaptive_sdp_modules['SDP'] = 0
            self.adaptive_sdp_modules['DP'] = 0
##
            params_list = []
            params_name_list = []
            for n, p in self.sharded_module.named_parameters():
                print(n)
                if('_fsdp_wrapped_module' in n):
                    self.adaptive_sdp_modules['FSDP'] += 1
                elif('_sdp_wrapped_module' in n):
                    self.adaptive_sdp_modules['SDP'] += 1
                elif('_dp_wrapped_module' in n):
                    self.adaptive_sdp_modules['DP'] += 1
                for scheduled_comp in ["FW", "BW"]:
                    self._schedule_comm_init[scheduled_comp][n] = None
                    self._scheduled_comms[scheduled_comp][n] = None
                self.model_parameter_names[p] = n
                self.synced_param_num_dict[p] = 0	
                params_name_list.append(n)
                params_list.append(p)
                print(p)
            #for scheduled_comp in ["BWTOFW", "FWTOBW"]:
            #	self._schedule_comm_init[scheduled_comp]["None"] = []
            #	self._scheduled_comms[scheduled_comp]["None"] = [] 
            dist.barrier()
##
            
            #if(self.rank == 0):
            schedule(self.rank, self.world_size, self.adaptive_sdp_modules, max_param_num, \
                    layer_bench_file_name='profile_data/layer_bench.csv', net_bench_file_name='profile_data/net_bench.csv')
            dist.barrier()
#	##
            make_schedule_from_json(self.rank, params_list, params_name_list, self._schedule_comm_init, self._scheduled_comms, self._locks, self.adaptive_sdp_modules)
#	#
        dist.barrier()

        self.optimizer = torch.optim.Adam(self.sharded_module.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.optimizer = ShardScheduler(self.sharded_module, self.sharded_module.named_parameters(), self.world_size, self.rank, self.optimizer,
                                    self.partition_threshold, self._done_counts, self._partition_counts,
                                self.health_check_scheduler_thread,
                                self.health_check_thread_ready,
                                self.trial_info,
                                thread,
                                self._locks,
###
                                self._conditions,
###
                                self.profile_target_layer, 
                                self.bucket_size,
                                10**6, self.comm_stream, self._schedule_comm_init, self._scheduled_comms)
        self.bucketer.set_optimizer(self.optimizer)
        self.optim_dict["optimizer"] = self.optimizer
        
        self.criterion = nn.CrossEntropyLoss()

        self.scaler = GradScaler()
        print("end inittialization trainer")
        
        
    def set_comm_mixin(self):
        self.set_group(self.group)
        self.set_streams(self.comm_stream, self.post_ar_stream)
        self.set_rank(self.rank)
        self.set_bucketer(self.bucketer)
        self.set_param_name_dict(self.model_parameter_names)
        self.set_synced_param_num_dict(self.synced_param_num_dict)


    def communicate_nonoverlap(self, tag_name):
        task = self._scheduled_comms.get(tag_name, None)
        if task is not None:
            if bool(task):
                for comm in task.comms : 
                    self.do_communication(comm, tag_name=tag_name)	


    def benchmark_step(self):       
        count = 0
        start = time.time()
        self.set_comm_mixin()
        self.sharded_module.train()     
        for data, target in tqdm(self.train_loader):
            data = data.cuda()
            target = target.cuda()          
            self.communicate_nonoverlap("BWTOFW")
            output = self.sharded_module(data)
            self.communicate_nonoverlap("FWTOBW")           
            loss = output[0]
            loss = self.criterion(output, target)
            customlogging.debug(self.rank, loss)
            loss.backward()
            
            #print(f"after backward  {(torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024 /1024}") 
            count += 1
            #if(not self.train_continue or count ==5):
            if(not self.train_continue):
                break

        trial_info["time"] = time.time() - start
        write_trial(trial_info)			
        #os._exit(0)


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
    


def get_args_or_env(env_key_name, args_key_name, args):
    value = os.environ.get(env_key_name, None)
    if value is None :
        value = vars(args).get(args_key_name, None)
    return value 


def set_env(target_env_key_name, args, source_env_key_name=None, source_args_key_name=None):

    if os.environ.get(target_env_key_name, None) is None :
        if source_env_key_name is not None :
            env_val = os.environ.get(source_env_key_name, None)
            if env_val is not None:
                print(target_env_key_name)
                print(env_val)				
                os.environ[target_env_key_name] = env_val
        if source_args_key_name is not None:
            args_val = vars(args).get(source_args_key_name, None)
            if args_val is not None:
                print(target_env_key_name)
                print(args_val)
                os.environ[target_env_key_name] = args_val


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_memory', default=7.0, type=float)
    parser.add_argument('--sdp_ratio', default=0, type=float)
    parser.add_argument('--fsdp_ratio', default=0, type=float)
    parser.add_argument('--dp_ratio', default=0, type=float)
    parser.add_argument('--bucket_size', default=1, type=float)
    parser.add_argument('--exp_tag', type=str)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="210.107.197.218")
    parser.add_argument("--master_port", type=str, default="30002")
    parser.add_argument("--profile", type=str, default="false")
    args = parser.parse_args()

    world_size = int(get_args_or_env("WORLD_SIZE", "world_size", args))
    rank = int(get_args_or_env("SLURM_PROCID", "rank", args))

    set_env('MASTER_PORT', args, source_env_key_name='TRAINER_PORT', source_args_key_name='master_port')
    set_env("MASTER_ADDR", args, source_args_key_name='master_addr',)


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
        if args.profile == "true" :
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:	
            	with record_function("test_model"):
            		trainer.benchmark_step()
            if(rank == 0):		
            	prof.export_chrome_trace("trace.json")
        else:
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
