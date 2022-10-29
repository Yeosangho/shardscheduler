from __future__ import absolute_import
import os
import threading
import logging
import time
import traceback
try:
    import queue
except ImportError:
    import Queue as queue
import torch
from torch.nn.parameter import Parameter

import time
import math
import torch.distributed as dist
import csv
from bucket import Bucket
from fairscale.utils.parallel import (
    chunk_and_pad,
    enable_pytorch_sync_bn,
    get_process_group_cached,
    validate_process_group,
)

def get_param_num_by_buffer_size(world_size, buffer_size):
    return (world_size/(world_size+1)) * buffer_size * 1024 * 1024 / 4         
#logging.basicConfig(level=logging.DEBUG)
class ShardScheduler(torch.optim.Optimizer):
    """An optimizer that wraps a hvd._DistributedOptimizer, intercepting allreduce operations and wrap as tasks."""
    def __init__(self, model, named_parameters, size, rank, opt, 
                 partition_threshold, done_counts, partition_counts, 
                 health_check_lock,
                 health_check_thread_ready,
                 locks,
                 conditions, 
                 profile_layer,
                 bucket_size=1,
                 num_steps=10**6, comm_stream=None, init_schedules=None, schedules=None):
        """Construct a new ScheduledOptimizer, which uses horovod optimizer under the hood for averaging gradients
         across all the Horovod ranks.

        Args:
            model: The training model. ByteScheduler uses the model object to register hooks.
            hvd_opt: Optimizer to use for averaging gradients and applying updates.
            num_steps: The maximum number of training steps. ByteScheduler needs to know when to stop cross-iteration
            scheduling.

        Usage example:
        ```
        import bytescheduler.pytorch.horovod as bsc
        bsc.init()
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters, compression)
        optimizer = bsc.ScheduledOptimizer(model, optimizer, num_steps)
        ```
        """
        print("!!!!!!!!!!!!!!!!!!!!")
        #handle = BYTESCHEDULER_LIB.bytescheduler_create_event(0)
        #super(self.__class__, self).__init__(model.parameters())
        self.health_check_lock = health_check_lock
        self.health_check_thread_ready = health_check_thread_ready
        self._model = model
        self._size= size
        self._rank = rank
        self.bucket_size = bucket_size
        self._opt = opt
        self._logger = logging.getLogger("ByteScheduler")
        self._logger.debug("hvd size {}, rank {}".format(size, rank))
        self._desc = "rank {}".format(rank)
        self._grad_accs = []
        self._requires_update = set()
        self._handles = {}
        #self._handlequeue = queue.Queue()
        self._handlequeue = []
        # Track training steps
        self._step = 0
        self._final_step = num_steps
        self._stop_event = threading.Event()

        
        self.partition_threshold = partition_threshold
        self.done_counts = done_counts
        self.partition_counts = partition_counts

        self._locks = locks
        self._conditions = conditions 

        self.comm_stream = comm_stream
        
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [(f'allreduce.noname.{i}.{j}', v)
                                for i, param_group in enumerate(self.param_groups)
                                for j, v in enumerate(param_group['params'])]
        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = ShardScheduler.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids = {id(v)
                         for param_group in self.param_groups
                         for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))
        backward_passes_per_step=1
        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}


        # Use lock to block the forward propagation of each parameter.
        self.init_schedules = init_schedules
        self.schedules = schedules
        self.is_first_itr = True

        # The closer to input layer, the higher the priority is.
        self._priority_indexes = {}
        priority = 0
        for p in model.parameters():
            self._priority_indexes[p] = priority
            priority += 1


        
        self.profile_layer = profile_layer
        # Poll whether the tensor is ready for allreduce or whether the allreduce is finished.
        self.event_queue = queue.Queue()
        self.all_reduce_stream = torch.cuda.Stream()

        self.scheduler_ready = threading.Lock()

        self._poller = threading.Thread(target=self._poll_FSDP, args=( ))
        #self._poller.excepthook = self.custom_hook

        self._poller.daemon = True
        self._poller.start()   


        # Let rank 0 decide the communication order.
        self._immediate = False
        self.train_continue = True
        #if self._rank != 0:
        #    self._immediate = True
        #core.start(self._parameter_names, rank=self._rank, arch="allreduce")

    #def start_scheduler(self):

    def stop(self):
        #exit()
        print("1.1")
        self.release_all_lock()
        print("1.2")

        self.train_continue = False
        self._stop_event.set()
        print("1.3")
        #self._poller.join()  
        print("1.4")

    def release_all_lock(self):
        for n, p in self._model.named_parameters():
            self._release_lock(self._locks['FW'][p], self._conditions['FW'][p])      
            self._release_lock(self._locks['BW'][p], self._conditions['BW'][p])  
        if self._locks['BWTOFW'].locked():   
            self._release_lock(self._locks['BWTOFW'], self._conditions['BWTOFW'])  
        if self._locks['FWTOBW'].locked():   
            self._release_lock(self._locks['FWTOBW'], self._conditions['FWTOBW'])                  
    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def __getattr__(self, item):
        return getattr(self._opt, item)

    def __del__(self):
        """Clean up"""
        self.event_queue.put((None, None, None, None, None))
        #self._poller.join()
        #core.shutdown(wait_for_all=False)

    def step(self, closure=None):
        """Override the default step function."""
        self._logger.debug("{} calls step() {}".format(self._desc, self._step))
        #for i in range(self._handlequeue.qsize()) :
    	#    handle = self._handlequeue.get()
    	#    handle.wait()
        #for i in self._handlequeue :
    	#    handle = self._handlequeue.pop(0)
    	#    handle.wait()        
        # Step 0 is called for parameter initialization after parameter broadcast
        if self._size > 1 and self._step > 0:
            # if it is the final training step, wait for the completion of all tensors
            if self._step == self._final_step:
                self._logger.debug("final step {}, waiting for allreduce completion.".format(self._final_step))
                while not self.event_queue.empty():
                    time.sleep(0.001)
            loss = None
            if closure is not None:
                loss = closure()
            self._step += 1
            return loss
        else:
            # SGD.step() will be triggered when user calls hvd.broadcast_optimizer_sate()
            #super(self._opt.__class__, self._opt).step()
            self._opt.step()
            self._step += 1

        #for i in self._handlequeue :
    	#    handle = self._handlequeue.pop(0)
    	#    #p.data = p_cpu.data.cuda()
    	#    handle.wait()
        #self._opt.step()
        #self._step += 1

    def zero_grad(self):
        """Override the default zero_grad function

        Clears the gradients of all optimized :class:`torch.Tensor` s.
        """
        self._logger.debug("{} calls zero_grad() of step {}".format(self._desc, self._step))
        if self._size > 1 and self._step > 0:
            return
        else:
            self._opt.zero_grad()

    def allreduce_grad_async(self, tensor, name):
        """Call horovod API to allreduce gradient asynchronously

        Arguments:
            tensor: The tensor to be allreduced.
            name: The name of the tensor.

        Returns:
            an allreduce handle and context
        """
        #ctx = tensor.type()
        ctx = name
        #print(f'{self._desc} before allreduce {name}  :  {torch.sum(tensor)}')
        #with open(f'before_{self._desc}.csv', 'a', newline='') as f:
        #    writer = csv.writer(f)
        #    writer.writerow([name, torch.sum(tensor).item()])
        handle = dist.all_reduce(tensor,async_op=True)
        self._handlequeue.put(handle)
        return handle, ctx

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

    def _poll_FSDP(self, ):
        print(self.comm_stream)


        try :
            #waiting until health check thread is ready
            while self.health_check_thread_ready.locked():
                time.sleep(0.5)

            #bucket size to parameter_num
            
            param_num =  get_param_num_by_buffer_size(self._size, self.bucket_size)  
            self.bucket = Bucket(param_num, self._size) #parameter_num          
            with torch.cuda.stream(self.comm_stream):
                #self.scheduler_ready.acquire()
                self.run_schedule(self.init_schedules , init=True)
                #self.scheduler_ready.acquire()
                #exit()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                while not self._stop_event.is_set():
                    self.run_schedule(self.schedules, init=False)
        except RuntimeError as error :
            print("Runtime error in scheduler")
            print(traceback.format_exc())
            #dist.destroy_process_group()
            self.health_check_lock.acquire()

    def run_schedule(self, schedule, init=False):
        for task in schedule:   
            #print(f"before {task.compType}")
     
            if(self._stop_event.is_set()):
                break           
            if(task.compType == 'FW' or task.compType == 'BW'):
                self._wait_unlock(self._locks[task.compType][task.comp], self._conditions[task.compType][task.comp])  
            else:
                self._wait_unlock(self._locks[task.compType], self._conditions[task.compType])  
            #print(f"after {task.compType}")

            for comm in task.comms : 
                #print(f"{comm} {len(comm.params)} ")
                #if(len(comm.params)> 43):
                #    print(comm.params[42])
                #print(comm.commType)
                if(comm.commType == "AG"): #AG recover original parameter 
                    remains = 0
                    stopped_idx = 0
                    comm_continue = True
                    while comm_continue : 
                        is_break = False

                        for idx, partiable_param in enumerate(comm.params[stopped_idx:], start=stopped_idx): 
                            
                            p = partiable_param.param
                            p_data = p.data.to(p._full_param_padded.device)
                            p_size = p._full_param_padded.size()
                            p_data.new_zeros(p_size)

                            org_size = p._orig_size
                            shard_size = p_data.numel()
                            start_idx = int(shard_size * partiable_param.start_ratio)
                            #print(partiable_param.end_ratio)
                            end_idx = int(shard_size * partiable_param.end_ratio)    

                            if(p.data_ptr() == self.profile_layer[0].data_ptr()):
                                print('before ag')
                                print(p.shape)
                                print(p.sum())
                            if(remains == 0):
                                remains = self.bucket.push(param=p,
                                            start_idx=start_idx,
                                            end_idx=end_idx,
                                            org_size=org_size, 
                                            shard_size=shard_size, 
                                            commType='AG')                            
                            else:
                                remains = self.bucket.push(param=p,
                                            start_idx=end_idx - remains,
                                            end_idx=end_idx,
                                            org_size=org_size, 
                                            shard_size=shard_size, 
                                            commType='AG') 
                            #print("###############################")
                            #print(remains) 
                            #print(len(comm.params)) 
                            #print(idx)
                            #print(p.shape)
                            #print(stopped_idx)
                            #print("############################")
                            if(remains>0):
                                stopped_idx = idx
                                is_break = True
                                break
                        if(idx == len(comm.params) -1 and not is_break):
                            comm_continue = False 
                    #print(f"ag p.shape {p.shape}" ) 
                    #print(f"end-start {end_idx-start_idx}")
                        #print("############################")
                        ##output_tensor_list = list(bucket.output.view(world_size, -1)[:self.bucket.offset].unbind(0))
                        handle = dist._all_gather_base(self.bucket.org_buffer[:self.bucket.offset*2], self.bucket.shard_buffer[:self.bucket.offset], async_op=True)
                        while not handle.is_completed() :
                            time.sleep(0.001)
                        output_tensor = self.bucket.org_buffer[:self.bucket.offset*2].view(2, -1)
                        pre_offset = 0
                        #!!!!!")
                        for param_wrap, offset in zip(self.bucket.params.params, self.bucket.params.offsets):   
                            #p._full_param_padded.storage().resize_(0)
                            param = param_wrap.param
                            p_size = param._full_param_padded.size()
                            if(param_wrap.start_idx == 0):
                                param._full_param_padded.storage().resize_(p_size.numel())                      
                            #print(param._full_param_padded.shape)
                            #print(param_wrap.shard_size)
                            listed_full_param = param._full_param_padded.view(2,param_wrap.shard_size)
                            #print(listed_full_param.shape)
                            #print(param_wrap.start_idx)
                            #print(param_wrap.end_idx)
                            #print(pre_offset)
                            #print(offset)                        
                            listed_full_param[:,param_wrap.start_idx : param_wrap.end_idx].copy_(output_tensor[:,pre_offset : offset])


                            pre_offset = offset
                            if(param_wrap.end_idx == param_wrap.shard_size):
                            #    print("!!!!!!!!!!!!!!!!!!!!!!!!")

                                #param.data =  param._full_param_padded
                                param.data = listed_full_param.view(-1)
                                param.data = param.data[: param_wrap.org_size.numel()].view(param_wrap.org_size)
                                if(param.data_ptr() == self.profile_layer[0].data_ptr()):
                                    print('after ag')
                                    print(param.shape)
                                    print(param.sum())
                                #torch.cuda.empty_cache() 
                                self._release_lock(self._locks['AG'][param], self._conditions['AG'][param])

                                #torch.cuda.synchronize()

                        self.bucket.flush()

                elif(comm.commType== "RS" and init == False): #after backward
                    
                    remains = 0
                    stopped_idx = 0
                    comm_continue = True
                    while comm_continue : 
                        is_break = False
                        for idx, partiable_param in enumerate(comm.params[stopped_idx:], start=stopped_idx): 
                            p = partiable_param.param

                            grad = p.grad.data
                            grad_chunks = chunk_and_pad(grad, 2)
                            #p.grad.data = torch.zeros_like(grad_chunks[0]).type(p.grad.dtype).to(p.device)

                            org_size = p._orig_size
                            shard_size = grad_chunks[0].numel()
                            start_idx = int(shard_size * partiable_param.start_ratio)
                            end_idx = int(shard_size * partiable_param.end_ratio)
                            #print("????")      
                            #print(start_idx)
                            #print(end_idx)                                               
                            #input_flattened = torch.cat(grad_chunks)
                            if(p.data_ptr() == self.profile_layer[0].data_ptr()):
                                print('before rs')
                            
                                print(p.grad.shape)
                                print(p.grad.sum())
                            #print(shard_size)
                            #print("##############before push")
                            #print(p.shape)
                            #print(partiable_param.start_ratio)
                            #print(partiable_param.end_ratio)
                            #print(remains)
                            #print(shard_size)
                            if(remains == 0):
                                remains = self.bucket.push(params=grad_chunks,
                                                grad=grad,
                                                param = p,
                                                start_idx= start_idx,
                                                end_idx=end_idx,
                                                org_size=org_size, 
                                                shard_size=shard_size, 
                                                commType='RS')                                  
                            else :
                                remains = self.bucket.push(params=grad_chunks,
                                                grad=grad,
                                                param = p,
                                                start_idx= end_idx - remains,
                                                end_idx=end_idx,
                                                org_size=org_size, 
                                                shard_size=shard_size, 
                                                commType='RS')  
                            #print("###############################")
                            #print(remains) 
                            #print(len(comm.params)) 
                            #print(idx)
                            #print(p.shape)
                            #print(stopped_idx)
                            #print("############################")


                            if(remains>0):
                                stopped_idx = idx
                                is_break = True
                                break                                            
                            #grad_chunks=None
                        if(idx == len(comm.params) -1 and not is_break ):
                            comm_continue = False                         
                        #print(self.bucket.offset)
                        handle = dist._reduce_scatter_base(self.bucket.shard_buffer[:self.bucket.offset], self.bucket.org_buffer[:, :self.bucket.offset].contiguous(), async_op=True)  
                        while not handle.is_completed():
                            time.sleep(0.001)
                        self.bucket.shard_buffer[:self.bucket.offset]= self.bucket.shard_buffer[:self.bucket.offset] / 2    
                        pre_offset = 0
                        count = 0
                        for param_wrap, offset in zip(self.bucket.params.params, self.bucket.params.offsets):   
                        
                            #p.grad = None  
    
                            param = param_wrap.param  
                            if(param_wrap.start_idx == 0):
                                #print( param.grad.data[:param_wrap.shard_size].size())
                                param._full_param_padded.data.storage().resize_( param.grad.data[:param_wrap.shard_size].size()[0])
                            param._full_param_padded.data[param_wrap.start_idx:param_wrap.end_idx].copy_(self.bucket.shard_buffer[pre_offset:offset])
                            pre_offset = offset
                            count += 1
                            if(param_wrap.end_idx == param_wrap.shard_size):
                                param.grad.data =  torch.zeros_like( param.grad.data[:param_wrap.shard_size]).type(param.grad.dtype).to(param.device)  
                                param.grad.data.copy_(param._full_param_padded.data[:param_wrap.shard_size])
                                param._full_param_padded.data.storage().resize_( 0)   
                                #print(f"output p.grad[0] {param.grad.shape} {torch.sum(param.grad)}")

                                #param.grad.data = param.grad.data 
                                if(param.data_ptr() == self.profile_layer[0].data_ptr()):
                                    print('after rs')
                                    print(param.shape)
                                    print("###################")
                                    print(param.grad.sum())
                                
                                #print(count)
                                #print(param_wrap.shard_size)
                                #print(param_wrap.org_size)   
                                self._post_reduction_hook(param, param.grad.data)
                                self._finalize_parameters(param)
                                self._adam(param)
                                self._zero_one_grad(param)
                                #param.grad.data = None
                                grad = None
                                param.grad = None
                                #torch.cuda.empty_cache() 
                                #param.grad = None
                                #param.sum()    
    
                        self.bucket.flush()
                elif(comm.commType== "AR" and init==False):
                    remains = 0
                    stopped_idx = 0
                    comm_continue = True
                    while comm_continue : 
                        is_break = False
                        for idx, partiable_param in enumerate(comm.params[stopped_idx:], start=stopped_idx): 
                            p = partiable_param.param
                            #print("#####################")
                            #print(p.shape)
                            #print(partiable_param.start_ratio)
                            #print(partiable_param.end_ratio)
                            grad = p.grad.data        

                            org_size = p._orig_size.numel()
                            start_idx = int(org_size * partiable_param.start_ratio)
                            end_idx = int(org_size * partiable_param.end_ratio)                            
    
                            
                            if(remains == 0):
                                remains = self.bucket.push(grad=grad,
                                                param = p,
                                                start_idx=start_idx,
                                                end_idx=end_idx,
                                                org_size=org_size, 
                                                shard_size=-1, 
                                                commType='AR')  
                            else:
                                remains = self.bucket.push(grad=grad,
                                                param = p,
                                                start_idx=end_idx - remains,
                                                end_idx=end_idx,
                                                org_size=org_size, 
                                                shard_size=-1, 
                                                commType='AR')  
                            if(remains>0):
                                stopped_idx = idx
                                is_break = True
                                break                                            
                            grad_chunks=None
                        if(idx == len(comm.params) -1 and not is_break ):
                            comm_continue = False  

                        handle = dist.all_reduce(self.bucket.fusion_buffer[:self.bucket.offset], async_op=True)  
                        while not handle.is_completed():
                            time.sleep(0.001)     
                        self.bucket.fusion_buffer[:self.bucket.offset]=  self.bucket.fusion_buffer[:self.bucket.offset] / 2
                        pre_offset = 0
                        for param_wrap, offset in zip(self.bucket.params.params, self.bucket.params.offsets):   

                            param = param_wrap.param  

                            param.grad.data[param_wrap.start_idx:param_wrap.end_idx].copy_(self.bucket.fusion_buffer[pre_offset:offset])
                            pre_offset = offset

                            if(param_wrap.end_idx == param_wrap.org_size):
                                #if(param.data_ptr() == self.profile_layer[0].data_ptr()):
                                        #print('after ar')
                                        #print(param.shape)
                                        #print(param.grad.sum())  
                                self._adam(param)
                                self._zero_one_grad(param)
                                #torch.cuda.empty_cache() 
                                self._release_lock(self._locks['AR'][param], self._conditions['AR'][param])

                                grad = None
                                #p.grad = None                          
                        self.bucket.flush()

            if(task.compType == 'FW' or task.compType == 'BW'):
                #if(not self.health_check_lock.locked()):
                self._acquire_lock(self._locks[task.compType][task.comp])
            else:
                #print(task.compType)
                #if(not self.health_check_lock.locked()):
                self._acquire_lock(self._locks[task.compType])   


    def _post_reduction_hook(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        """Hook to call on each param after the reduce-scatter."""

        if param._is_sharded:
            # Accumulate into the gradient shard.
            if getattr(param, "_saved_grad_shard", None) is None:
                param._saved_grad_shard = reduced_grad.data
            else:
                assert (
                    param._saved_grad_shard.shape == reduced_grad.shape
                ), f"{param._saved_grad_shard.shape} vs {reduced_grad.shape}"
                param._saved_grad_shard.data += reduced_grad.data
            reduced_grad = param._saved_grad_shard.data



    def _finalize_parameters(self, p):
        if not p.requires_grad:
            return
        if hasattr(p, "_shard_bwd_hook"):
            assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
            p._shard_bwd_hook[1].remove()
            delattr(p, "_shard_bwd_hook")

        # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
        # remains the unsharded gradient accumulated from prior no-sync passes, and p._saved_grad_shard
        # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
        # sync passes, if desired.
        #if not self._require_backward_grad_sync:
        #    return

        # Parameter and gradient devices must match.
        if hasattr(p, "_cpu_grad"):
            assert p.device == torch.device("cpu")
            p.grad = p._cpu_grad
        elif hasattr(p, "_saved_grad_shard"):
            assert p.device == p._saved_grad_shard.device
            #f"finalize parameter p.grad.shape {p.grad.shape}")
            #print(f"finalize parameter p.grad.shpae {p._saved_grad_shard.shape}")
            p.grad = p._saved_grad_shard
    
        if hasattr(p, "_saved_grad_shard"):
            delattr(p, "_saved_grad_shard")

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as PyTorch accumulates gradients by default.

        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            # Not sure whether to do detach_ or not
            p.grad.detach_()
            p.grad.zero_()

    """Below are the implementations of optimizers, e.g., SGD, Adam."""

    def _sgd(self, p):
        """Performs a single optimization step using SGD optimizer on a parameter.

        Arguments:
            p: The parameter to be updated.
        """
        # TODO: support other optimizers later, or figure out a walk around way
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                d_p = p.grad.data
                #print(self._parameter_names[p])
                #print(p.data.shape)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                #d_p = None
                #print(f"p.shape & data {p.shape} {p.data[0]}")
                break

    def _adam(self, p):
        """Performs a single optimization step using Adam optimizer on a parameter.

        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            for gp in group['params']:
                if self._parameter_names[p] != self._parameter_names[gp] or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._parameter_names[p]))
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(-step_size, exp_avg, denom)
                #print(p.data.sum())
                break


def init():
    """Replace _register_hook() function in hvd._DistributedOptimizer with empty function."""

    def hijack(obj, func_name):
        orig_func = getattr(obj, func_name)
        print("hijack function {}".format(orig_func))

        def wrapped_func(*args, **kwargs):
            print("function {} is hijacked to do nothing.".format(orig_func))
            return
        setattr(obj, func_name, wrapped_func)

    #hijack(hvd._DistributedOptimizer, '_register_hooks')
