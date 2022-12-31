import math
import time
from custom_logger import customlogging
import torch
import torch.distributed as dist



class Param:
    def __init__(self, param, start_idx, end_idx, org_size, shard_size, grad=None, param_name=None):
        self.param = param
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.org_size = org_size
        self.shard_size = shard_size
        self.grad=grad
        self.param_name=param_name

class ParamList:
    def __init__(self):
        self.params = []
        self.offsets = []
        
    def add(self, param, start_idx, end_idx, org_size, shard_size, offset, grad=None, param_name=None):
        p = Param(param, start_idx, end_idx, org_size, shard_size, grad=grad,  param_name=param_name)

        self.params.append(p)
        param_num = end_idx - start_idx + 1

        self.offsets.append(offset)
    def flush(self):
        self.params = []
        self.offsets = [] 

class ARBucketer:
    '''
    This buckerter is refactoring version of bucket.py .
    ARBucketer add feature about "collective communincation" and "check tensor size to decide whether push this tensor into bucket or not."
    '''
    def __init__(self, parameter_num, world_size):
        self.world_size = world_size
        self.parameter_num = parameter_num
        self.shard_size = math.ceil(self.parameter_num / self.world_size )

        self.offset = 0


        self.comm_param_size_dict = {}

        self.params  = ParamList()
        self.rank = None
        self.synced_param_num_dict = None
        self.group = None
        self.fusion_buffer = torch.zeros((int(self.shard_size * (world_size+1)))).cuda()
    def set_synced_param_num_dict(self, synced_param_num_dict):
        self.synced_param_num_dict = synced_param_num_dict
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer 
    def allreduce_async(self,param=None, grad=None, param_name=None, start_idx=None, end_idx=None, org_size=None, shard_size=None, commType=None):
        param_num = end_idx - start_idx
        param_size = self.comm_param_size_dict.get(param_name, None)
        if param_size is None:
            customlogging.debug(self.rank, "add param_num to self.comm_param_size_dict")
            self.comm_param_size_dict[param_name] = param_num
        else:
            self.comm_param_size_dict[param_name] += param_num
            customlogging.debug(self.rank, "same parameter is in the dictionary")

        #customlogging.debug(self.rank, f"before allreduce {param_name} :: {torch.sum(grad.data)}")
        if(param_num > self.parameter_num):
            #dist.all_reduce(grad, async_op=False)
            self.direct_comm(param, grad, param_name, start_idx, end_idx, org_size, shard_size)
            #optimizer call when all parameters in layer is communicated. currently, I remove condition statement for testing. 
#
        else:
            self.iterative_push(param, grad, param_name, start_idx, end_idx, org_size, shard_size, commType)
   

        if(org_size == param_size):
            customlogging.debug(self.rank, "check shard size is equal to param_size")
    def direct_comm(self, param, grad, param_name, start_idx, end_idx, org_size, shard_size):
        customlogging.debug(self.rank, f"before allreduce direct comm :: {torch.sum(grad[start_idx:end_idx])}")
        dist.all_reduce(grad[start_idx:end_idx], group=self.group)
        #handle = dist.all_reduce(grad[start_idx:end_idx], async_op=True)
        #handle.wait()
        #while not handle.is_completed():
        #    pass
        customlogging.debug(self.rank, f"after allreduce direct comm :: {torch.sum(grad[start_idx:end_idx])}")

        self.update_param(param, param_name, start_idx, end_idx, org_size)

    def update_param(self, param, param_name, start_idx, end_idx, org_size):
        self.synced_param_num_dict[param] += end_idx - start_idx                         
        customlogging.debug(self.rank, f"scheduled communitcation param {param_name}, start_idx {start_idx}, end_idx {end_idx}, org_size {org_size} current communicated num {self.synced_param_num_dict[param] }")
        if(org_size == end_idx):
            customlogging.debug(self.rank, f"after allreduce {param_name} :: {torch.sum(param.grad.data)}")
            customlogging.debug(self.rank, f"scheduled params is fully communicated  param {param_name}, start_idx {start_idx}, end_idx {end_idx}, org_size {org_size}")
            self.optimize_param(param)

    def optimize_param(self, param):
        self.optimizer._adam(param)
        self.optimizer._zero_one_grad(param)

    def iterative_push(self, param, grad, param_name, start_idx, end_idx, org_size, shard_size, commType):
        remains, start_idx = self.push( param, grad, param_name, start_idx, end_idx, org_size, shard_size, commType)
        while remains > 0:
            self.flush()
            remains, start_idx = self.push( param, grad, param_name, start_idx, end_idx, org_size, shard_size, commType)

    def push(self, param=None, grad=None,  param_name=None, start_idx=None, end_idx=None, org_size=None, shard_size=None, commType=None):
        param_num = end_idx - start_idx
        #print("###################################")
        #print(param_num)
        #print(grad.shape)
        #print(end_idx)
        #print(start_idx)
        #remains = 0 
        #if(self.fusion_buffer[self.offset : self.offset + param_num ].size() != param[start_idx : end_idx ].size()[0]):
        remains = param_num - self.fusion_buffer[self.offset : self.offset + param_num ].size()[0]            
        #print(f"{self.rank} remains : {remains}")
        #print(f"{self.rank} buffer size : {self.fusion_buffer[self.offset : self.offset + param_num ].size()}")
        #print(f"{self.rank} param_num : {param_num}  {end_idx} {start_idx}")
        #print(f"{self.rank} grad size : {grad.size()} {grad[start_idx : end_idx-remains].size()}")

        self.fusion_buffer[self.offset : self.offset + param_num-remains].copy_(grad[start_idx : end_idx-remains]) 
        self.offset += param_num - remains
        self.params.add(param, start_idx, end_idx-remains, org_size, shard_size, self.offset, grad=grad, param_name=param_name)
        start_idx = end_idx - remains
        return remains, start_idx

    def bucket_comm(self):
        customlogging.debug(self.rank, f"before allreduce fusion buffer :: {torch.sum(self.fusion_buffer[:self.offset])}")
        dist.all_reduce(self.fusion_buffer[:self.offset], group=self.group)
        #handle = dist.all_reduce(self.fusion_buffer[:self.offset], async_op=True)
        #handle.wait()
        #while not handle.is_completed():
        #    pass
        customlogging.debug(self.rank, f"after allreduce fusion buffer :: {torch.sum(self.fusion_buffer[:self.offset])}")
        self.fusion_buffer[:self.offset]=  self.fusion_buffer[:self.offset] / self.world_size
        pre_offset = 0
        for param_wrap, offset in zip(self.params.params, self.params.offsets):   

            param = param_wrap.param  
            param_name = param_wrap.param_name
            start_idx = param_wrap.start_idx
            end_idx = param_wrap.end_idx
            org_size = param_wrap.org_size
            param.grad.data[start_idx:end_idx].copy_(self.fusion_buffer[pre_offset:offset])
            pre_offset = offset

            self.update_param(param, param_name, start_idx, end_idx, org_size)

    def flush(self):
        self.bucket_comm()
        self.offset = 0
        self.params.flush()
        self.fusion_buffer = self.fusion_buffer.view(-1)

