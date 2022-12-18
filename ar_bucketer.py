import math
import torch



class Param:
    def __init__(self, param, start_idx, end_idx, org_size, shard_size, grad=None):
        self.param = param
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.org_size = org_size
        self.shard_size = shard_size
        self.grad=grad


class ParamList:
    def __init__(self):
        self.params = []
        self.offsets = []
        
    def add(self, param, start_idx, end_idx, org_size, shard_size, offset, grad=None):
        p = Param(param, start_idx, end_idx, org_size, shard_size, grad=grad)

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
        self.fusion_buffer = torch.zeros((int(self.shard_size * (world_size+1)))).cuda()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer 
    def allreduce_async(self,param=None, grad=None, param_name=None, params=None,  start_idx=None, end_idx=None, org_size=None, shard_size=None, commType=None):
        param_num = end_idx - start_idx
        param_size = self.comm_param_size_dict.get(param_name, None)
        if param_size is None:
            print("add param_num to self.comm_param_size_dict")
            self.comm_param_size_dict[param_name] = param_num
        else:
            self.comm_param_size_dict[param_name] += param_num
            print("same parameter is in the dictionary")

        if(param_num > self.parameter_num):
            dist.all_reduce(grad, async_op=False)

            #optimizer call when all parameters in layer is communicated. currently, I remove condition statement for testing. 
            self.optimizer._adam(param)
            self.optimizer._zero_one_grad(param)
        else:
            self.iterative_push(param, grad, params, start_idx, end_idx, org_size, shard_size, commType)
   

        if(org_size == param_size):
            print("check shard size is equal to param_size")
    def iterative_push(self, param, grad, params, start_idx, end_idx, org_size, shard_size, commType)
        remains = self.push( param, grad, params, start_idx, end_idx, org_size, shard_size, commType)
        while remains > 0:
            self.flush()
            remains = self.push( param, grad, params, start_idx, end_idx, org_size, shard_size, commType)

    def push(self, param=None, grad=None, params=None,  start_idx=None, end_idx=None, org_size=None, shard_size=None, commType=None):
        param_num = end_idx - start_idx
        #print("###################################")
        #print(param_num)
        #print(grad.shape)
        #print(end_idx)
        #print(start_idx)
        remains = 0 
        if(self.fusion_buffer[self.offset : self.offset + param_num ].size() != param[start_idx : end_idx ].size()[0]):
            remains = param_num - self.fusion_buffer[self.offset : self.offset + param_num ].size()[0]            

        self.fusion_buffer[self.offset : self.offset + param_num].copy_(grad[start_idx : end_idx-remains]) 
        self.offset += param_num
        self.params.add(param, start_idx, end_idx-remains, org_size, shard_size, self.offset, grad=grad)
        return remains

    def flush(self):
        self.offset = 0
        self.params.flush()
        self.org_buffer = self.org_buffer.view(-1)

