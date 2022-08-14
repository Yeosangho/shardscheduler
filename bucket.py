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

class Bucket:
    def __init__(self, size, world_size):

        self.world_size = world_size
        self.parameter_num = size * 1024 * 1024 / 4
        self.offset = 0



        self.params  = ParamList()
        self.input = torch.zeros((int(self.parameter_num))).cuda()
        self.output = torch.zeros((int(self.parameter_num))).cuda()
        self.ar_buffer = torch.zeros((int(self.parameter_num))).cuda()

    def has_enough_space(self, data):
        None
    
    def push(self, param=None, grad=None, params=None,  start_idx=None, end_idx=None, org_size=None, shard_size=None, commType=None):

        if(commType == 'AG'):
            param_num = end_idx - start_idx 

            self.input[self.offset : self.offset + param_num ].copy_(param[start_idx : end_idx ])
            self.offset += param_num
            self.params.add(param, start_idx, end_idx, org_size, shard_size, self.offset)

        elif(commType == 'RS'):
            param_num = end_idx - start_idx 
            self.input = self.input.view(self.world_size, -1)
            stacked_input = torch.stack(params).view(self.world_size, -1)
            self.input[:, self.offset : self.offset + param_num].copy_(stacked_input[:,start_idx : end_idx])
            self.offset += param_num
            self.params.add(param, start_idx, end_idx, org_size, shard_size, self.offset, grad=grad)

        elif(commType == 'AR'):
            param_num = end_idx - start_idx
            #print("###################################")
            #print(param_num)
            #print(grad.shape)
            #print(end_idx)
            #print(start_idx)
            self.ar_buffer[self.offset : self.offset + param_num].copy_(grad[start_idx : end_idx]) 
            self.offset += param_num
            self.params.add(param, start_idx, end_idx, org_size, shard_size, self.offset, grad=grad)

    def flush(self):
        self.offset = 0
        self.params.flush()
        self.input = self.input.view(-1)
        self.output = self.output.view(-1)

