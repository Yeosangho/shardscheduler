
import torch
import torch.nn as nn
import functools

from custom_logger import customlogging
class CommMixin:
    def set_group(self, group):
        if self.bucketer.group is None:
            self.bucketer.group = group
    def set_rank(self, rank):
        self.rank = rank
        if self.bucketer.rank is None:
            self.bucketer.rank = rank 
    def set_streams(self, comm_stream, post_ar_stream):
        self.comm_stream = comm_stream
        
        if self.bucketer.comm_stream is None:
            self.bucketer.comm_stream = comm_stream
            self.bucketer.post_ar_stream = post_ar_stream
    def set_bucketer(self, ar_bucketer):
        self.bucketer = ar_bucketer 
    def set_param_name_dict(self, param_name_dict):
        self.param_name_dict = param_name_dict

    def set_synced_param_num_dict(self, synced_param_num_dict):
        self.synced_param_num_dict = synced_param_num_dict
        if self.bucketer.synced_param_num_dict is None:
            self.bucketer.set_synced_param_num_dict(synced_param_num_dict)
    def flush(self):
        self.bucketer.flush()

        

    def do_communication(self, comm, tag_name: str=None, comm_loc: str=None):
        if comm_loc == "forward":
            torch.cuda.current_stream().wait_stream(self.comm_stream)
        if(tag_name is not None):
            customlogging.debug(self.rank, f"communication is scheduled in {tag_name}")
        if comm.commType == "AG":
            None
        elif comm.commType == "AR":
            for idx, partiable_param in enumerate(comm.params): 
                self.do_allreduce_async(partiable_param)
            self.bucketer.flush()
        elif comm.commType == "RS":
            None 

    def do_allreduce_async(self, partiable_param):
        
        p = partiable_param.param
        #print("#####################")
        #print(p.shape)
        #print(partiable_param.start_ratio)
        #print(partiable_param.end_ratio)
        if p.grad is not None:
            grad = p.grad.data        
            param_name = self.param_name_dict[p]
            org_size = p._orig_size.numel()
            start_idx = partiable_param.start_idx
            end_idx = partiable_param.end_idx  
            customlogging.debug(self.rank, "do all reduce async")
            self.bucketer.allreduce_async(grad=grad,
                                                param_name=param_name,
                                                param=p,
                                                start_idx=start_idx,
                                                end_idx=end_idx,
                                                org_size=org_size, 
                                                shard_size=-1, 
                                                commType='AR')
