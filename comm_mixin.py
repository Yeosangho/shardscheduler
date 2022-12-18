
import torch.nn as nn

class CommMixin:
    def set_bucketer(self, ar_bucketer):
        self.bucketer = ar_bucketer 
    def set_param_name_dict(self, param_name_dict):
        self.param_name_dict = param_name_dict
    def set_synced_param_num_dict(self, synced_param_dict):
        self.synced_param_dict = synced_param_dict
    def do_communication(self, comm, tag_name: str=None):
        if(tag_name is not None):
            print(f"communication is scheduled in {tag_name}")
        if comm.commType == "AG":
            None
        elif comm.commType == "AR":
            for idx, partiable_param in enumerate(comm.params): 
                self.do_allreduce_async(partiable_param)
            self.bucketer.flush()

        elif comm.commType == "RS":
            None 

    def do_allreduce_async(self, partiable_param):
        print("do all reduce async")
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
            self.synced_param_num_dict[p] += end_idx - start_idx                         
            print(f"scheduled communitcation param {param_name}, start_idx {start_idx}, end_idx {end_idx}, org_size {org_size} current communicated num {self.synced_param_num_dict[p] }")
            if(org_size == end_idx):
                print(f"scheduled params is fully communicated  param {param_name}, start_idx {start_idx}, end_idx {end_idx}, org_size {org_size}")
            #self.bucketer.allreduce_async(grad=grad,
            #                                    param_name=param_name,
            #                                    param=p,
            #                                    start_idx=start_idx,
            #                                    end_idx=end_idx,
            #                                    org_size=org_size, 
            #                                    shard_size=-1, 
            #                                    commType='AR')
