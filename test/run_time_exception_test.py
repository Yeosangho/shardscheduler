import torch
import os, time 
import datetime
import argparse
import torch.distributed as dist
from pynvml import *
import torch.cuda as cutorch

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '210.107.197.218'
    os.environ['MASTER_PORT'] = '30000'
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', dest='rank', default=0, type=int) 
    args = parser.parse_args()  
    world_size = 2
    rank = args.rank    
    #world_size = int(os.environ["WORLD_SIZE"])
    #rank = int(os.environ['SLURM_PROCID'])	

    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = 7 * 1024 * 1024 * 1024
    fraction = target_memory  /  total_memory
    #print(fraction)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)    	
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)



    #t = torch.cuda.get_device_properties(0).total_memory
    #r = torch.cuda.memory_reserved(0)
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print(t)
    #print(r)
    #print(a)
    #print(f)
    #print(torch.cuda.max_memory_allocated(0))
    ##size = 8000
    ##parameter_num = size * 1024 * 1024 / 4
#
    ##one = torch.ones((1)).cuda()
    ##handle = dist.all_reduce(one, async_op=True)
    ##time_slept = 0
    ##while not handle.is_completed() :
    ##    time.sleep(0.001)
    ##    time_slept += 0.001
    ##    print(time_slept)
    ##    if(time_slept > 1):
    ##        raise  Exception('!!!')
    #nvmlInit()
    #h = nvmlDeviceGetHandleByIndex(0)
    #info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total}')
    #print(f'free     : {info.free}')
    #print(f'used     : {info.used}')
    #torch.cuda.empty_cache()
#
    print(torch.cuda.mem_get_info(0)[0])
    #free_mem = torch.cuda.mem_get_info(0)[0]
    #free_mem_tensor = torch.tensor(free_mem).cuda()
    #dist.all_reduce(free_mem_tensor,op=dist.ReduceOp.MIN )
    #free_mem = torch.cuda.mem_get_info(0)[0]
    #free_mem_tensor = torch.tensor(free_mem).cuda()
    #dist.all_reduce(free_mem_tensor,op=dist.ReduceOp.MIN )    
    #print(free_mem_tensor)
    #print(free_mem)
    ##torch.cuda.empty_cache()
#
    #diff_free_mem = int((free_mem -free_mem_tensor.item()))
    #print(diff_free_mem)
    #parameter_num = int(diff_free_mem / 4 )
    #if(diff_free_mem != 0 ):
    #    diff_mem_tensor = torch.zeros(int(parameter_num)).cuda()
#
    #free_mem = torch.cuda.mem_get_info(0)[0]
    #free_mem_tensor = torch.tensor(free_mem).cuda()
    #dist.all_reduce(free_mem_tensor,op=dist.ReduceOp.MIN )    
    #print(free_mem_tensor)
    #print(free_mem)
    ##torch.cuda.empty_cache()
#
    #diff_free_mem = int((free_mem -free_mem_tensor.item()))
    #print(diff_free_mem)
    #parameter_num = int(diff_free_mem / 4 )
    #if(diff_free_mem != 0 ):
    #    print("assign tensor")
    #    diff_mem_tensor = torch.zeros(int(parameter_num)).cuda()
#
    #print(torch.cuda.mem_get_info(0)[0])
    #print(torch.cuda.mem_get_info(0)[0])

    times = 1.00001
    while True :
        try :
            dist.barrier()
            parameter_num = int((target_memory*times) / 4)
            ar_buffer = torch.zeros((int(parameter_num))).cuda()
#
            print("?????")
            break
        except RuntimeError :
            #print(torch.cuda.mem_get_info(0)[0])
            print("!!!!!")
            times -= 0.000002

