import argparse
import os
import threading
import torch
import time
import torch.distributed as dist

world_size = 8
def is_any_completed(handles):

    for handle in handles:
        if handle.is_completed():
            return True
    return False 

#multiple process health check logic test 
def run(health_check_main_proc, groups, world_size, rank):
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
    handles = []
    for i in range(world_size):
        if(i!=rank):
            handle = dist.broadcast(tensor_one, group=groups[f"{i}"], src=i, async_op=True)
            handles.append(handle)

    #handle.wait()
    while not is_any_completed(handles) :
    
        print(f"wait error status from other process")
        if health_check_main_proc.locked():
            handle_me = dist.broadcast(tensor_one, group=groups[f"{rank}"], src=rank, async_op=True)
            while not handle_me.is_completed() :
                print(f"broadcast {rank}")
                time.sleep(0.5)
            break
        time.sleep(0.5) 
    print("!!!!!!!!!!! run with exception") 
    os._exit(1)

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '210.107.197.219'
    os.environ['MASTER_PORT'] = '30005'
    os.environ["NCCL_SOCKET_IFNAME"]="eno1,eth0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', dest='rank', default=0, type=int)
    args = parser.parse_args()
    rank = args.rank

    health_check_main_proc = threading.Lock()
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank) 

    groups = {}
    for i in range(world_size):
        for j in range(world_size):
            if(i != j):
                proc_list = list(range(world_size))
                group = dist.new_group(proc_list, backend='gloo')
                groups[f'{i}'] = group

    thread = threading.Thread(target=run, args=(health_check_main_proc, groups, world_size, rank, ))
    thread.daemon = True
    thread.start()
    target_ranks = [1,2]
    if(rank not in target_ranks):	
        time.sleep(10)
    elif(rank in target_ranks):
        time.sleep(5)
        health_check_main_proc.acquire()



    