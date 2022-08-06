import csv
import functools
backward_ts = 0
comm_ts = 0

class CompOp:
    def __init__(self, name, idx, comptime, type):
        self.name = name
        self.idx = idx
        self.total_comptime = comptime
        self.overlappable_time = comptime
        self.type = type
        self.schedulable_comms = []
        self.scheduled_ag = []
        self.scheduled_ag_params = []
        self.scheduled_ar = []
        self.scheduled_ar_params = []
        self.scheduled_rs = []
        self.scheduled_rs_params = []

    def set_idx(self, idx):
        self.idx = idx
    def set_schedulable_comms(self, comms):
        self.schedulable_comms = comms
    def set_scheduled_comms(self, comm):
        if(comm.type == 'AG'):
            self.scheduled_ag.append(comm)
        elif(comm.type == 'RS'):
            self.scheduled_rs.append(comm)
        elif(comm.type == 'AR'):
            self.scheduled_ar.append(comm)
    def __str__(self):
        return f"{self.name}, {self.idx}, {self.type}, {self.overlappable_time} {len(self.scheduled_ag)} {len(self.scheduled_rs)} {len(self.scheduled_ar)} "
    def __repr__(self):
        return f"{self.name}, {self.idx}, {self.type}, {self.overlappable_time} {len(self.scheduled_ag)} {len(self.scheduled_rs)} {len(self.scheduled_ar)} "
class CommOp:
    def __init__(self, name, idx, param_num, type, time):
        self.name = name
        self.idx = idx
        self.type = type
        self.time = time
        self.residual_time = time 
        self.param_num = param_num
        self.overlappable_param_num = param_num
        self.schedulable_comps = []
        self.scheduled_comps = []
        self.scheduled_params = []

    def get_possible_schedulable_time(self):
        schedulable_time = 0 
        for comp in self.schedulable_comps:
            schedulable_time += comp.overlappable_time 
        return schedulable_time 


    def set_scheduled_comp(self, comp, param_num, time):
        self.scheduled_comps.append(comp)
        self.scheduled_params.append(param_num)
        self.overlappable_param_num -= param_num
        self.residual_time -= time
        

    def __str__(self):
        return f"{self.name}, {self.idx}, {self.type}, {self.get_possible_schedulable_time()} {self.param_num} {self.overlappable_param_num}"


    def __repr__(self):
        return f"{self.name}, {self.idx}, {self.type}, {self.get_possible_schedulable_time()} {self.param_num} {self.overlappable_param_num}"        

def get_layer_len(comp_ops):
    list_len = 0
    for comp_op in comp_ops:
        if(comp_op.type == 'forward'):
            list_len += 1
    return list_len

        

def schedule(adaptive_sdp):
    #schedule


    total_comp_times = 0
    total_backward_times = 0
    total_forward_times = 0
    comp_times = {}
    forward_times = {}
    backward_times = {}
    all_comp_times = {}
    param_nums = {}
    forward_ops = [] 
    backward_ops = [] 
    comp_ops = []
    f = open('layer_bench.csv','r')
    rdr = csv.reader(f)
    idx = 0
    for line in rdr:
        layer_name = line[0] 
        #if(btime < alpha)
        btime = float(line[1])
        ftime = float(line[2])

        total_comp_times += ftime
        total_forward_times += ftime
        total_comp_times += btime
        total_backward_times += btime


        param_nums[layer_name] = int(line[3])
        forward_op = CompOp(layer_name, idx, ftime, 'forward')
        forward_ops.append(forward_op)
        backward_op = CompOp(layer_name, idx, btime, 'backward')
        backward_ops.append(backward_op)

        comp_ops.append(forward_op)
        comp_ops.append(backward_op)

        idx += 1

    #layer_len = get_layer_len(comp_ops)
    #for comp_op in comp_ops:
    #    if(comp_op.type == 'backward'):
    #        comp_op.idx += layer_len


    f = open('net_bench.csv','r')
    rdr = csv.reader(f)
    alpha =  None
    beta = None
    for line in rdr:
        alpha = float(line[0])
        beta = float(line[1])


    #print(comp_times)
    #print(param_nums)
    #print(alpha)
    #print(beta)
    #1. adaptive shard on dp ---> sdp 
    # later layer has higher priority 
    #dp_start_idx = adaptive_sdp['sdp'] + adaptive_sdp['fsdp']
    #dp_end_idx = adaptive_sdp['sdp'] + adaptive_sdp['fsdp'] + adaptive_sdp['dp']
    ##2. adaptive shard on sdp ---> fsdp 
    ## earlier layer has higher priority 
    #fsdp_start_idx = 0
    #fsdp_end_idx = adaptive_sdp['fsdp']
    #sdp_start_idx =  adaptive_sdp['fsdp']
    #sdp_end_idx = adaptive_sdp['fsdp'] + adaptive_sdp['sdp'] 

    total_comm_times = 0
    comm_times_ag_rs = {}
    comm_ag_list = []
    comm_rs_list = []
    comm_ar_list = []
    comm_ag_fsdp_list = []
    comm_list = []
    #idx = 0
    for idx, key in enumerate(param_nums):
        time = alpha + beta * param_nums[key] *4 #32bit
        
        comm_ag = CommOp(key, idx, param_nums[key], 'ag', time)
        comm_ag_list.append(comm_ag)
        comm_rs = CommOp(key, idx, param_nums[key], 'rs', time)
        comm_rs_list.append(comm_rs)
        
        total_comm_times += time * 2
        comm_list.append(comm_ag)
        comm_list.append(comm_rs)

    #1. Initialize CompOps 
    for comp in comp_ops :
        overlappable_comms = [] 
        overlappable_comms.extend(comm_rs_list[comp.idx+1:])
        overlappable_comms.extend(comm_ag_list[comp.idx+1:])
        comp.set_schedulable_comms(overlappable_comms)
    #2. Find schedulable range of each comm op. 
    for comm in comm_list:
        overlappable_comps = []
        if(comm.type == 'ag'):
            overlappable_comps.extend(forward_ops[:comm.idx+1])
            overlappable_comps.extend(backward_ops[:comm.idx+1])
        elif(comm.type == 'rs'):
            overlappable_comps.extend(forward_ops[:comm.idx+1])
            overlappable_comps.extend(backward_ops[:comm.idx+1])

        comm.schedulable_comps = overlappable_comps

    #3. Find comp ops with maximum calc time 
    residual_total_param_num = 0
    residual_time = 0
    for comm in comm_list:
        residual_total_param_num += comm.overlappable_param_num 
        residual_time += comm.residual_time
    print(residual_time)
    print(residual_total_param_num)
    continue_schedule = True
    temp_comp_ops = []
    for comp in comp_ops:
        temp_comp_ops.append(comp)

    overlappable_times = 0  
    for comp in temp_comp_ops :
        overlappable_times += comp.overlappable_time
    print("comp time")
    print(overlappable_times)

    while continue_schedule :
        ordered_comp_ops = sorted(comp_ops, key=lambda x: x.overlappable_time, reverse=True)
        schedulable_comms = ordered_comp_ops[0].schedulable_comms
        #print(temp_comp_ops[3].overlappable_time)
        #print(ordered_comp_ops[0])
        #print(schedulable_comms)
        #4 Find comm ops with low schedulable range

        def compare(item1, item2):
            if ordered_comp_ops[0].type == 'forward' and item1.type != item2.type:
                if(item1.type == 'ag'):
                    return 1
                else:
                    return -1
            elif ordered_comp_ops[0].type == 'backward' and item1.type != item2.type:
                if(item1.type == 'rs'):
                    return 1
                else:
                    return -1
            else:
                return 0

        ordered_comm_ops = sorted(schedulable_comms, key=functools.cmp_to_key(compare), reverse=True) #schedulable_comms.sort( key=compare)
        ordered_comm_ops = sorted(ordered_comm_ops, key=lambda x:x.get_possible_schedulable_time())



        if(len(ordered_comm_ops) == 0):
            print("impossible to overlap!")
            comp_ops.remove(ordered_comp_ops[0])

        else:
            #for comm in ordered_comm_ops:
                #print(comm)
            target_comm = ordered_comm_ops[0]
            target_comp = ordered_comp_ops[0]
            #print(f"target_comm {target_comm}")
            #print(f"target_comp {target_comp}")
            #print(comm.type)
            if(target_comm.type == 'ag'):
                time = alpha + beta * target_comm.overlappable_param_num * 4
                param_num = target_comm.overlappable_param_num

                over_param_num = 0
                if(len(target_comp.scheduled_ag) > 0):
                    time = time - alpha
                    #print("fusion")
                #time = time - alpha
                if(time > target_comp.overlappable_time ):
                    if(len(target_comp.scheduled_ag) == 0):
                        if(target_comp.overlappable_time > alpha):
                            overlapped_param_num = (target_comp.overlappable_time - alpha) / (beta*4)
                            target_comp.overlappable_time = 0 
                            comp_ops.remove(target_comp)
                            target_comp.scheduled_ag.append(target_comm)
                            target_comp.scheduled_ag_params.append(overlapped_param_num)
                            target_comm.set_scheduled_comp(target_comp, overlapped_param_num, target_comp.overlappable_time)  
                        else:
                            #print("111")
                            target_comp.schedulable_comms.remove(target_comm)
                    else:
                        #print("111")
                        overlapped_param_num = (target_comp.overlappable_time ) / (beta*4)
                        #if(param_num - over_param_num > 0):
                        target_comp.overlappable_time = 0 
                        comp_ops.remove(target_comp)
                        target_comp.scheduled_ag.append(target_comm)
                        target_comp.scheduled_ag_params.append(overlapped_param_num)

                        target_comm.set_scheduled_comp(target_comp, overlapped_param_num, target_comp.overlappable_time)  
                        #print(target_comm.overlappable_param_num)

                elif(time <= target_comp.overlappable_time) :
                    target_comp.scheduled_ag.append(target_comm)
                    target_comp.scheduled_ag_params.append(param_num)

                    target_comm.set_scheduled_comp(target_comp, param_num, time)
                    target_comp.overlappable_time -= time 

            elif(target_comm.type == 'rs'):
                time = alpha + beta * target_comm.overlappable_param_num * 4
                param_num = target_comm.overlappable_param_num

                over_param_num = 0
                #if(len(target_comp.scheduled_rs) > 0):
                #    time = time - alpha
                #    print("fusion")
                if(len(target_comp.scheduled_rs) > 0):
                    time = time - alpha
                    #print("fusion")
                #time = time - alpha
                if(time > target_comp.overlappable_time ):
                    if(len(target_comp.scheduled_rs) == 0):

                        if(target_comp.overlappable_time > alpha):
                            overlapped_param_num = (target_comp.overlappable_time - alpha) / (beta*4)
                            target_comp.overlappable_time = 0 
                            comp_ops.remove(target_comp)
                            target_comp.scheduled_rs.append(target_comm)
                            target_comp.scheduled_rs_params.append(overlapped_param_num)

                            target_comm.set_scheduled_comp(target_comp, overlapped_param_num, target_comp.overlappable_time) 
                        else:
                            #print("111")
                            target_comp.schedulable_comms.remove(target_comm)
                    else:
                        overlapped_param_num = (target_comp.overlappable_time ) / (beta*4)
                        #if(param_num - over_param_num > 0):
                        target_comp.overlappable_time = 0 
                        comp_ops.remove(target_comp)
                        target_comp.scheduled_rs.append(target_comm)
                        target_comp.scheduled_rs_params.append(overlapped_param_num)
                        target_comm.set_scheduled_comp(target_comp, overlapped_param_num, target_comp.overlappable_time)  
                        #print(target_comm.overlappable_param_num)

                    #print("2")

                elif(time <= target_comp.overlappable_time) :
                    target_comp.scheduled_rs.append(target_comm)
                    target_comp.scheduled_rs_params.append(param_num)
                    target_comm.set_scheduled_comp(target_comp, param_num, time)
                    target_comp.overlappable_time -= time 
                    #print(target_comm.overlappable_param_num)


                #print(param_num)
            if(target_comm.overlappable_param_num == 0):
                for comp in comp_ops:
                    if(target_comm in comp.schedulable_comms):
                        #print("remove comms!")
                        comp.schedulable_comms.remove(target_comm)


        if(len(comp_ops) == 0):
            continue_schedule = False
            #print("comp_ops is zero!")
        residual_total_param_num = 0
        for comm in comm_list:
            residual_total_param_num += comm.overlappable_param_num 
        if(residual_total_param_num == 0):
            continue_schedule = False

    residual_total_param_num = 0
    residual_time = 0
    non_overlappable_comms = []
    for comm in comm_list:
        residual_total_param_num += comm.overlappable_param_num 
        residual_time += comm.residual_time
        if(comm.overlappable_param_num > 0):
            non_overlappable_comms.append(comm)
        #print(comm)
    #print(residual_time)
    #print(residual_total_param_num)
    overlappable_times = 0
    overlappable_params = 0  
    schedule_list = []

    comm_ops = ['ag', 'rs']
    for comp in temp_comp_ops :
        #print(comp)
        schedule_comp = {}
        schedule_comp['name'] = comp.name
        schedule_comp['idx']  = comp.idx
        schedule_comp['type'] = comp.type
        schedule_comp['total_comptime'] = comp.total_comptime
        schedule_comp['overlappable_time'] = comp.overlappable_time
        schedule_comp['scheduled_ag'] = []
        schedule_comp['scheduled_rs'] = []
        for comm_type in comm_ops : 
            comm_key = f'scheduled_{comm_type}'
            param_key = f'scheduled_{comm_type}_params'
            for comm, param in zip(getattr(comp, comm_key), getattr(comp, param_key)):
                comm_dict = {}
                comm_dict['name'] = comm.name
                comm_dict['idx'] = comm.idx
                comm_dict['type'] = comm.type 
                comm_dict['org_size'] = comm.param_num            
                comm_dict['param'] = param
                comm_dict['overlappable_param'] = comm.overlappable_param_num
                comm_dict['time'] = comm.time
                comm_dict['residual_time'] = comm.residual_time 
                schedule_comp[comm_key].append(comm_dict)
        schedule_list.append(schedule_comp)

    schedule_comp = {}
    schedule_comp['name'] = 'BWTOFW'
    schedule_comp['type'] = 'bwtofw'
    schedule_comp['scheduled_ag'] = []
    schedule_comp['scheduled_rs'] = []    

    for comm_type in comm_ops:
        comm_key = f'scheduled_{comm_type}'
        param_key = f'scheduled_{comm_type}_params' 
        for comm in non_overlappable_comms :
        
            comm_dict = {}
            comm_dict['name'] = comm.name
            comm_dict['idx'] = comm.idx
            comm_dict['type'] = comm.type 
            comm_dict['org_size'] = comm.param_num
            comm_dict['param'] = comm.overlappable_param_num
            comm_dict['time'] = comm.time
            comm_dict['residual_time'] = comm.residual_time
            if(comm.type == comm_type):
                schedule_comp[comm_key].append(comm_dict)

    schedule_list.append(schedule_comp)

    #print(schedule_list)
    import json 
    with open("schedule.json", "w") as json_file:
        json.dump(schedule_list, json_file)
    overlappable_times += comp.overlappable_time
