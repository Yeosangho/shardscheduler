import copy 
import math

class Task:
	def __init__(self,  comp,  compType, comms, idx=0):
		self.comp = comp
		self.idx=idx
		self.compType = compType
		self.comms = comms
	def __repr__(self):
		return f"{self.idx}, {self.compType}, {self.comms}"

	def __str__(self):
		return f"{self.idx}, {self.compType}, {self.comms}"	

class PartiableParam:
	def __init__(self, param, start_idx, end_idx, idx=0):
		self.param = param
		self.idx = idx
		self.start_idx = start_idx
		self.end_idx = end_idx
		self.shard_size = -1

	def __repr__(self):
		return f"[{self.idx}, {self.param._orig_size} {self.start_idx} {self.start_idx}]"	
	def __str__(self):
		return  f"[{self.idx}, {self.param._orig_size}  {self.end_idx} {self.end_idx}]"	


class Comm:
	def __init__(self, commType, params, fsdp=False):
		self.commType = commType
		self.params = params
		self.fsdp = fsdp

	def __repr__(self):
		return f"{self.commType}, {self.params}"	

	def __str__(self):
		return f"{self.commType}, {self.params}"	

def make_task(stage, task_dict, comm_ratio, comms, comm_ops, comp_types, params_list, params_name_list, comps_by_type):
	for comp_type in comp_types : 
		for comp in comps_by_type[comp_type]:
			if(stage == 'sdp' or stage == 'fsdp'):
				comp_param = params_list[int(comp['idx'])] if 'idx' in comp else None 
				comp_param_name = params_name_list[int(comp['idx'])] if 'idx' in comp else None
				comms = []  			
			for comm_op in comm_ops:
				target_comm_params = []
				for comm in comp['scheduled_comm'][comm_op]:
					param = params_list[int(comm['idx'])]
#
					start_idx = 0
					end_idx = 0
					if(comm['org_size'] == comm['param']):
						start_idx = 0
						end_idx = comm['param']
						print(f"end index when param is fully allocated to comm {end_idx}")
						target_comm_params.append(PartiableParam(param, start_idx, end_idx, idx=int(comm['idx'])))
					else:
						if(param in comm_ratio[comm_op]):
							start_idx = comm_ratio[comm_op][param]
						else:
							start_idx = 0
						end_idx = start_idx  + comm['param']

						if( start_idx < end_idx):
							target_comm_params.append(PartiableParam(param, start_idx, end_idx, comm['idx']))
							comm_ratio[comm_op][param] = end_idx
				if(len(target_comm_params) > 0):
					comm_merge = Comm(comm_op.upper(), target_comm_params)
					comms.append(comm_merge)
			
			if(len(comms) > 0 and stage == 'sdp'):
				idx = comp['idx'] if 'idx' in comp else None
				task = Task(comp_param, comp_type, comms, idx)
				if comp_param is not None :
					task_dict[comp_type][comp_param_name] = task
				else:
					task_dict[comp_type] = task
			if(len(comms) > 0 and stage == 'fsdp'):
				idx = comp['idx'] if 'idx' in comp else None
				
				#find comp is scheduled in previous steps
				exist_task = None			
				for task in task_dict[comp_type] :
					
					if(task.idx == idx ):
						exist_task = task
						print(f"!!!!!!! {task.idx} {idx}")
				if(exist_task != None):
					exist_task.comms.extend(comms)
				else:
					task = Task(comp_param, comp_type, comms, idx)
					if comp_param is not None :
						task_dict[comp_type][comp_param_name] = task
					else:
						task_dict[comp_type] = task
	if stage == "init":
		task = Task(None, 'BWTOFW', comms, None)	
		task_dict['INIT']= task

def make_schedule_from_json(rank, params_list, params_name_list, scheduled_comms_init , scheduled_comms, locks, adaptive_sdp_modules, json_path='schedule.json'):
	fsdp_num = adaptive_sdp_modules['FSDP']
	dp_num = adaptive_sdp_modules['DP']
	sdp_num = adaptive_sdp_modules['SDP']	
	import json
	with open(f"schedule_{rank}.json", "r") as schedule_json:
		schedule_json = json.load(schedule_json)
	
	layer_dp_list = schedule_json['dp_type']
	schedule_list = schedule_json['schedule']

	target_comm_params = []
	
	#AG scheduled_comms_init 
	ag_init_params = []
	for dp_type, params in  zip(layer_dp_list, params_list):
		if(dp_type == 'fsdp' or dp_type == 'sdp'):
			ag_init_params.append(params)
#
	#target_comm_params = get_patial_param_list(ag_init_params)
	#comm = Comm('AG', target_comm_params)
	#task = Task(None, 'BWTOFW', [comm])
	#scheduled_comms_init.append(task)	

	comps_by_type = {}
	comps_by_type['FW'] = []
	comps_by_type['BW'] = []
	comps_by_type["BWTOFW"] = []
	comps_by_type["FWTOBW"] = []


	for comp in schedule_list:
		print(comp)
		print(comp['type'])
		if(comp['type'] == 'forward'):
			comps_by_type['FW'].append(comp)
		elif(comp['type'] == 'backward'):
			comps_by_type['BW'].append(comp)
		elif(comp['type'] == 'bwtofw'):
			comps_by_type["BWTOFW"].append(comp)
		elif(comp['type'] == 'fwtobw'):
			comps_by_type["FWTOBW"].append(comp)

	for comp in comps_by_type['FW']:
		print(comp)

	#현재는 하나의 레이어에서 이루어지는 동일한 유형의 통신은 모두 병합하도록 정의되어있음 -> 이 부분은 버퍼사이즈에 따라 분할되도록  수정 필요 .
	task_dict = {}
	task_dict['INIT'] = {}
	task_dict['BWTOFW'] = {}
	task_dict['FW'] = {}
	task_dict['FWTOBW'] = {}
	task_dict['BW'] = {}
	comps_by_type['BW'] = list(reversed(comps_by_type['BW']))


	comm_ratio = {}
	comm_ratio['ag'] = {}

	comm_ops = ['ag']
	comp_types = ['BW','BWTOFW']
	
	comms = []
	make_task("init",  task_dict, comm_ratio, comms, comm_ops, comp_types, params_list, params_name_list, comps_by_type)



	comm_ratio = {}
	comm_ratio['ag'] = {}
	comm_ratio['rs'] = {}
	comm_ratio['ar'] = {}

	comm_ops = ['rs', 'ag', 'ar']
	comp_types = ['BW','BWTOFW', 'FW',]
	
	make_task("sdp", task_dict, comm_ratio, comms, comm_ops, comp_types, params_list, params_name_list, comps_by_type)

	comm_ratio = {}
	comm_ratio['ag_fsdp'] = {}
	comm_param_num = {}
	comm_param_num['ag_fsdp'] = {}
	comm_ops = ['ag_fsdp']
	comp_types = ['FW','FWTOBW', 'BW',]

	#target_comm_params = get_patial_param_list([params_list[-1]])
	#comm = Comm('AG', target_comm_params)
	#task_fwtobw = Task(None, 'FWTOBW', [comm])	
	#for idx, param in enumerate(params_list[1:]):
	#	target_comm_params = get_patial_param_list([params_list[idx]])
	#	comm = Comm('AG', target_comm_params)	
	#	exist_task = None	
	#	for task in task_dict['BW'] :
	#		
	#		if(task.idx == idx+1 ):
	#			exist_task = task
	#			print(f"!!!!!!! {task.idx} {idx}")
	#	if(exist_task != None):
	#		exist_task.comms.append(comm)
	#	else:
	#		task = Task(param, 'BW', [comm], idx+1)	
	#		task_dict['BW'].append(task)					

	make_task("fsdp",  task_dict, comm_ratio, comms, comm_ops, comp_types, params_list, params_name_list, comps_by_type)


	#Sorting scheudle BWTOFW -> FW -> FWTOBW -> BW
	#find BWTOFW

	fw_ops = task_dict['FW']
	bw_ops = task_dict['BW']
	#fw_ops = sorted(task_dict['FW'], key=lambda x: x.idx)

	#bw_ops = sorted(task_dict['BW'], key=lambda x: x.idx, reverse=True)

	#scheduled_comms.append(task_fwtobw)

	scheduled_comms['FWTOBW'] = task_dict['FWTOBW']
	scheduled_comms['BW'] = bw_ops

	scheduled_comms['BWTOFW'] = task_dict['BWTOFW']
	scheduled_comms['FW'] = fw_ops


	if(sdp_num + fsdp_num > 0):
		scheduled_comms_init['FWTOBW'] = task_dict['INIT']
		scheduled_comms_init['FW'] = fw_ops
	else:
		scheduled_comms_init['FWTOBW'] = scheduled_comms['FWTOBW']
		scheduled_comms_init['BW'] = scheduled_comms['BW']
		scheduled_comms_init['FW'] = scheduled_comms['FW']
		scheduled_comms_init['BWTOFW'] = scheduled_comms['BWTOFW']
				

	#os._exit()


	#print(task_init_list)
	#print(task_dict['BWTOFW'])
	#import os
	#os._exit(0)
	#for task in scheduled_comms :
	#	for comm in task.comms :
	#		for param_wrap in comm.params : 
	#			param = param_wrap.param				
	#			if(comm.commType == 'AG'):
	#				if(locks[comm.commType][param].locked() == False):
	#					locks[comm.commType][param].acquire()			
#
	#locks['FWTOBW'].acquire()
