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
	def __init__(self, param, start_ratio=0.0, end_ratio=1.0, idx=0):
		self.param = param
		self.idx = idx
		self.start_ratio = start_ratio
		self.end_ratio = end_ratio
		self.shard_size = -1

	def __repr__(self):
		return f"[{self.idx}, {self.param._orig_size} {self.start_ratio} {self.end_ratio}]"	
	def __str__(self):
		return  f"[{self.idx}, {self.param._orig_size}  {self.start_ratio} {self.end_ratio}]"	


class Comm:
	def __init__(self, commType, params):
		self.commType = commType
		self.params = params
	def __repr__(self):
		return f"{self.commType}, {self.params}"	

	def __str__(self):
		return f"{self.commType}, {self.params}"	

def get_patial_param_list(param_list, partial_param_list=None, start_ratio=0.0, end_ratio=1.0):
	if(partial_param_list is None):
		partial_params = []
	else:
		partial_params = partial_param_list 
	for param in param_list:
		partial_param = PartiableParam(param, start_ratio, end_ratio)
		partial_params.append(partial_param)
	return partial_params

def make_schedule_wfbp_fsdp(params_list, scheduled_comms_init , scheduled_comms, locks):
        target_comm_params = get_patial_param_list([params_list[0]])
        comm = Comm('AG', target_comm_params)
        task = Task(None, 'BWTOFW', [comm])
        scheduled_comms_init.append(task)       
        idx = 1
        for param in params_list[:-1] :
        	comp_param = param
        	target_comm_params = get_patial_param_list([params_list[idx]])
        	comm = Comm('AG', target_comm_params )
        	task = Task(comp_param, 'FW', [comm])
        	scheduled_comms_init.append(task)       
        	idx += 1        
        target_comm_params = get_patial_param_list([params_list[-1]])        
        len_target_comm_params = len(target_comm_params)
        comm = Comm('AG', target_comm_params)
        task  = Task(None, 'FWTOBW', [comm])
        scheduled_comms_init.append(task)			        
        idx = 0
        reversed_params = list(reversed(params_list))
        len_params = len(reversed_params)
        for param in reversed_params:
            comms = []
            comp_param = param
            if(idx < len_params-1):
                target_comm_params =get_patial_param_list([reversed_params[idx+1]])  
                comm = Comm('AG', target_comm_params )
                comms.append(comm)
            if(idx > 0):
                target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
                comm = Comm('RS', target_comm_params )
                comms.append(comm)
            task = Task(comp_param, 'BW', comms)
            scheduled_comms_init.append(task)     
            idx += 1   

        comms = []
        target_comm_params = get_patial_param_list([reversed_params[-1]])       
        len_target_comm_params = len(target_comm_params)
        comm = Comm('RS', target_comm_params)
        comms.append(comm)
        target_comm_params = get_patial_param_list([params_list[0]])
        comm = Comm('AG', target_comm_params)
        comms.append(comm)

        task = Task(None, 'BWTOFW', comms)
        scheduled_comms.append(task)    

        idx = 1
        for param in params_list[:-1] :
        	comp_param = param
        	target_comm_params = get_patial_param_list([params_list[idx]])
        	comm = Comm('AG', target_comm_params )
        	task = Task(comp_param, 'FW', [comm])
        	scheduled_comms.append(task)       
        	idx += 1        
        target_comm_params = get_patial_param_list([params_list[-1]])        
        len_target_comm_params = len(target_comm_params)
        comm = Comm('AG', target_comm_params)
        task  = Task(None, 'FWTOBW', [comm])
        scheduled_comms.append(task)			        

        idx = 0
        reversed_params = list(reversed(params_list))
        len_params = len(reversed_params)
        for param in reversed_params:
            comms = []
            comp_param = param
            if(idx < len_params-1):
                target_comm_params =get_patial_param_list([reversed_params[idx+1]])  
                comm = Comm('AG', target_comm_params )
                comms.append(comm)
            if(idx > 0):
                target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
                comm = Comm('RS', target_comm_params )
                comms.append(comm)
            task = Task(comp_param, 'BW', comms)
            scheduled_comms.append(task)     
            idx += 1   


        for task in scheduled_comms :
        	for comm in task.comms :
        		for param_wrap in comm.params : 
        			param = param_wrap.param
        			if(comm.commType == 'AG'):
        				if(locks[comm.commType][param].locked() == False):
        					locks[comm.commType][param].acquire()
        #self._locks['BWTOFW'].release()
        locks['FWTOBW'].acquire()

def make_schedule_wfbp_sdp(params_list, scheduled_comms_init , scheduled_comms, locks):
        target_comm_params = get_patial_param_list([params_list[0]])
        comm = Comm('AG', target_comm_params)
        task = Task(None, 'BWTOFW', [comm])
        scheduled_comms_init.append(task)       
        idx = 1
        for param in params_list[:-1] :
        	comp_param = param
        	target_comm_params = get_patial_param_list([params_list[idx]])
        	comm = Comm('AG', target_comm_params )
        	task = Task(comp_param, 'FW', [comm])
        	scheduled_comms_init.append(task)       
        	idx += 1        

        comms = []		
        task  = Task(None, 'FWTOBW',comms)
        scheduled_comms_init.append(task)

        idx = 0
        reversed_params = list(reversed(params_list))
        len_params = len(reversed_params)
        for param in reversed_params:
            comms = []
            comp_param = param
            if(idx > 0):
                target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
                comm = Comm('RS', target_comm_params )
                comms.append(comm)
            task = Task(comp_param, 'BW', comms)
            scheduled_comms_init.append(task)     
            idx += 1   

        comms = []
        target_comm_params = get_patial_param_list([reversed_params[-1]])       
        len_target_comm_params = len(target_comm_params)
        comm = Comm('RS', target_comm_params)
        comms.append(comm)
        target_comm_params = get_patial_param_list([params_list[0]])
        comm = Comm('AG', target_comm_params)
        comms.append(comm)

        task = Task(None, 'BWTOFW', comms)
        scheduled_comms.append(task)    

        idx = 1
        for param in params_list[:-1] :
        	comp_param = param
        	target_comm_params = get_patial_param_list([params_list[idx]])
        	comm = Comm('AG', target_comm_params )
        	task = Task(comp_param, 'FW', [comm])
        	scheduled_comms.append(task)       
        	idx += 1        
		        

        comms = []
        task  = Task(None, 'FWTOBW',comms)
        scheduled_comms.append(task)
		
        idx = 0
        reversed_params = list(reversed(params_list))
        len_params = len(reversed_params)
        for param in reversed_params:
            comms = []
            comp_param = param
            if(idx > 0):
                target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
                comm = Comm('RS', target_comm_params )
                comms.append(comm)
            task = Task(comp_param, 'BW', comms)
            scheduled_comms.append(task)     
            idx += 1   

        for task in scheduled_comms :
        	for comm in task.comms :
        		for param_wrap in comm.params : 
        			param = param_wrap.param
        			if(comm.commType == 'AG'):
        				if(locks[comm.commType][param].locked() == False):
        					locks[comm.commType][param].acquire()
        #self._locks['BWTOFW'].release()
        locks['FWTOBW'].acquire()


def make_schedule_wfbp_dp(params_list, scheduled_comms_init , scheduled_comms, locks):
        comms = []
        task  = Task(None, 'FWTOBW',comms)
        scheduled_comms_init.append(task)
        idx = 0
        reversed_params = list(reversed(params_list))
        len_params = len(reversed_params)
        for param in reversed_params:
            comms = []
            comp_param = param
            if(idx > 0):
                target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
                comm = Comm('AR', target_comm_params )
                comms.append(comm)
            task = Task(comp_param, 'BW', comms)
            scheduled_comms_init.append(task)     
            idx += 1   

        comms = []
        target_comm_params = get_patial_param_list([reversed_params[-1]])       
        len_target_comm_params = len(target_comm_params)
        comm = Comm('AR', target_comm_params)
        comms.append(comm)


        task = Task(None, 'BWTOFW', comms)
        scheduled_comms.append(task)    


        idx = 0
        reversed_params = list(reversed(params_list))
        len_params = len(reversed_params)
        for param in reversed_params:
            comms = []
            comp_param = param
            if(idx > 0):
                target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
                comm = Comm('AR', target_comm_params )
                comms.append(comm)
            task = Task(comp_param, 'BW', comms)
            scheduled_comms.append(task)     
            idx += 1   

        comms = []
        task  = Task(None, 'FWTOBW',comms)
        scheduled_comms.append(task)

        for task in scheduled_comms :
        	for comm in task.comms :
        		for param_wrap in comm.params : 
        			param = param_wrap.param
        			#if(comm.commType == 'AR'):
        			#	if(locks[comm.commType][param].locked() == False):
        			#		locks[comm.commType][param].acquire()
        locks['BWTOFW'].acquire()
        locks['FWTOBW'].acquire()

def make_schedule_fsdp(params_list, scheduled_comms_init , scheduled_comms, locks):
		target_comm_params = get_patial_param_list([params_list[0]])
		comm = Comm('AG', target_comm_params)
		task = Task(None, 'BWTOFW', [comm])
		scheduled_comms_init.append(task)

		idx = 1
		for param in params_list[:-1] :
			comp_param = param
			target_comm_params = get_patial_param_list([params_list[idx]])
			comm = Comm('AG', target_comm_params )
			task = Task(comp_param, 'FW', [comm])
			scheduled_comms_init.append(task)

			idx += 1

		target_comm_params = get_patial_param_list(params_list)

		len_target_comm_params = len(target_comm_params)
		comm = Comm('AG', target_comm_params)
		task  = Task(None, 'FWTOBW', [comm])
		scheduled_comms_init.append(task)			


		target_comp = params_list[29]
		target_comms = get_patial_param_list(params_list[30:40])
		comm = Comm('RS', target_comms)
		task  = Task(target_comp, 'BW', [comm])
		scheduled_comms_init.append(task)


		comms = []
		target_comm_params = get_patial_param_list(params_list[:30])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[40:])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[:50])
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.0, end_ratio=0.41)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		#target_comm_params = self.get_patial_param_list(params_list[:70])
		#comm = Comm('AG', target_comm_params)
		#comms.append(comm)

		task = Task(None, 'BWTOFW', comms)
		scheduled_comms_init.append(task)


		####################################################################
		#make test scheduling(backward)
		#10번 foward 연산을 11,12 params AG와 overlap
		#10번 forward 연산을 7,8,9 AG FSDP와 overlap

		target_comp = params_list[29]
		comms = []
		#target_comm_params = self.get_patial_param_list(params_list[70:])
		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.41, end_ratio=1.0)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[1:28])
		comm = Comm('AG', target_comm_params)
		comms.append(comm)


		task  = Task(target_comp, 'FW', comms)


		scheduled_comms.append(task)


		#FWTOBW
		#여기에는 나머지 AGFSDP 통신 정의
		comms = []
		target_comm_params = get_patial_param_list(params_list[:1])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[28:])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		task  = Task(None, 'FWTOBW',comms)
		scheduled_comms.append(task)
		#make test scheduling(backward)
		#backward_params =  reversed(list(self.sharded_module.named_parameters()))

		#5번 param backward 연산을 11,12 backward params의 RS 와 overlap
		target_comp = params_list[29]
		#target_comms = self.get_patial_param_list(params_list[30:40])
		target_comm_params = get_patial_param_list(params_list[30:40], start_ratio=0.0, end_ratio=0.75)
		comm = Comm('RS', target_comm_params)
		task  = Task(target_comp, 'BW', [comm])
		scheduled_comms.append(task)


		#BWTOFW
		#여기에는 나머지 AG와 RS 통신 정의
		comms = []
		target_comm_params = get_patial_param_list(params_list[:30])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[30:40], start_ratio=0.75, end_ratio=1.0)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[40:])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		#target_comm_params = self.get_patial_param_list(params_list[:70])
		target_comm_params = get_patial_param_list(params_list[:50])
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.0, end_ratio=0.41)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		#target_comm_params = params_list[40:]
		#comm = Comm('AG', target_comm_params)
		#comms.append(comm)

		task = Task(None, 'BWTOFW', comms)
		scheduled_comms.append(task)


		for task in scheduled_comms :
			for comm in task.comms :
				for param_wrap in comm.params : 
					param = param_wrap.param
					if(comm.commType == 'AG'):
						if(locks[comm.commType][param].locked() == False):
							locks[comm.commType][param].acquire()
		#self._locks['BWTOFW'].release()
		locks['FWTOBW'].acquire()

def make_schedule_dp(params_list, scheduled_comms_init , scheduled_comms, locks):
		comms = []
		target_comm_params = get_patial_param_list(params_list[:50])
		comm = Comm('AR', target_comm_params)
		comms.append(comm)
		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.0, end_ratio=0.41)
		comm = Comm('AR', target_comm_params)
		comms.append(comm)



		task = Task(None, 'BWTOFW', comms)
		scheduled_comms_init.append(task)


		####################################################################
		#make test scheduling(backward)
		#10번 foward 연산을 11,12 params AG와 overlap
		#10번 forward 연산을 7,8,9 AG FSDP와 overlap

		target_comp = params_list[29]
		comms = []
		#target_comm_params = self.get_patial_param_list(params_list[70:])
		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.41, end_ratio=1.0)
		comm = Comm('AR', target_comm_params)
		comms.append(comm)
		task  = Task(target_comp, 'FW', comms)


		scheduled_comms.append(task)

		task = Task(None, 'FWTOBW', [])
		scheduled_comms.append(task)



		#make test scheduling(backward)
		#backward_params =  reversed(list(self.sharded_module.named_parameters()))

		#5번 param backward 연산을 11,12 backward params의 RS 와 overlap
		target_comp = params_list[29]
		comms = []
		#target_comms = self.get_patial_param_list(params_list[30:40])
		target_comm_params = get_patial_param_list(params_list[30:50])
		comm = Comm('AR', target_comm_params)
		comms.append(comm)

		task  = Task(target_comp, 'BW', comms)

		scheduled_comms.append(task)


		#BWTOFW
		#여기에는 나머지 AG와 RS 통신 정의
		comms = []
		target_comm_params = get_patial_param_list(params_list[:30])
		comm = Comm('AR', target_comm_params)
		comms.append(comm)

		#target_comm_params = self.get_patial_param_list(params_list[:70])
		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.0, end_ratio=0.41)
		comm = Comm('AR', target_comm_params)
		comms.append(comm)


		task = Task(None, 'BWTOFW', comms)
		scheduled_comms.append(task)


		for task in scheduled_comms :
			for comm in task.comms :
				for param_wrap in comm.params : 
					param = param_wrap.param
					#if(comm.commType == 'AR'):
					#	if(self._locks[comm.commType][param].locked() == False):
					#		self._locks[comm.commType][param].acquire()
		locks['BWTOFW'].acquire()	
		locks['FWTOBW'].acquire()


def make_schedule_sdp(params_list, scheduled_comms_init , scheduled_comms, locks):
		target_comm_params = get_patial_param_list([params_list[0]])
		comm = Comm('AG', target_comm_params)
		task = Task(None, 'BWTOFW', [comm])
		scheduled_comms_init.append(task)

		idx = 1
		for param in params_list[:-1] :
			comp_param = param
			target_comm_params = get_patial_param_list([params_list[idx]])
			comm = Comm('AG', target_comm_params )
			task = Task(comp_param, 'FW', [comm])
			scheduled_comms_init.append(task)

			idx += 1


		target_comp = params_list[29]
		target_comms = get_patial_param_list(params_list[30:40])
		comm = Comm('RS', target_comms)
		task  = Task(target_comp, 'BW', [comm])
		scheduled_comms_init.append(task)


		comms = []
		target_comm_params = get_patial_param_list(params_list[:30])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[40:])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[:50])
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.0, end_ratio=0.41)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		#target_comm_params = self.get_patial_param_list(params_list[:70])
		#comm = Comm('AG', target_comm_params)
		#comms.append(comm)

		task = Task(None, 'BWTOFW', comms)
		scheduled_comms_init.append(task)


		####################################################################
		#make test scheduling(backward)
		#10번 foward 연산을 11,12 params AG와 overlap
		#10번 forward 연산을 7,8,9 AG FSDP와 overlap

		target_comp = params_list[29]
		comms = []
		#target_comm_params = self.get_patial_param_list(params_list[70:])
		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.41, end_ratio=1.0)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		task  = Task(target_comp, 'FW', comms)


		scheduled_comms.append(task)



		#FWTOBW
		#여기에는 나머지 AGFSDP 통신 정의
		comms = []

		task  = Task(None, 'FWTOBW',comms)
		scheduled_comms.append(task)
		#make test scheduling(backward)
		#backward_params =  reversed(list(self.sharded_module.named_parameters()))

		#5번 param backward 연산을 11,12 backward params의 RS 와 overlap
		target_comp = params_list[29]
		#target_comms = self.get_patial_param_list(params_list[30:40])
		target_comm_params = get_patial_param_list(params_list[30:40], start_ratio=0.0, end_ratio=0.75)
		comm = Comm('RS', target_comm_params)
		task  = Task(target_comp, 'BW', [comm])
		scheduled_comms.append(task)


		#BWTOFW
		#여기에는 나머지 AG와 RS 통신 정의
		comms = []
		target_comm_params = get_patial_param_list(params_list[:30])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[30:40], start_ratio=0.75, end_ratio=1.0)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[40:])
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)

		#target_comm_params = self.get_patial_param_list(params_list[:70])
		target_comm_params = get_patial_param_list(params_list[:50])
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		target_comm_params = get_patial_param_list(params_list[50:], start_ratio=0.0, end_ratio=0.41)
		comm = Comm('AG', target_comm_params)
		comms.append(comm)

		#target_comm_params = params_list[40:]
		#comm = Comm('AG', target_comm_params)
		#comms.append(comm)

		task = Task(None, 'BWTOFW', comms)
		scheduled_comms.append(task)


		for task in scheduled_comms :
			for comm in task.comms :
				for param_wrap in comm.params : 
					param = param_wrap.param
					if(comm.commType == 'AG'):
						
						if(locks[comm.commType][param].locked() == False):
							locks[comm.commType][param].acquire()
		#locks['BWTOFW'].release()
		locks['FWTOBW'].acquire()



def make_schedule_from_json(params_list, scheduled_comms_init , scheduled_comms, locks, adaptive_sdp_modules, json_path='schedule.json'):
	import json
	with open("schedule.json", "r") as schedule_json:
		schedule_json = json.load(schedule_json)
	
	layer_dp_list = schedule_json['dp_type']
	schedule_list = schedule_json['schedule']

	target_comm_params = []
	
	#AG scheduled_comms_init 
	ag_init_params = []
	for dp_type, params in  zip(layer_dp_list, params_list):
		if(dp_type == 'fsdp' or dp_type == 'sdp'):
			ag_init_params.append(params)

	target_comm_params = get_patial_param_list(ag_init_params)
	comm = Comm('AG', target_comm_params)
	task = Task(None, 'BWTOFW', [comm])
	scheduled_comms_init.append(task)	

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

	comps_by_type['BW'] = list(reversed(comps_by_type['BW']))

	comm_ratio = {}
	comm_ratio['ag'] = {}
	comm_ratio['rs'] = {}
	comm_ratio['ar'] = {}

	comm_ops = ['rs', 'ag', 'ar']
	comp_types = ['BW','BWTOFW', 'FW',]
	

	for comp_type in comp_types : 
		for comp in comps_by_type[comp_type]:
			comp_param = params_list[int(comp['idx'])] if 'idx' in comp else None 
			comms = []
			for comm_op in comm_ops:
				target_comm_params = []
				for comm in comp['scheduled_comm'][comm_op]:
					param = params_list[int(comm['idx'])]

					start_ratio = 0.0
					end_ratio = 0.0
					if(comm['org_size'] == comm['param']):
						target_comm_params.append(PartiableParam(param, idx=int(comm['idx'])))
						start_ratio = 0.0
						end_ratio = 1.0
					else:
						if(param in comm_ratio[comm_op]):
							start_ratio = comm_ratio[comm_op][param]
						else:
							start_ratio = 0.0
						current_ratio = comm['param'] / comm['org_size']
						end_ratio = round(start_ratio + current_ratio, 4)
						if( start_ratio < end_ratio):
							target_comm_params.append(PartiableParam(param, start_ratio, end_ratio, comm['idx']))
							comm_ratio[comm_op][param] = end_ratio
				if(len(target_comm_params) > 0):
					comm_merge = Comm(comm_op.upper(), target_comm_params)
					comms.append(comm_merge)
			if(len(comms) > 0):
				idx = comp['idx'] if 'idx' in comp else None
				task = Task(comp_param, comp_type, comms, idx)	
				scheduled_comms.append(task)

	comm_ratio = {}
	comm_ratio['ag_fsdp'] = {}

	comm_ops = ['ag_fsdp']
	comp_types = ['FW','FWTOBW', 'BW',]
	

	for comp_type in comp_types : 
		for comp in comps_by_type[comp_type]:
			comp_param = params_list[int(comp['idx'])] if 'idx' in comp else None 

			comms = []
			for comm_op in comm_ops:
				target_comm_params = []
				for comm in comp['scheduled_comm'][comm_op]:
					param = params_list[int(comm['idx'])]

					start_ratio = 0.0
					end_ratio = 0.0
					if(comm['org_size'] == comm['param']):
						target_comm_params.append(PartiableParam(param, idx=int(comm['idx'])))
						start_ratio = 0.0
						end_ratio = 1.0
					else:
						if(param in comm_ratio[comm_op]):
							start_ratio = comm_ratio[comm_op][param]
						else:
							start_ratio = 0.0
						current_ratio = comm['param'] / comm['org_size']
						end_ratio = round(start_ratio + current_ratio, 4)
						if( start_ratio < end_ratio):
							target_comm_params.append(PartiableParam(param, start_ratio, end_ratio, comm['idx']))
							comm_ratio[comm_op][param] = end_ratio
				if(len(target_comm_params) > 0):
					comm_merge = Comm('AG', target_comm_params)
					comms.append(comm_merge)
			if(len(comms) > 0):
				idx = comp['idx'] if 'idx' in comp else None
				#find comp is scheduled in previous steps
				exist_task = None			
				for task in scheduled_comms :
					if(task.compType == 'comp_type' and task.idx == idx ):
						exist_task = task
				if(exist_task == None):
					task = Task(comp_param, comp_type, comms, idx)	
					scheduled_comms.append(task)
				else:
					exist_task.comms.extend(comms)
						


	for comm in scheduled_comms:
		print(comm)
	#import os
	#os._exit(0)
	for task in scheduled_comms :
		for comm in task.comms :
			for param_wrap in comm.params : 
				param = param_wrap.param
				if(comm.commType == 'AG'):
					if(locks[comm.commType][param].locked() == False):
						locks[comm.commType][param].acquire()

	locks['FWTOBW'].acquire()



#{'FSDP': 31, 'DP': 44, 'SDP': 32}
def make_schedules_adaptive_sdp_auto(params_list, scheduled_comms_init , scheduled_comms, locks, adaptive_sdp_modules):
		fsdp_num = adaptive_sdp_modules['FSDP']
		dp_num = adaptive_sdp_modules['DP']
		sdp_num = adaptive_sdp_modules['SDP']

		if(dp_num == 0):
			target_comm_params = get_patial_param_list([params_list[0]])
			comm = Comm('AG', target_comm_params)
			task = Task(None, 'BWTOFW', [comm])
			scheduled_comms_init.append(task)

		idx = 1
		for param in params_list[:-1] :
			comp_param = param
			if(idx < fsdp_num):
				target_comm_params = get_patial_param_list([params_list[idx]])
				comm = Comm('AG', target_comm_params )
				task = Task(comp_param, 'FW', [comm])
				scheduled_comms_init.append(task)
			elif(idx >= fsdp_num+dp_num):
				target_comm_params = get_patial_param_list([params_list[idx]])
				comm = Comm('AG', target_comm_params )
				task = Task(comp_param, 'FW', [comm])
				scheduled_comms_init.append(task)				
			idx += 1	


		if(fsdp_num > 0):
			target_comm_params = get_patial_param_list(params_list[:fsdp_num])
			comm = Comm('AG', target_comm_params )
			task = Task(None, 'FWTOBW', [comm])
			scheduled_comms_init.append(task)
		else :
			task = Task(None, 'FWTOBW', [])
			scheduled_comms_init.append(task)	


		idx = 1
		reversed_params = list(reversed(params_list))
		len_params = len(reversed_params)
		for param in reversed_params[1:]:
			comms = []
			comp_param = param
			if(idx <= sdp_num or idx > sdp_num + dp_num):
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('RS', target_comm_params )
				comms.append(comm)
			else:
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('AR', target_comm_params )
				comms.append(comm)	
				
			task = Task(comp_param, 'BW', comms)
			scheduled_comms_init.append(task)     
			idx += 1  

		comms = []
		if(dp_num == 0):
		#if(fsdp_num > 0):
			target_comm_params = get_patial_param_list([reversed_params[-1]])       
			len_target_comm_params = len(target_comm_params)
			comm = Comm('RS', target_comm_params)
			comms.append(comm)
			target_comm_params = get_patial_param_list([params_list[0]])
			comm = Comm('AG', target_comm_params)
			comms.append(comm)	
		else:
			target_comm_params = get_patial_param_list([params_list[0]])
			comm = Comm('AR', target_comm_params)
			comms.append(comm)					
		task = Task(None, 'BWTOFW', comms)
		scheduled_comms.append(task)    


		idx = 1
		for param in params_list[:-1] :
			comp_param = param
			if(idx < fsdp_num):
				target_comm_params = get_patial_param_list([params_list[idx]])
				comm = Comm('AG', target_comm_params )
				task = Task(comp_param, 'FW', [comm])
				scheduled_comms.append(task)
			elif(idx >= fsdp_num+dp_num):
				target_comm_params = get_patial_param_list([params_list[idx]])
				comm = Comm('AG', target_comm_params )
				task = Task(comp_param, 'FW', [comm])
				scheduled_comms.append(task)				
			idx += 1	



		if(fsdp_num > 0):
			target_comm_params = get_patial_param_list(params_list[:fsdp_num])
			comm = Comm('AG', target_comm_params )
			task = Task(None, 'FWTOBW', [comm])
			scheduled_comms.append(task)
		else :
			task = Task(None, 'FWTOBW', [])
			scheduled_comms.append(task)			

		idx = 1
		reversed_params = list(reversed(params_list))
		len_params = len(reversed_params)
		for param in reversed_params[1:]:
			comms = []
			comp_param = param
			if(idx <= sdp_num or idx > sdp_num + dp_num):
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('RS', target_comm_params )
				comms.append(comm)
			else:
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('AR', target_comm_params )
				comms.append(comm)	
				
			task = Task(comp_param, 'BW', comms)
			scheduled_comms.append(task)     
			idx += 1  


		for task in scheduled_comms :
			for comm in task.comms :
				for param_wrap in comm.params : 
					param = param_wrap.param
					if(comm.commType == 'AG'):
						if(locks[comm.commType][param].locked() == False):
							locks[comm.commType][param].acquire()
		if(dp_num > 0):
			locks['BWTOFW'].acquire()
		locks['FWTOBW'].acquire()


def make_schedules_adaptive_sdp(params_list, scheduled_comms_init , scheduled_comms, locks):
		target_comm_params = get_patial_param_list([params_list[0]])
		comm = Comm('AG', target_comm_params)
		task = Task(None, 'BWTOFW', [comm])
		scheduled_comms_init.append(task)

		idx = 1
		for param in params_list[:62] :
			comp_param = param
			target_comm_params = get_patial_param_list([params_list[idx]])
			comm = Comm('AG', target_comm_params )
			task = Task(comp_param, 'FW', [comm])
			scheduled_comms_init.append(task)

			idx += 1	

		target_comm_params = get_patial_param_list(params_list[:31])
		comm = Comm('AG', target_comm_params )
		task = Task(None, 'FWTOBW', [comm])
		scheduled_comms_init.append(task)


		idx = 1
		reversed_params = list(reversed(params_list))
		len_params = len(reversed_params)
		for param in reversed_params[1:]:
			comms = []
			comp_param = param
			if(idx <= 44):
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('AR', target_comm_params )
				comms.append(comm)				
			else:
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('RS', target_comm_params )
				comms.append(comm)

			task = Task(comp_param, 'BW', comms)
			scheduled_comms_init.append(task)     
			idx += 1   			
		comms = []
		target_comm_params = get_patial_param_list([reversed_params[-1]])       
		len_target_comm_params = len(target_comm_params)
		comm = Comm('RS', target_comm_params)
		comms.append(comm)
		target_comm_params = get_patial_param_list([params_list[0]])
		comm = Comm('AG', target_comm_params)
		comms.append(comm)		
		task = Task(None, 'BWTOFW', comms)
		scheduled_comms.append(task)    


		idx = 1
		for param in params_list[:62] :
			comp_param = param
			target_comm_params = get_patial_param_list([params_list[idx]])
			comm = Comm('AG', target_comm_params )
			task = Task(comp_param, 'FW', [comm])
			scheduled_comms.append(task)

			idx += 1	

		target_comm_params = get_patial_param_list(params_list[:31])
		comm = Comm('AG', target_comm_params )
		task = Task(None, 'FWTOBW', [comm])
		scheduled_comms.append(task)


		idx = 1
		reversed_params = list(reversed(params_list))
		len_params = len(reversed_params)
		for param in reversed_params[1:]:
			comms = []
			comp_param = param
			if(idx <= 44):
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('AR', target_comm_params )
				comms.append(comm)				
			else:
				target_comm_params =get_patial_param_list([reversed_params[idx-1]])  
				comm = Comm('RS', target_comm_params )
				comms.append(comm)

			task = Task(comp_param, 'BW', comms)
			scheduled_comms.append(task)     
			idx += 1   		


		for task in scheduled_comms :
			for comm in task.comms :
				for param_wrap in comm.params : 
					param = param_wrap.param
					if(comm.commType == 'AG'):
						if(locks[comm.commType][param].locked() == False):
							locks[comm.commType][param].acquire()
		#if(dp_num > 0):
		#	locks['BWTOFW'].acquire()
		locks['FWTOBW'].acquire()