
import os, threading
import torch.distributed as dist

import transformers
from auto_wrap_custom import enable_wrap, auto_wrap, wrap

os.environ['MASTER_ADDR'] = '210.107.197.219'
os.environ['MASTER_PORT'] = '30005'
os.environ["NCCL_SOCKET_IFNAME"]="eno1,eth0"
dist.init_process_group(backend='nccl', world_size=2, rank=0)

gpt2_configuration = transformers.GPT2Config()

gpt2_model = transformers.GPT2Model(gpt2_configuration)
_locks = {}
_conditions = {} 
profiled_memory_utilization = []
model_parameter_names = {}
health_check_main_proc = threading.Lock()
profile_target_layer = []
wrap_params = dict( mixed_precision=False, flatten_parameters=True, 

						locks=_locks,
						health_check_main_proc=health_check_main_proc, 

						conditions=_conditions, 

						profile_layer = profile_target_layer,

						memory_record=profiled_memory_utilization,
						
						model_parameter_names=model_parameter_names
						)

adaptive_sdp = {}
adaptive_sdp['FSDP'] = 10000000000
adaptive_sdp['DP'] = 0
adaptive_sdp['SDP'] = 0 


with enable_wrap(**wrap_params):
	sharded_module = auto_wrap(adaptive_sdp, gpt2_model)