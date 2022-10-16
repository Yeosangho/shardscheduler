import os
import threading
import time
import datetime
import argparse 
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from test_cases import make_schedule_from_json
from auto_wrap_custom import enable_wrap, auto_wrap, wrap
from algo import schedule
from torch_scheduler import ShardScheduler

import nltk


def module_check(module):
	#if (len(list(module.children())) == 0 ):
		#print(module)
		#print(module.data.size())
	total_wrapped_layers = 0
	for name, child in module.named_children():
		count = module_check(child)
		total_wrapped_layers += count
	if (len(list(module.children())) == 0 ):
		return 1
	else :
		return total_wrapped_layers
	return total_wrapped_layers

def _release_lock(lock, condition):
    if lock.locked():
        lock.release()
    with condition :
        condition.notify_all()	

os.environ['MASTER_ADDR'] = '210.107.197.219'
os.environ['MASTER_PORT'] = '30005'
os.environ["NCCL_SOCKET_IFNAME"]="eno1,eth0"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]='1'
parser = argparse.ArgumentParser()
parser.add_argument('--rank', dest='rank', default=0, type=int)
parser.add_argument('--sdp_ratio', default=0, type=float)
parser.add_argument('--fsdp_ratio', default=0, type=float)
parser.add_argument('--dp_ratio', default=0, type=float)
args = parser.parse_args()
world_size = 2
rank = args.rank
adaptive_shard_ratio = {}
adaptive_shard_ratio['dp'] = args.dp_ratio
adaptive_shard_ratio['sdp'] = args.sdp_ratio
adaptive_shard_ratio['fsdp'] = args.fsdp_ratio
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

nltk.download('punkt')

df = pd.read_csv ("review_dataset/Reviews.csv")  
df = df[:600]
print(df)
print(len(df))
df.dropna(inplace=True) #remove NA values
reviews = df.Text.copy() #just use the main bio text in this example

doc_lengths = []

for review in reviews:

    # get rough token count distribution
    tokens = nltk.word_tokenize(review)

    doc_lengths.append(len(review))

doc_lengths = np.array(doc_lengths)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
batch_size = 2

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 

dataset = GPT2Dataset(reviews, tokenizer, max_length=768)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()




# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

# some parameters I cooked up that work reasonably well

epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 100
partition_threshold = 20000
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
#optimizer = Adam(model.parameters(),
#                  lr = learning_rate,
#                  eps = epsilon
#                )

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
#scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                            num_warmup_steps = warmup_steps, 
#                                            num_training_steps = total_steps)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


total_t0 = time.time()

training_stats = []

model = model.to(device)


######
_locks = {}
_conditions = {} 

_rs_locks = {}
_ag_locks = {}
_ar_locks = {}
_ag_fsdp_locks = {}

_rs_conditions = {}
_ag_conditions = {}
_ar_conditions = {}
_ag_fsdp_conditions = {} 


_forward_locks = {}
_backward_locks = {}

_forward_conditions = {}
_backward_conditions = {}

_lazy_init_locks = {}
_lazy_init_conditions = {}

_partition_counts = {}
_scheduled_comms = []
_schedule_comm_init = []
_done_counts = {}
model_parameter_names = {}
profiled_memory_utilization = []

######
health_check_scheduler_thread = threading.Lock()		
health_check_main_proc = threading.Lock()
comm_stream = torch.cuda.Stream()
profile_target_layer = []
wrap_params = dict( mixed_precision=False, flatten_parameters=True, 

						locks=_locks,
						health_check_main_proc=health_check_main_proc, 

						conditions=_conditions, 

						profile_layer = profile_target_layer,

						memory_record=profiled_memory_utilization,
						
						model_parameter_names=model_parameter_names
						)

len_module = module_check(model)
fsdp_num =  int(len_module * adaptive_shard_ratio['fsdp'])
sdp_num =  int(len_module * adaptive_shard_ratio['sdp'])
dp_num = int(len_module) -fsdp_num - sdp_num
adaptive_sdp = {}
adaptive_sdp['FSDP'] = fsdp_num
adaptive_sdp['DP'] = dp_num
adaptive_sdp['SDP'] = sdp_num 

with enable_wrap(**wrap_params):
	sharded_module = auto_wrap(adaptive_sdp, model)
	print(len(list(sharded_module.named_parameters())))
	adaptive_sdp_modules = {}
	adaptive_sdp_modules['FSDP'] = 0 
	adaptive_sdp_modules['SDP'] = 0
	adaptive_sdp_modules['DP'] = 0

	for n, p in sharded_module.named_parameters():
		print(n)
		if('_fsdp_wrapped_module' in n):
			adaptive_sdp_modules['FSDP'] += 1
		elif('_sdp_wrapped_module' in n):
			adaptive_sdp_modules['SDP'] += 1
		elif('_dp_wrapped_module' in n):
			adaptive_sdp_modules['DP'] += 1
	print(adaptive_sdp_modules)

	for n, p in sharded_module.named_parameters():
		#print(n)
		_partition_counts[p] = (p.numel() // partition_threshold) + 1
		_done_counts[p] = 0

		_rs_locks[p] = threading.Lock()
		_ag_locks[p] = threading.Lock()
		_ar_locks[p] = threading.Lock()
		_ag_fsdp_locks[p] = threading.Lock()

		_forward_locks[p] = threading.Lock()
		_backward_locks[p] = threading.Lock()

		_forward_locks[p].acquire()
		#self._rs_locks[p].acquire()
		#self._ag_fsdp_locks[p].acquire()
		_backward_locks[p].acquire()

		_rs_conditions[p] = threading.Condition(threading.Lock())
		_ag_conditions[p] = threading.Condition(threading.Lock())
		_ar_conditions[p] = threading.Condition(threading.Lock())
		_ag_fsdp_conditions[p] = threading.Condition(threading.Lock())

		_forward_conditions[p] = threading.Condition(threading.Lock())
		_backward_conditions[p] = threading.Condition(threading.Lock())

		_lazy_init_locks[p] = threading.Lock()
		_lazy_init_conditions[p] = threading.Condition(threading.Lock())

		model_parameter_names[p] = n

	_locks["FW"] 	    = _forward_locks
	_locks["BW"]	    = _backward_locks 
	_locks["AG"] 		= _ag_locks
	_locks["AR"]		= _ar_locks
	_locks["FWTOBW"]   = threading.Lock()
	_locks["BWTOFW"]   = threading.Lock()
	_locks["BWTOFW"].acquire()
	#self._locks["AGFSDP"]   = self._ag_fsdp_locks
	#self._locks["RS"]       = self._rs_locks

	_conditions["FW"]        = _forward_conditions    
	_conditions["BW"]        = _backward_conditions    
	_conditions["AG"]        = _ag_conditions  
	_conditions["AR"]		  = _ar_conditions
	_conditions["FWTOBW"]   = threading.Condition(threading.Lock())
	_conditions["BWTOFW"]   = threading.Condition(threading.Lock())

	#self._conditions["AGFSDP"]    = self._ag_fsdp_conditions       
	#self._conditions["RS"]        = self._rs_conditions    
	schedule(adaptive_sdp_modules, "layer_bench_gpt2.csv")
	params_list = list(sharded_module.parameters())
	make_schedule_from_json(params_list, _schedule_comm_init, _scheduled_comms, _locks, adaptive_sdp_modules)
	#make_schedule_wfbp_sdp(params_list, self._schedule_comm_init, self._scheduled_comms, self._locks)
	#os._exit(1)
	dist.barrier()
	print(f"before init optimizer  {torch.cuda.memory_allocated() / 1024 /1024}") 
	#self.optimizer = torch.optim.SGD(self.sharded_module.parameters() , lr=0.001, momentum=0.9, nesterov=True)
	optimizer = torch.optim.Adam(sharded_module.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	optimizer = ShardScheduler(sharded_module, sharded_module.named_parameters(), 2, rank, optimizer,
	                                partition_threshold, _done_counts, _partition_counts,
									health_check_scheduler_thread,
									_locks,

									_conditions,

									profile_target_layer, 

									10**6, comm_stream, _schedule_comm_init, _scheduled_comms)

model = sharded_module
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)

        b_masks = batch[1].to(device)
        model.zero_grad()        
        if _locks['BWTOFW'].locked():   
            _release_lock(_locks['BWTOFW'], _conditions['BWTOFW'])	
        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        if _locks['FWTOBW'].locked():   
            _release_lock(_locks['FWTOBW'], _conditions['FWTOBW'])	

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss


        loss.backward()

        if _locks['BWTOFW'].locked():   
            _release_lock(_locks['BWTOFW'], _conditions['BWTOFW'])	
        if(step == 3):
            break

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
