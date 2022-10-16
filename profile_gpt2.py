import torchvision.models as models
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

import torch.distributed as dist
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from profiling import benchmark, benchmark_gpt2, CommunicationProfiler
from sklearn.linear_model import LinearRegression
import numpy as np
import csv
import argparse
import torch 
import os


parser = argparse.ArgumentParser()
parser.add_argument('--rank', dest='rank', default=0, type=int)
args = parser.parse_args()
os.environ['MASTER_ADDR'] = '210.107.197.219'
os.environ['MASTER_PORT'] = '30002'
os.environ["NCCL_SOCKET_IFNAME"]="eno1,eth0"

dist.init_process_group(backend='nccl', world_size=2, rank=args.rank)
group = dist.new_group(backend='gloo')
#프로파일러 코드들 ==> 독립된 프로세스로 호출하고, 프로파일링 결과를 파일에 기록하도록 구성하기.
#(이래야 프로파일링 결과로 실행되는 학습에 영향(메모리, 통신 백엔드)을 안줌)

target_shape = 2, 768,
target_dtype = torch.long
input_shape = 2, 768,
input_dtype = torch.long
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# instantiate the model
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()


print(len(list(model.named_parameters())))
for n,p in model.named_parameters():
    print(p.numel())
layer_bench = benchmark_gpt2(model, input_shape, input_dtype, target_shape, target_dtype)

btimes = torch.Tensor(layer_bench[1].copy())
dist.all_reduce(btimes, op=dist.ReduceOp.MAX, group=group)

ftimes = torch.Tensor(layer_bench[2].copy())
dist.all_reduce(ftimes, op=dist.ReduceOp.MAX, group=group)

with open("layer_bench_gpt2.csv", "w") as f:
    wr = csv.writer(f)
    for name, btime, ftime, num in zip(layer_bench[0], btimes, ftimes, layer_bench[3]):
        wr.writerow([name, btime.item(), ftime.item(), num])
proc_exec = True
target_mem = 7.3

