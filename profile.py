import torchvision.models as models
import torch.nn as nn

import torch.distributed as dist
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from profiling import benchmark, CommunicationProfiler
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
dist.init_process_group(backend='nccl', world_size=2, rank=args.rank)
group = dist.new_group(backend='gloo')
#프로파일러 코드들 ==> 독립된 프로세스로 호출하고, 프로파일링 결과를 파일에 기록하도록 구성하기.
#(이래야 프로파일링 결과로 실행되는 학습에 영향(메모리, 통신 백엔드)을 안줌)

target_shape = 32,
target_dtype = torch.long
input_shape = 32, 3, 32, 32,
input_dtype = torch.float
criterion = nn.CrossEntropyLoss()
model = ResNet(Bottleneck, [3, 4, 6, 3]) #it means "resnet18 model"
model.cuda()
print(len(list(model.named_parameters())))
for n,p in model.named_parameters():
    print(p.numel())
layer_bench = benchmark(model, criterion, input_shape, input_dtype, target_shape, target_dtype)

btimes = torch.Tensor(layer_bench[1].copy())
dist.all_reduce(btimes, op=dist.ReduceOp.MAX, group=group)

ftimes = torch.Tensor(layer_bench[2].copy())
dist.all_reduce(ftimes, op=dist.ReduceOp.MAX, group=group)

with open("layer_bench.csv", "w") as f:
    wr = csv.writer(f)
    for name, btime, ftime, num in zip(layer_bench[0], btimes, ftimes, layer_bench[3]):
        wr.writerow([name, btime.item(), ftime.item(), num])
proc_exec = True
target_mem = 7.3


comm_profiler = CommunicationProfiler(dist.all_gather)
sizes, times  = comm_profiler.benchmark()
print(sizes)
print(times)
def _fit_linear_function(x, y):
    X = np.array(x).reshape((-1, 1)) * 4
    Y = np.array(y)
    model = LinearRegression()
    model.fit(X, Y)
    alpha = model.intercept_
    beta = model.coef_[0]
    return alpha, beta
alpha, beta = _fit_linear_function(sizes, times)

print(alpha, beta)
net_info = torch.Tensor([alpha, beta])
dist.broadcast(net_info, src=0, group=group)
with open("net_bench.csv", "w") as f:
    f.write(f"{net_info[0]}, {net_info[1]}")