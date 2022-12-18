import math 
a = 100
b = 5
param_num = (b/(b+1)) * a * 1024 * 1024 / 4
shard_size = param_num/ b 
size = int(shard_size * (b+1)) * 4
print(size/(1024*1024))