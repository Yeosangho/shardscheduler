from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from transformers import AdamW, get_linear_schedule_with_warmup



class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            raise ValueError("Not a valid model, please provide a 'nn.Module' instance.")
        for n,p in model.named_parameters():
            print(n)
        self.model = model
        self._parameter_names = {}
        self._seq_keys = []
        self._set_parameter_names(self.model, "")
        self._set_seq_keys(self.model, "")


        self._backward_seq_keys = []
        self._backward_key_sizes = []
        self._grad_accs = []
        self._handles = {}
        self.hook_done = False
        self._start = time.time()
        self._register_hooks(self.model, "")
        self._is_profiling = False

        self._forward_seq_keys = []
        self._forward_key_sizes = []
        self._forward_handles = {}

    def _set_parameter_names(self, module, name):
        for child_name, child in list(module.named_children()):
            self._set_parameter_names(child, f"{name}.{child_name}")

        if (len(list(module.children())) == 0 and 'relu' not in name ):
            self._parameter_names[module] = name

    def _set_seq_keys(self, module, name):
        for child_name, child in list(module.named_children()):
            self._set_seq_keys(child, f"{name}.{child_name}")

        if (len(list(module.children())) == 0 and  'relu' not in name ):
            self._seq_keys.append(name)


    def _register_hooks(self, module, name):

        for child_name, child in list(module.named_children()):
            print(len(list(module.children())))
            self._register_hooks(child, f"{name}.{child_name}")

        if (len(list(module.children())) == 0 and 'relu' not in name and 'pool' not in name ):
            module.register_backward_hook(self._make_hook(name, module))
            #module.register_backward_hook(self._print_hook())

            module.register_forward_hook(self._make_forward_hook(name, module))

    def _print_hook(self):
        def hook(*ignore):
            print("dup hook test")
        return hook
    def _make_hook(self, name, p):
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)

            if len(self._backward_seq_keys) != len(self._seq_keys) and name not in self._backward_seq_keys:
                param_sum = 0 
                for param in p.parameters():
                    param_sum += param.numel()
                self._backward_seq_keys.append(name)
                self._backward_key_sizes.append(param_sum)
                #print(f"name : {name}, param_sum {param_sum}" )
            
            if name not in self._handles:
                self._handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            self._handles[name].append(ct - self._start)
        return hook

    def _make_forward_hook(self, name, p):
        def hook(*ignore):
            if not self._is_profiling:
                return
            name = self._parameter_names.get(p)
            if len(self._forward_seq_keys) != len(self._seq_keys) and name not in self._forward_seq_keys:    
                param_sum = 0 
                for param in p.parameters():
                    param_sum += param.numel()
                self._forward_key_sizes.append(param_sum)
                self._forward_seq_keys.append(name)

            if name not in self._forward_handles:
                self._forward_handles[name] = []
            torch.cuda.synchronize()
            ct = self._timestamp(name)
            self._forward_handles[name].append(ct - self._start)
        return hook

    def reset_start(self):
        self._start = time.time()

    def reset(self):
        self._start = time.time()
        self._handles.clear()

    def stop(self):
        self._is_profiling = False

    def start(self):
        self._is_profiling = True
        self._start = time.time()

    def get_backward_seq_keys(self):
        return self._backward_seq_keys

    def get_backward_key_sizes(self):
        return self._backward_key_sizes

    def get_forward_seq_keys(self):
        return self._forward_seq_keys

    def get_forward_key_sizes(self):
        return self._forward_key_sizes


    def get_forward_layerwise_times(self):
        num_trials = len(self._forward_handles[self._seq_keys[0]])

        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._forward_seq_keys):
                t = self._forward_handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                #print(f"{k} {t-s}")
                total += (t-s)
                s = total

            layerwise_times_multipletest.append(layerwise_times)
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0)
        return layerwise_times, np.mean(totals)



    def get_layerwise_times(self):
        num_trials = len(self._handles[self._seq_keys[0]])
        layerwise_times_multipletest = []
        totals = []
        for j in range(num_trials):
            s = 0
            total = 0.0
            layerwise_times = [] # from the last layer to the first layer
            #for i, k in enumerate(self._seq_keys[::-1]):
            for i, k in enumerate(self._backward_seq_keys):
                t = self._handles[k][j]
                #print('name: ', k, ' diff: ', t-s)
                layerwise_times.append(t-s)
                total += (t-s)
                s = total
                #if(total < 0):
                #    print(k)                
            layerwise_times_multipletest.append(layerwise_times)
            totals.append(total)
        array = np.array(layerwise_times_multipletest)
        layerwise_times = np.mean(array, axis=0)
        return layerwise_times, np.mean(totals)

    def _timestamp(self, name):
        return time.time()


def benchmark(model, criterion, input_shape, input_dtype, label_shape, label_dtype):
    # Benchmark to achieve the backward time per layer
    p = Profiling(model)
    # Warmup
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50

    for i in range(iteration+warmup):

        inputs = torch.rand(input_shape, dtype=input_dtype).cuda()
        labels = torch.randint(0, 1, label_shape, dtype=label_dtype).cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        torch.cuda.synchronize()

        if i >= warmup:
            p.start()
        loss.backward()
        torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    forward_times, _ = p.get_forward_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    return seq_keys[::-1], layerwise_times[::-1], forward_times[::-1],  p.get_backward_key_sizes()[::-1]

def benchmark_gpt2(model, input_shape, input_dtype, label_shape, label_dtype):
    # Benchmark to achieve the backward time per layer
    p = Profiling(model)
    # Warmup
    warmup = 5 # warmup should be 0 on some GPUs (e.g., P102-100)
    iteration = 50

    for i in range(iteration+warmup):
        inputs = torch.randint(0, 50000, input_shape, dtype=input_dtype).cuda()
        labels = torch.randint(0, 50000, label_shape, dtype=label_dtype).cuda()
        masks = torch.randint(0, 1, label_shape, dtype=label_dtype).cuda()

        # forward + backward + optimize
        outputs = model(  inputs,
                          labels=labels, 
                          attention_mask = masks,
                          token_type_ids=None
                        )

        loss = outputs[0]          
        torch.cuda.synchronize()

        if i >= warmup:
            p.start()
        loss.backward()
        torch.cuda.synchronize()
    layerwise_times, sum_total = p.get_layerwise_times()
    forward_times, _ = p.get_forward_layerwise_times()
    seq_keys = p.get_backward_seq_keys()
    p.stop()
    return seq_keys[::-1], layerwise_times[::-1], forward_times[::-1],  p.get_backward_key_sizes()[::-1]

class CommunicationProfiler(object):
    def __init__(self, comm_op, sizes=None):
        self.comm_op = comm_op
        self.sizes = sizes
        self.all_reduce_stream = torch.cuda.Stream()

    def benchmark(self, num_iters=30):
        if self.sizes is None:
            small_sizes = [8*1024*i for i in range(50, 200)] # 1K to 1M
            large_sizes = [1024*1024*i for i in range(8)] # 1M to 512M
            sizes = small_sizes+large_sizes
        else:
            sizes = self.sizes
        warmup = 100
        size = 1024
        tensor = torch.rand(size).float().cuda()
        tensor_list = [torch.zeros_like(tensor).cuda() for _ in range(2)]

        stime = time.time()
        for i in range(warmup):
            name = 'warmup-%d' % i
            self.comm_op(tensor_list, tensor)
            #print(tensor)
        etime = time.time()
        elapsed_times = []
        for s in sizes:
            tensor = torch.rand(s).float().cuda()
            tensor_list = [torch.zeros_like(tensor).cuda() for _ in range(2)]

            #option 1
            #stime = time.time()
            #for i in range(num_iters):
            #    with torch.cuda.stream(self.all_reduce_stream):            
            #            #print(f"before {t[0]}")
            #             self.comm_op(tensor_list, tensor)
            #            #handle.wait()
            #    torch.cuda.default_stream().wait_stream(self.all_reduce_stream)
            #    torch.cuda.synchronize()
            #etime = time.time()

            #option 2
            stime = time.time()
            for i in range(num_iters):
                handle = self.comm_op(tensor_list, tensor, async_op=True)
                while not handle.is_completed():
                    time.sleep(0.001)
                handle.wait()
            etime = time.time()


            print("!")
            elapsed_times.append((etime-stime)/num_iters)

        return sizes, elapsed_times


