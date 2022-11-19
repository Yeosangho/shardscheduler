# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast

import torch

def register_memory_profiling_hooks(module, name):
    for child_name, child in list(module.named_children()):
        register_memory_profiling_hooks(child, f"{name}.{child_name}")
    if (len(list(module.children())) == 0 and 
        'act' not in name and
        'drop' not in name and
        'relu' not in name and 
        'pool' not in name ):
        module.register_backward_hook(_make_hook(name, module))
        module.register_forward_hook(_make_forward_hook(name, module))
        
def _make_hook(name, p):
    def hook(*ignore):
        print({torch.cuda.memory_allocated()/1024**2})
    return hook

def _make_forward_hook(name, p):
    def hook(*ignore):
        print({torch.cuda.memory_allocated()/1024**2})
    return hook