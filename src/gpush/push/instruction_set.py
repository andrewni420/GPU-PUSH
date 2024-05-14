from __future__ import annotations
from .instruction import Instruction
from .instructions.utils import create_instructions, InstructionWrapper
import numpy as np 
from scipy.special import softmax
from typing import Callable, Union, List 
import time 

class InstructionSet(dict[str,Instruction]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logprobs = {}
        self.rng = np.random.default_rng()
        self.updated = False 
        self.sampler = None 

    def register(self, instr: Instruction, logprob=0):
        self[instr.name] = instr 
        self.logprobs[instr.name]=logprob
        self.updated = True

    def unregister(self, instr: Instruction):
        del self[instr.name]
        self.updated = True 
    
    def sample(self, n: int = 1, squeeze: bool = True) -> Instruction | List[Instruction]:
        if self.updated or self.sampler is None:
            keys = list(self.keys())
            self.sampler = {"instructions": [self[k] for k in keys], "probs": softmax(np.array([self.logprobs.get(k,0) for k in keys]))}
            self.updated = False 
        if n==1 and squeeze:
            return self.rng.choice(self.sampler["instructions"], p=self.sampler["probs"])  
        else:
            return self.rng.choice(self.sampler["instructions"], self.sampler["probs"])
    
    def filter(self, fn: Callable) -> InstructionSet:
        instr = {k:v for k,v in self.items() if fn(v)}
        logprobs = {k:self.logprobs.get(k,0) for k in instr.keys()}
        instr = InstructionSet(**instr)
        instr.logprobs = logprobs
        return instr 
    
    def unpack_register(self, fn=create_instructions):
        def wrap(wrapper: InstructionWrapper):
            instructions = wrapper.apply(fn)
            for instr in instructions:
                self.register(instr)
            return wrapper
        return wrap 
    

GLOBAL_INSTRUCTIONS = InstructionSet()

