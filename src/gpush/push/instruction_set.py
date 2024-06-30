from __future__ import annotations
from .instruction import Instruction
from .instructions.utils import create_instructions, InstructionWrapper
import numpy as np 
from scipy.special import softmax
from typing import Callable, Union, List 
import time 

class InstructionSet(dict[str,Instruction]):
    """A dictionary mapping the names of instructions to `Instruction` objects"""
    
    logprobs: dict[str,float]
    "Maps the names of instructions to their log-probability of being chosen"
    updated: bool 
    "Have new instructions been added since the probabilities were last calculated?"
    rng: np.random.Generator
    "Random number generator"
    sampler: dict[str,list]
    """A dictionary `{"instructions": [instructions], "probs": [probs]}` containing all of the instructions, along with their probabilities of being chosen"""
    stack_to_instr: dict[frozenset,list[Instruction]]
    "A dictionary mapping sets of stacks to instructions using those sets of stacks."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logprobs = {}
        self.rng = np.random.default_rng()
        self.updated = False 
        self.sampler = None 
        self.stack_to_instr = {}

    def register(self, instr: Instruction, logprob: float = 0):
        """Register an instruction in the instruction set, along with the associated log-probability
        
        Parameters:
            instr (Instruction): The instruction to be registered
            logprob (float): The log-probability of choosing this instruction"""
        
        self[instr.name] = instr 
        self.logprobs[instr.name]=logprob
        self.updated = True
        req_stacks = frozenset(instr.required_stacks())
        if req_stacks in self.stack_to_instr:
            self.stack_to_instr[req_stacks].append(instr)
        else:
            self.stack_to_instr[req_stacks] = [instr]

    def register_all(self, instr_set: InstructionSet):
        """Registers all instructions contained in the given instruction set. Ignores repeat instructions"""
        for k,v in instr_set.items():
            if k not in self:
                self.register(v,instr_set.logprobs[v.name])

    def unregister(self, instr: Instruction):
        """Unregisters a previously registered instruction"""
        del self[instr.name]
        self.stack_to_instr[frozenset(instr.required_stacks())].remove(instr)
        self.updated = True 

    def unregister_all(self, instr_set: InstructionSet):
        """Unregisters all instructions contained in the given instruction set. Ignores instructions not present in `self`"""
        for k,v in instr_set.items():
            if k in self:
                self.unregister(v)
    
    def sample(self, n: int = 1, squeeze: bool = True) -> Instruction | List[Instruction]:
        """Randomly samples n instructions
        
        Parameters:
            n (int): The number of instructions to sample
            squeeze (bool): Whether to squeeze the list down to a single element when `n==1`
            
        Returns:
            Instruction | list[Instruction]: The sampled instruction(s)"""
        
        if self.updated or self.sampler is None:
            keys = list(self.keys())
            self.sampler = {"instructions": [self[k] for k in keys], "probs": softmax(np.array([self.logprobs.get(k,0) for k in keys]))}
            self.updated = False 
        if n==1 and squeeze:
            return self.rng.choice(self.sampler["instructions"], p=self.sampler["probs"])  
        else:
            return self.rng.choice(self.sampler["instructions"], self.sampler["probs"])
    
    def filter(self, fn: Callable[[Instruction], bool]) -> InstructionSet:
        """Filters the instruction set based on a filtering function
        
        Parameters:
            fn (Callable): A function to decide whether to keep each instruction
        
        Returns:
            InstructionSet: The filtered instruction set"""
        
        instr = {k:v for k,v in self.items() if fn(v)}
        logprobs = {k:self.logprobs.get(k,0) for k in instr.keys()}
        instr = InstructionSet(**instr)
        instr.logprobs = logprobs
        return instr 
    
    def unpack_register(self, fn: Callable = create_instructions) -> Callable[[InstructionWrapper], InstructionWrapper]:
        """A utility function wrapper to create multiple variants of a function and register them in the instruction set
        
        Parameters:
            fn (Callable): A function that takes in a function, as well as whatever information is present in the InstructionWrapper,
            and returns a list of `Instruction` objects.
            
        Returns:
            Callable: A wrapper function that takes a given `InstructionWrapper`, makes and registers the variants, and then returns the 
            `InstructionWrapper` unchanged."""
        def wrap(wrapper: InstructionWrapper) -> InstructionWrapper:
            instructions = wrapper.apply(fn)
            for instr in instructions:
                self.register(instr)
            return wrapper
        return wrap 
    
    def stack_instructions(self, stacks: List[str]) -> InstructionSet:
        """Get the instructions that use only the provided stacks
        
        Parameters:
            stacks (list[str]): The list of available stacks
            
        Returns:
            InstructionSet: The set of instructions that use only those stacks"""
        
        stacks = frozenset(stacks)
        instr = []
        for k in self.stack_to_instr.keys():
            if k.issubset(stacks):
                instr.extend(self.stack_to_instr[k])
        iset = InstructionSet()
        for i in instr:
            iset.register(i,self.logprobs[i.name])
        return iset 
    
GLOBAL_INSTRUCTIONS = InstructionSet()
"""A global `InstructionSet` containing all defined instructions"""

ACTIVATION_INSTRUCTIONS = InstructionSet()
"""A global `InstructionSet` containing all activation function instructions"""
