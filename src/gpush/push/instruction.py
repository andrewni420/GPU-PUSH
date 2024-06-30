from __future__ import annotations
from abc import ABC, abstractmethod 
from typing import Callable, Set, Union
from .state import PushState
from .dag.expr import Function, Parameter, Input
from .dag.shape import Shape 
import numpy as np  

class Instruction(ABC):
    def __init__(self, name: str, code_blocks: int, docstring = None):
        self.name = name
        self.code_blocks = code_blocks
        self.docstring = docstring 

    def __eq__(self, other: Instruction) -> bool:
        return self.name==other.name 
    
    def __call__(self, state: PushState) -> PushState:
        return self.evaluate(state).step()
    
    def __str__(self):
        return self.name 

    def __repr__(self):
        return self.name 
    
    @abstractmethod 
    def evaluate(self, state: PushState) -> PushState:
        pass 

    @abstractmethod
    def required_stacks(self) -> Set[str]:
        pass 


class StateToStateInstruction(Instruction):
    "An instruction that applies a function which takes a `PushState` and returns a `PushState`"
    def __init__(self, name: str, fn: Callable[[PushState],PushState], stacks_used: Set[str], code_blocks: int, docstring = None, validator: Callable[[PushState], bool] = None):
        self.name = name
        self.fn = fn 
        self.stacks_used = stacks_used
        self.code_blocks = code_blocks
        self.docstring = docstring 
        self.validator = validator

    def __eq__(self, other: Instruction) -> bool:
        return self.name==other.name 
    
    def evaluate(self, state: PushState) -> PushState:
        if self.validator is not None:
            if not self.validator(state):
                return state 
        return self.fn(state)

    def required_stacks(self) -> Set[str]:
        return self.stacks_used 

class SimpleInstruction(Instruction):
    """A simple instruction that takes some arguments, taken from a `PushState`, optionally checks for noops with a validator function, 
    and returns some outputs to be pushed onto the `PushState`"""
    def __init__(self, name: str, fn: Callable, input_stacks: Union[dict[str,int], tuple[str], str], output_stacks: Union[dict[str,int], tuple[str], str], code_blocks: int, docstring=None, validator: Callable = None):
        super().__init__(name, code_blocks, docstring=docstring)
        self.fn = fn 
        self.input_stacks = (input_stacks,) if isinstance(input_stacks,str) else input_stacks
        self.output_stacks = output_stacks
        self.validator = validator
        self.takes_nsteps = False
    
    def evaluate(self, state: PushState) -> PushState:
        # Get arguments and check if there's enough present
        args = state.observe(self.input_stacks)
        if args is None:
            return state 
        
        if self.validator is not None:
            if isinstance(self.input_stacks,tuple) and not self.validator(*args):
                return None 
            if isinstance(self.input_stacks,dict) and not self.validator(**args):
                return None 
            
            
        
        if isinstance(self.input_stacks, dict):
            result = self.fn(state.nsteps, **args) if self.takes_nsteps else self.fn(**args)
        else:
            result = self.fn(state.nsteps, *args) if self.takes_nsteps else self.fn(*args)

        if result is None:
            return state 
        

        if isinstance(self.output_stacks, tuple):
            result = {k:[v] for k,v in zip(self.output_stacks, result)}
        elif isinstance(self.output_stacks,str):
            result = {self.output_stacks:[result]}


        to_pop = self.input_stacks
        if not isinstance(to_pop,dict):
            to_pop = {k:1 for k in to_pop}

        state.pop_from_stacks(to_pop)
        return state.push_to_stacks(result)
    
    def required_stacks(self) -> Set[str]:
        def get_stacks(s):
            if isinstance(s,dict):
                return set(s.keys())
            if isinstance(s,str):
                return {s}
            return set(s)
        return get_stacks(self.input_stacks).union(get_stacks(self.output_stacks))
    
class SimpleExprInstruction(SimpleInstruction):
    """A simple instruction that constructs an Expression"""
    def __init__(self, name: str, fn: Callable, signature: Callable, input_stacks: tuple[str], output_stack: str, code_blocks: int, docstring=None, validator: Callable = None):
        def make_expression(nsteps, *args, **kwargs):
            shape, dtype = self.signature(**kwargs) if isinstance(input_stacks,dict) else self.signature(*args)
            # print(f"shape {shape} dtype {dtype}")
            if shape is None or dtype is None:
                return None 
            children=kwargs if isinstance(input_stacks,dict) else tuple(args)
            return Function(nsteps, fn, children=children, shape=shape, dtype=dtype)
        super().__init__(name, make_expression, input_stacks, output_stack, code_blocks, docstring=docstring, validator=validator)
        self.signature = signature
        self.takes_nsteps = True

    
class InputInstruction(Instruction):
    "Pushes a symbolic Input Expression representing one of the inputs to the neural network"
    def __init__(self, name: str, input_idx: int, output_stack: str, docstring: str = None):
        super().__init__(name, 0, docstring=docstring)
        self.input_idx = input_idx 
        self.output_stack = output_stack
    
    def evaluate(self, state:PushState, nsteps: int = 0) -> PushState:
        input = Input(nsteps, self.input_idx, shape=state.input[self.input_idx]["shape"], dtype=state.input[self.input_idx]["dtype"])
        return state.push_to_stacks({self.output_stack: [input]})
    
    def required_stacks(self) -> Set[str]:
        return {self.output_stack}
    
class ParamInstruction(Instruction):
    "Pushes a symbolic Parameter Expression representing one of the parameters of the neural network"
    def __init__(self, name: str, shape: Shape, dtype: str, output_stack: str, docstring: str = None):
        super().__init__(name, 0, docstring=docstring)
        self.shape = shape 
        self.dtype = dtype 
        self.output_stack = output_stack
    
    def evaluate(self, state:PushState) -> PushState:
        input = Parameter(state.nsteps, len(state.params), shape=self.shape, dtype=self.dtype)
        state.params.append({"shape": self.shape, "dtype": self.dtype})
        return state.push_to_stacks({self.output_stack: [input]})
    
    def required_stacks(self) -> Set[str]:
        return {self.output_stack}
    
class ParamBuilderInstruction(Instruction):
    """Builds and pushes a symbolic Parameter Expression representing one of the parameters of the neural network. 
    Uses elements from the integer stack to fill in None values in the given shape."""
    def __init__(self, name: str, shape: Shape, dtype: str, output_stack: str, docstring: str = None):
        super().__init__(name, 0, docstring=docstring)
        self.shape = shape 
        num_empty = 0
        for s in self.shape:
            if s is None:
                num_empty+=1
        self.num_empty=num_empty
        self.input_stacks = {"int":self.num_empty}
        self.dtype = dtype 
        self.output_stack = output_stack
    
    def evaluate(self, state:PushState) -> PushState:
        args = state.observe(self.input_stacks)
        if args is None:
            return state 

        res_shape = []
        shape_args = args["int"]
        for s in self.shape:
            if s is None:
                res_shape.append(shape_args.pop(0))
            else:
                res_shape.append(s)
        shape = Shape(*res_shape)

        input = Parameter(state.nsteps, len(state.params), shape=shape, dtype=self.dtype)
        state.params.append({"shape": shape, "dtype": self.dtype})
        return state.push_to_stacks({self.output_stack: [input]})
    
    def required_stacks(self) -> Set[str]:
        return {self.output_stack}
    
class LiteralInstruction(Instruction):
    "Pushes a literal, such as a float or an integer"
    def __init__(self, name: str, value, output_stack: str, docstring: str = None):
        super().__init__(name, 0, docstring=docstring)
        self.value = value 
        self.output_stack = output_stack
    
    def evaluate(self, state:PushState, nsteps: int = 0) -> PushState:
        return state.push_to_stacks({self.output_stack: [self.value]})
    
    def required_stacks(self) -> Set[str]:
        return {self.output_stack}
    
class CodeBlockClose():
    "Closes a code block"
    def __eq__(self, other):
        return isinstance(other,CodeBlockClose) 
    def __str__(self):
        return "CodeBlockClose()"
    def __repr__(self):
        return str(self)
