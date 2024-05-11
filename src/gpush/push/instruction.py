from __future__ import annotations
from abc import ABC, abstractmethod 
from typing import Callable, Set 
from .state import PushState
from .dag.expr import Function

class Instruction(ABC):
    def __init__(self, name: str, code_blocks: int, docstring = None):
        self.name = name
        self.code_blocks = code_blocks
        self.docstring = docstring 

    def __eq__(self, other: Instruction) -> bool:
        return self.name==other.name 
    
    @abstractmethod 
    def evaluate(self, state: PushState) -> PushState:
        pass 

    @abstractmethod
    def required_stacks(self) -> Set[str]:
        pass 


class StateToStateInstruction(Instruction):
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
    """A simple instruction"""
    def __init__(self, name: str, fn: Callable, input_stacks: tuple[str], output_stacks: tuple[str], code_blocks: int, docstring=None, validator: Callable = None):
        super().__init__(name, code_blocks, docstring=docstring)
        self.fn = fn 
        self.input_stacks = input_stacks
        self.output_stacks = output_stacks
        self.validator = validator
    
    def evaluate(self, state: PushState) -> PushState:
        # Get arguments and check if there's enough present
        args = state.observe(self.input_stacks)
        if args is None:
            return state 
        if self.validator is not None:
            if not self.validator(*args):
                return None 
        result = self.fn(*args)
        result = {k:[v] for k,v in zip(self.output_stacks, result)}
        return state.push_to_stacks(result)
    
    def required_stacks(self) -> Set[str]:
        return set(self.input_stacks)+set(self.output_stacks)
    
class SimpleExpressionInstruction(Instruction):
    """A simple instruction that constructs an Expression"""
    def __init__(self, name: str, fn: Callable, signature: Callable, input_stacks: tuple[str], output_stack: str, code_blocks: int, docstring=None, validator: Callable = None):
        super().__init__(name, code_blocks, docstring=docstring)
        self.fn = fn 
        self.input_stacks = input_stacks
        self.output_stack = output_stack
        self.validator = validator
        self.signature = signature

    def evaluate(self, state: PushState) -> PushState:
        # Get arguments and check if there's enough present
        args = state.observe(self.input_stacks)
        if args is None:
            return state 
        if self.validator is not None:
            if not self.validator(*args):
                return None 
        signature = self.signature(*args)
        result = {self.output_stack: Function(self.fn, children=tuple(args), shape=signature["shape"], dtype=signature["dtype"])}
        return state.push_to_stacks(result)
    
    def required_stacks(self) -> Set[str]:
        return set(self.input_stacks)+{self.output_stack}

class MultiArgumentInstruction(Instruction):
    """An instruction that takes multiple arguments from one or more stacks and constructs an Expression"""
    def __init__(self, name: str, fn: Callable, signature: Callable, input_stacks: dict[str,int], output_stack: str, code_blocks: int, docstring=None, validator: Callable = None):
        super().__init__(name, code_blocks, docstring=docstring)
        self.fn = fn 
        self.input_stacks = input_stacks
        self.output_stack = output_stack
        self.validator = validator
        self.signature = signature

    def evaluate(self, state: PushState) -> PushState:
        # Get arguments and check if there's enough present
        args = state.observe(self.input_stacks)
        if args is None:
            return state 
        if self.validator is not None:
            if not self.validator(**args):
                return None 
        signature = self.signature(**args)
        result = {self.output_stack: Function(self.fn, children=tuple(args), shape=signature["shape"], dtype=signature["dtype"])}
        return state.push_to_stacks(result)
    
    def required_stacks(self) -> Set[str]:
        return set(self.input_stacks)+{self.output_stack}