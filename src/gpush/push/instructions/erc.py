from abc import ABC, abstractmethod
from ..instruction import LiteralInstruction, ParamBuilderInstruction
from ..dag.shape import Shape 
import numpy as np 
from typing import Callable


class ERCGenerator(ABC):
    def __init__(self, name: str = None):
        self.name = name 

    @abstractmethod
    def sample(self):
        pass 

    def __str__(self):
        return self.name 
    def __repr__(self):
        return self.name 


class ERCFloatRange(ERCGenerator):
    "Generates an ERC float from a predefined uniform range, with an optional inverse integral transform"
    def __init__(self, name: str = None, low: float = None, high: float = None, magnitude: float = None, fn: Callable = None):
        if magnitude is None:
            self.low = low or -float('inf')
            self.high = high or float('inf')
        else:
            self.low = min(-magnitude, low or float('inf'))
            self.high = max(magnitude, high or -float('inf'))

        self.rng = np.random.default_rng()
        self.fn = fn 
        super().__init__(name=name or f"ERCFloat[{self.low},{self.high}]")

    def sample(self):
        val = self.rng.uniform(low=self.low, high=self.high)
        val = val if self.fn is None else self.fn(val)
        return LiteralInstruction(f"float({val})", val, "float")
    

class ERCIntRange(ERCGenerator):
    "Generates an ERC integer in a predefined uniform range, with an optional inverse integral transform"
    def __init__(self, name: str = None, low: int = None, high: int = None, magnitude: int = None, fn: Callable = None):
        for var in [low,high,magnitude]:
            if var is not None and not isinstance(var,int):
                raise ValueError(f"Int range specification of wrong dtype: {var}")
        if magnitude is None:
            self.low = low or -float('inf')
            self.high = high or float('inf')
        else:
            self.low = min(-magnitude, low or int('inf'))
            self.high = max(magnitude, high or -float('inf'))

        self.rng = np.random.default_rng()
        self.fn = fn 
        super().__init__(name=name or f"ERCInt[{self.low},{self.high}]")
    
    def sample(self):
        val = self.rng.integers(self.low, self.high)
        val = val if self.fn is None else self.fn(val)
        return LiteralInstruction(f"int({val})", val, "int")
    
class LambdaERC(ERCGenerator):
    "Generates an ERC from a lambda function"
    def __init__(self, fn: Callable, *args, output_stack: str = None, name: str = None, **kwargs):
        self.fn=fn
        self.output_stack=output_stack
        self.args = args 
        self.kwargs = kwargs 
        self.name = name or f"LambdaERC({fn})"
    def sample(self):
        val = self.fn(*self.args, **self.kwargs)
        return LiteralInstruction(f"{self.output_stack}({val})", val, self.output_stack)
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    @staticmethod
    def make_erc(*args, name: str=None, output_stack: str=None, **kwargs):
        def wrapper(fn):
            return LambdaERC(fn, *args, output_stack=output_stack, name=name, **kwargs)
        return wrapper
    

class ChoiceERC(ERCGenerator):
    def __init__(self, vals: list, p: list = None, output_stack: str = None):
        self.vals = vals 
        self.p = p 
        self.output_stack = output_stack
        self.rng = np.random.default_rng()

    def sample(self):
        val = self.rng.choice(self.vals,p=self.p)
        return LiteralInstruction(f"ChoiceERC({val})",val,self.output_stack)
    
class ParamBuilderERC(ERCGenerator):
    def __init__(self, ndims = None, dim_sizes = None, unset_dims=None, dtype=None):
        self.ndims = self.process_input(ndims)
        self.dim_sizes = self.process_input(dim_sizes)
        self.unset_dims = self.process_input(unset_dims)
        self.dim_sizes = self.process_input(dim_sizes)
        self.dtype = self.process_input(dtype,type=str)
        self.rng = np.random.default_rng()

    def process_input(self, input,type=int):
        if isinstance(input,type):
            return ([input],[1])
        elif isinstance(input,list):
            if isinstance(input[0],list):
                return input 
            else:
                return ([input],[1/len(input)]*len(input))
        else:
            raise ValueError("Invalid input to ParamBuilderERC")
        
    def choose(self,attr,n=None):
        return self.rng.choice(attr[0],p=attr[1],size=n)
    
    def sample(self):
        ndims = self.choose(self.ndims)
        unset_dims = self.choose(self.unset_dims)
        unset_dims = max(ndims,unset_dims)
        preset_dims = ndims-unset_dims
        preset_sizes = self.choose(self.dim_sizes,n=preset_dims)
        sizes = preset_sizes+[None]*unset_dims
        self.rng.shuffle(sizes)
        dtype = self.choose(self.dtype)
        return ParamBuilderInstruction(f"ParamBuilder({sizes},{dtype})",Shape(sizes),dtype,f"{dtype}_jax_expr")

    
    
