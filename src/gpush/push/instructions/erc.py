from abc import ABC, abstractmethod
from ..instruction import LiteralInstruction
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
    
