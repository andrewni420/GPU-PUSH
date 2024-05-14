from abc import ABC, abstractmethod
import jax.numpy as jnp  

class Limiter(ABC):
    "Base class for limiting the outputs of functions"
    @abstractmethod
    def __call__(self, output):
        pass 

    def apply(self, fn, output):
        "Apply a function to all elements of an output."
        if isinstance(output, list):
            output = [fn(o) for o in output]
        elif isinstance(output, dict):
            output = {k:[fn(v_) for v_ in v] for k,v in output.items()}
        else:
            output = fn(output)
        return output

    def limit(self, fn):
        "Create a wrapper around the function to limit its output"
        def wrapper(*args, **kwargs):
            res = fn(*args, **kwargs)
            return self.__call__(res)
        return wrapper 
 
class LambdaLimiter(Limiter):
    "General function applied to each element"
    def __init__(self, fn=None):
        self.fn = fn 

    def __call__(self, output):
        return output if self.fn is None else self.apply(self.fn, output)

class SizeLimiter(Limiter):
    "Clip the output to a certain range"
    def __init__(self, low=None, high=None, magnitude=None):
        if magnitude is None:
            self.low = low or -float('inf')
            self.high = high or float('inf')
        else:
            self.low = min(-magnitude, low or float('inf'))
            self.high = max(magnitude, high or -float('inf'))

    def __call__(self, output):
        return self.apply(lambda x: jnp.maximum(jnp.minimum(x,self.high),self.low), output)

class GrowthLimiter(Limiter):
    "Limit the number of items pushed onto the push state"
    def __init__(self, limit=None):
        self.limit = limit or float('inf')

    def __call__(self, output):
        if isinstance(output,list) and len(output)>self.limit:
            return None 
        if isinstance(output, dict) and sum([len(i) for i in output.values()])>self.limit:
            return None 


DEFAULT_SIZE_LIMIT = 1E6
DEFAULT_SIZE_LIMITER = SizeLimiter(magnitude=DEFAULT_SIZE_LIMIT)