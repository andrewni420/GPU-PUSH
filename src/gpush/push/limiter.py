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
            return fn(output)
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
    "Limit the total number of items pushed onto the push state"
    maxlen: int 
    "Maximum number of items pushed onto the state"

    def __init__(self, maxlen: int = None):
        self.maxlen = maxlen or float('inf')

    def __call__(self, output):
        if isinstance(output,list) and len(output)>self.maxlen:
            return output[:self.maxlen] 
        if isinstance(output, dict):
            total = sum([len(i) for i in output.values()])
            while total>self.maxlen:
                sorted_keys = sorted(output.keys(),key=lambda x:len(output[x]), reverse=True)
                maxlen = len(output[sorted_keys[0]])
                second_idx=1
                while second_idx<len(sorted_keys) and len(output[sorted_keys[second_idx]])==maxlen:
                    second_idx+=1
                second_len = 0 if second_idx==len(sorted_keys) else len(output[sorted_keys[second_idx]])
                cur_keys = [k for k in sorted_keys if len(output[k])==maxlen]
                to_subtract = min(maxlen-second_len, int(jnp.ceil((total-self.maxlen)/len(cur_keys))))
                for k in cur_keys:
                    output[k] = output[k][:-to_subtract]
                total-=to_subtract*len(cur_keys)
            return output 
        
class StackLimiter(Limiter):
    "Limit the number of items pushed onto each stack"
    maxlen: int 
    "Maximum number of items pushed onto each stack"

    def __init__(self, maxlen: int = None):
        self.maxlen = float('inf') if maxlen is None else maxlen
        if self.maxlen<0:
            raise ValueError("maxlen must be nonnegative")

    def __call__(self, output):
        if isinstance(output,list) or isinstance(output, tuple):
            return output if self.maxlen>0 else [] 
        if isinstance(output, dict):
            return {k:v[:self.maxlen] for k,v in output.items()}

DEFAULT_SIZE_LIMIT = 1E6
DEFAULT_SIZE_LIMITER = SizeLimiter(magnitude=DEFAULT_SIZE_LIMIT)
DEFAULT_GROWTH_LIMIT = 100
DEFAULT_GROWTH_LIMITER = GrowthLimiter(maxlen=DEFAULT_GROWTH_LIMIT)
DEFAULT_STACK_LIMIT = 100
DEFAULT_STACK_LIMITER = StackLimiter(maxlen=DEFAULT_STACK_LIMIT)