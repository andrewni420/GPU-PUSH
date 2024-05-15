from __future__ import annotations
from abc import ABC,abstractmethod
from jax import jit, Array
import jax.numpy as jnp 
from typing import List, Any, Callable, Union 
from dataclasses import dataclass
from functools import wraps 
import numpy as np 

@dataclass(frozen=True)
class Optimizer(ABC):
    @abstractmethod
    def _init(self, params: List[Array]) -> Any:
        pass 

    @abstractmethod
    def _update(self, i: int, grad: List[Array], state: Any) -> Any:
        pass 

    @abstractmethod
    def _get_params(self, state: Any) -> List[Array]:
        pass 

    @property 
    def init(self) -> Callable[[List[Array]], Any]:
        @wraps(self.init)
        @jit
        def wrap_init(params):
            return self.init(params)
        return wrap_init
    
    @property 
    def update(self) -> Callable[[List[Array],List[Array], Any], Any]:
        @wraps(self.update)
        @jit
        def wrap_update(params,grad,state):
            return self.update(params,grad,state)
        return wrap_update
    
    @property 
    def get_params(self) -> Callable[[Any], List[Array]]:
        @wraps(self.get_params)
        @jit
        def wrap_params(state):
            return self.get_params(state)
        return wrap_params
    
    def make_schedule(self, value: float):
        if isinstance(value,Scheduler):
            return value 
        else:
            return Scheduler(value)

@dataclass(frozen=True)
class SGD(Optimizer):
    init_lr: Union[float,Scheduler] = 1E-4
    def _init(self, params: List[Array]) -> dict[str,Union[List[Array],float, Scheduler]]:
        return {"params": params, "lr":self.make_schedule(self.init_lr)}
    def _update(self, i: int, grad: List[Array], state: dict[str,Union[List[Array],float,Scheduler]]) -> dict[str,Union[List[Array],float, Scheduler]]:
        x = state["params"]
        lr = state["lr"]
        return {"params": [x_-g*lr for x_,g in zip(x,grad)], "lr": lr.step()}
    def _get_params(self, state: dict[str,Union[List[Array],float, Scheduler]]) -> List[Array]:
        return state["params"]

@dataclass(frozen=True)
class Momentum(SGD):
    init_m: Union[float,Scheduler] = 0.9
    def _init(self, params: List[Array]) -> dict[str,Union[List[Array],float, Scheduler]]:
        return {"params": params, "momentum": [jnp.zeros_like(p) for p in params], "m": self.make_schedule(self.init_m), "lr": self.make_schedule(self.init_lr)}
    def _update(self, i: int, grad: List[Array], state: dict[str,Union[List[Array],float, Scheduler]]) -> dict[str,Union[List[Array],float, Scheduler]]:
        m,lr,x,velocity = [state[k] for k in ["m", "lr", "x", "params"]]
        velocity = [v*m+g for v,g in zip(velocity,grad)]
        x = [x_-self.lr*v for x_,v in zip(x,m)]
        return {"params": x, "momentum": velocity, "m": m.step(), "lr": lr.step()}

@dataclass(frozen=True)
class Adam(SGD):
    init_b1: Union[float,Scheduler] = 0.9
    init_b2: Union[float,Scheduler] = 0.99 
    eps: float = 1E-8
    def _init(self, params: List[Array]) -> dict[str,Union[List[Array],float, Scheduler]]:
        return {"params": params, 
                "lr": self.make_schedule(self.init_lr),
                "b1": self.make_schedule(self.init_b1), 
                "b2":self.make_schedule(self.init_b2),
                "eps":self.eps,
                "velocity": [jnp.zeros_like(p) for p in params],
                "var": [jnp.zeros_like(p) for p in params]}
    def _update(self, i: int, grad: List[Array], state: dict[str,Union[List[Array],float, Scheduler]]) -> dict[str,Union[List[Array],float, Scheduler]]:
        params,b1,b2,lr,eps,velocity,var = [state[k] for k in ["params","b1","b2","lr","eps","velocity","var"]]
        var = [((1-b1)*g+b1*v)/(1-jnp.asarray(b1)**(i+1)) for g,v in zip(grad,var)]
        velocity = [((1-b2)*jnp.square(g))/(1-jnp.asarray(b2)**(i+1)) + b2*v for g,v in zip(grad,velocity)]
        x = [x-lr*v/(jnp.sqrt(v_)+eps) for x,v,v_ in zip(params,velocity,var)]
        return {"params":x, "b1":b1.step(),"b2":b2.step(),"lr":lr.step(),"eps":eps,"velocity":velocity,"var":var}


class Scheduler(float, ABC):
    def __init__(self, val: float, step: int = 0):
        self.i = step 
    def __new__(cls, val: float, step: int = 0):
        super().__new__(cls, val)

    @abstractmethod
    def step(self) -> Scheduler:
        return self 

class LambdaScheduler(Scheduler):
    def __init__(self, val: float, init_step: int = 0, fn: Callable[[int,float],float] = None):
        self.i = init_step 
        self.fn = fn 
    def __new__(cls, val: float, init_step: int = 0, fn: Callable[[int,float],float] = None):
        super().__new__(cls, val) 

    def step(self):
        newval = self.fn(self.i,self)
        return LambdaScheduler(newval,step=self.i+1,fn=self.fn)
    
class CosineScheduler(Scheduler):
    def __init__(self, val: float, step: int = 0, max_steps: int = 0):
        super().__init__(val, step=step)
        self.max_steps = max_steps 
    def __new__(cls, val: float, step: int = 0, max_steps: int = 0):
        super().__new__(cls, val)
    
    def step(self):
        newval = jnp.cos(self.i*np.pi/(self.max_steps*2))
        return CosineScheduler(newval, step=self.i+1, max_steps = self.max_steps)
    
class StepwiseLinear(Scheduler):
    def __init__(self, val: float, step: int = 0, step_vals: list[tuple[int,float]] = None):
        super().__init__(val, step=step)
        self.step_vals = step_vals 
    def __new__(cls, val: float, step: int = 0, max_steps: int = 0):
        super().__new__(cls, val)
    
    def step(self):
        prev_step, prev_val, cur_step, cur_val = [None]*4 
        for i,val in self.step_vals:
            cur_step = i 
            cur_val = val 
            if cur_step>self.i:
                break 
            prev_step = cur_step 
            prev_val = cur_val 
        if prev_val is None:
            return StepwiseLinear(cur_val,step=self.i+1, step_vals = self.step_vals)
        else:
            val = prev_val + (cur_val-prev_val)*(self.i-prev_step)/(cur_step-prev_step)
            return StepwiseLinear(val, step=self.i+1, step_vals=self.step_vals)
    
