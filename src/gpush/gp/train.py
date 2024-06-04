from __future__ import annotations
from abc import ABC,abstractmethod
from jax import jit, Array
import jax.numpy as jnp 
from typing import List, Any, Callable, Union 
from dataclasses import dataclass, field
from functools import wraps, update_wrapper
import numpy as np 
from frozendict import frozendict 
import jax.lax as lax 


@dataclass(frozen=True)
class Schedule(ABC):
    @abstractmethod
    def step(self, i: int, cur_val: float) -> float:
        return cur_val
    
    def initial_value(self) -> float:
        return self.step(0,None) 
    
@dataclass(frozen=True)
class ConstantSchedule(Schedule):
    init_val: float

    def step(self, i: int, cur_val: float) -> float:
        return self.init_val
    def initial_value(self) -> float:
        return self.init_val

@dataclass(frozen=True)
class LambdaSchedule(Schedule):
    fn: Callable
    args: tuple = tuple()
    kwargs: frozendict = frozendict()

    def step(self, i: int, cur_val: float) -> float:
        return self.fn(i, cur_val, *self.args, **self.kwargs)
    
@dataclass(frozen=True)
class CosineSchedule(Schedule):
    init_val: int 
    max_steps: int 
    final_val: int = 0
    def __post_init__(self):
        if self.max_steps<=0:
            raise ValueError("Cosine Scheduler max steps must be positive")

    def step(self, i: int, cur_val: float) -> float:
        return self.final_val + (self.init_val-self.final_val)*jnp.cos(i*jnp.pi/(self.max_steps*2))
    
@dataclass(frozen=True)
class StepwiseLinearSchedule(Schedule):
    step_vals: tuple[tuple[int,float]] 
    def safe_step(self, i: int, val: float) -> float:
        prev_step, prev_val, cur_step, cur_val = [None]*4 
        for idx,val in self.step_vals:
            cur_step = idx
            cur_val = val 
            if cur_step>i:
                break 
            prev_step = cur_step 
            prev_val = cur_val 
        denom = cur_step-prev_step 
        protected_denom = jnp.where(denom==0,1,denom)
        return prev_val + (cur_val-prev_val)*(i-prev_step)/protected_denom

    def step(self, i: int, val: float) -> float:
        res = self.safe_step(i,val)
        res = jnp.where(i>=self.step_vals[-1][0], jnp.array(self.step_vals[-1][1]), res)
        res = jnp.where(i<self.step_vals[0][0],jnp.array(self.step_vals[0][1]),res)
        return res 
        
    
State = dict[str,Union[List[Array],float, Schedule]]
Params = List[Array]

class Optimizer(ABC):
    @abstractmethod
    def init(self, params: List[Array]) -> Any:
        pass 

    @abstractmethod
    def update(self, i: int, grad: List[Array], state: Any) -> Any:
        pass 

    @abstractmethod
    def get_params(self, state: Any) -> List[Array]:
        pass 
    
    def make_schedule(self, value: float):
        if isinstance(value,Schedule):
            return value 
        else:
            return ConstantSchedule(value)

@dataclass(frozen=True)
class SGD(Optimizer):
    lr: Schedule = ConstantSchedule(1E-4)
    def init(self, params: Params) -> State:
        return {"params": params, "lr":self.lr.initial_value()}
    def update(self, i: int, grad: Params, state: State) -> State:
        x = state["params"]
        lr = self.lr.step(i,state["lr"])
        return {"params": [x_-g*lr for x_,g in zip(x,grad)], "lr": lr}
    def get_params(self, state: State) -> Params:
        return state["params"]

@dataclass(frozen=True)
class Momentum(SGD):
    mass: Schedule = ConstantSchedule(0.9)
    def init(self, params: Params) -> State:
        return {"params": params, "momentum": [jnp.zeros_like(p) for p in params], "mass": self.mass.initial_value(), "lr": self.lr.initial_value()}
    def update(self, i: int, grad: Params, state: State) -> State:
        m,lr,x,velocity = [state[k] for k in ["mass", "lr", "params", "momentum"]]
        lr = self.lr.step(i,lr)
        m = self.mass.step(i,m)
        velocity = [v*m+g for v,g in zip(velocity,grad)]
        x = [x_-lr*v for x_,v in zip(x,velocity)]
        return {"params": x, "momentum": velocity, "mass": m, "lr": lr}

@dataclass(frozen=True)
class Adam(SGD):
    b1: Schedule = ConstantSchedule(0.9)
    b2: Schedule = ConstantSchedule(0.99)
    eps: Schedule = ConstantSchedule(1E-8)
    def init(self, params: Params) -> State:
        return {"params": params, 
                "lr": self.lr.initial_value(),
                "b1": self.b1.initial_value(), 
                "b2":self.b2.initial_value(),
                "eps":self.eps.initial_value(),
                "velocity": [jnp.zeros_like(p) for p in params],
                "var": [jnp.zeros_like(p) for p in params]}
    def update(self, i: int, grad: Params, state: State) -> State:
        params,b1,b2,lr,eps,velocity,var = [state[k] for k in ["params","b1","b2","lr","eps","velocity","var"]]
        b1 = self.b1.step(i,b1)
        b2 = self.b2.step(i,b2)
        lr = self.lr.step(i,lr)
        eps = self.eps.step(i,eps)
        velocity = [((1-b1)*g+b1*v) for g,v in zip(grad,velocity)]
        mhat = [m/(1-jnp.asarray(b1)**(i+1)) for m in velocity]
        var = [((1-b2)*jnp.square(g)+ b2*v) for g,v in zip(grad,var)]
        vhat = [v/(1-jnp.asarray(b2)**(i+1)) for v in var]
        x = [x-lr*v/(jnp.sqrt(v_)+eps) for x,v,v_ in zip(params,mhat,vhat)]
        return {"params":x, "b1":b1,"b2":b2,"lr":lr,"eps":eps,"velocity":velocity,"var":var}

    
