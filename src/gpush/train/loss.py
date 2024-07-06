from __future__ import annotations
from abc import ABC, abstractmethod
from ..utils import map_pytree, PyTree, pytree_to_list
from dataclasses import dataclass
from typing import Union, Optional, Any
from jax import Array
import jax.numpy as jnp
import jax.nn as nn 

@dataclass(frozen=True)
class BaseLoss(ABC):
    """A class to represent a single loss function"""
    
    def __call__(self, params: PyTree, output: Array, target: Array) -> Array:
        return self.eval(params, output, target)
    
    @abstractmethod
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        "Evaluate the loss function, given outputs and parameters of the neural network and target values"
        pass  

@dataclass(frozen=True)
class AggregatedLoss(BaseLoss):
    """A class to represent a multivalued loss function which is potentially aggregated into a single value"""
    reduction: str = "mean"
    def __post_init__(self):
        if self.reduction not in ["mean", "sum", "none", None]:
            raise ValueError(f"Unrecognized reduction: {self.reduction}")
    def reduce(self, res):
        if self.reduction=="mean":
            return jnp.mean(res)
        elif self.reduction=="sum":
            return jnp.sum(res)
        else:
            return res

@dataclass(frozen=True)
class L1Reg(BaseLoss):
    "L1 regularization"
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        return jnp.sum(jnp.array([jnp.abs(arr) for arr in pytree_to_list(params)])) 

@dataclass(frozen=True)
class L2Reg(BaseLoss):
    "L2 regularization"
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        return jnp.sqrt(jnp.sum(jnp.array([jnp.sum(arr**2) for arr in pytree_to_list(params)]))) 
    
@dataclass(frozen=True)
class MSELoss(AggregatedLoss):
    "Mean Squared Error Loss"
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        return self.reduce((output-target)**2)
    
@dataclass(frozen=True)
class MAELoss(AggregatedLoss):
    "Mean Absolute Error Loss"
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        return self.reduce(jnp.abs(output-target))

@dataclass(frozen=True)
class LpLoss(AggregatedLoss):
    "Lp loss for a given p, default 2"
    p: float = 2
    def __post_init__(self):
        if self.p==0:
            raise ValueError("LpLoss p cannot be 0")
        super().__post_init__()

    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        res = jnp.abs(output-target)**self.p
        res = self.reduce(res)
        return res**(1/self.p)

def L2Loss(reduction="mean"):
    return LpLoss(reduction=reduction,p=2)
def L1Loss(reduction="mean"):
    return LpLoss(reduction=reduction,p=1)


@dataclass(frozen=True)
class NLLLoss(AggregatedLoss):
    "Negative log likelihood loss"
    index: bool = True 
    "Whether the targets will be provided as (num_batches,) indices or (num_batches, num_classes) one-hot vectors"
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        if self.index:
            loss = jnp.take_along_axis(output,target[:,None],-1)
        else:
            loss = jnp.sum(target*output)
        return self.reduce(-loss)  

@dataclass(frozen=True)
class CrossEntropyLoss(NLLLoss):
    "Cross entropy loss"
    index: bool = True 
    "Whether the targets will be provided as (num_batches,) indices or (num_batches, num_classes) vectors"
    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        output = nn.log_softmax(output,axis=1)
        if self.index:
            return super().eval(params, output, target)
        else:
            return self.reduce(-(jnp.sum(target*output,axis=1)))

@dataclass(frozen=True)
class CompositeLoss(BaseLoss):
    """A class to represent a general loss function, constructed as a weighted sum of individual loss functions"""
    coefficients: dict[BaseLoss,float]

    @classmethod 
    def make_coefs(coefficients: Union[BaseLoss,list[BaseLoss],dict[BaseLoss,float]]) -> dict[BaseLoss,float]:
        "Takes a single loss function, a list of loss functions, or a dictionary, and turns it into a dictionary suitable for use in a `Loss` object"
        if isinstance(coefficients,dict):
            return coefficients 
        elif isinstance(coefficients,list):
            return {k:1 for k in coefficients}
        else:
            return {coefficients:1}

    def eval(self, params: PyTree, output: Array, target: Array) -> Array:
        "Returns a weighted sum of various loss functions"
        return jnp.sum(jnp.array([loss(params, output, target)*coef for loss,coef in self.coefficients.items()]))



