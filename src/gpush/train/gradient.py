from __future__ import annotations
from abc import ABC, abstractmethod
from ..utils import map_pytree
from dataclasses import dataclass
from ..utils import PyTree, map_pytree, pytree_to_list
import jax.numpy as jnp
from typing import Union 
from jax import Array

@dataclass(frozen=True)
class BaseTransformation(ABC):
    "A class to apply some transformation to a gradient pytree"
    def __call__(self, grad: PyTree) -> PyTree:
        "Sugar for `self.apply()`"
        return self.apply(grad)

    @abstractmethod
    def apply(self, grad: PyTree) -> PyTree:
        "Apply some transformation to the gradients, returning the modified gradients"
        pass 


@dataclass(frozen=True)
class ClipGradNorm(BaseTransformation):
    max_norm: float 
    "Maximum value of the norm"
    norm_type: Union[str,float] 
    "Type of the used p-norm. Can be 'inf' or '-inf' for the infinity norms"
    def compute_norm(self, grad: PyTree) -> Array:
        flat_grad = jnp.concatenate(pytree_to_list(grad), axis=None)
        if self.norm_type=="inf":
            return jnp.max(jnp.abs(flat_grad))
        elif self.norm_type=="-inf":
            return jnp.min(jnp.abs(flat_grad))
        else:
            return jnp.sum(jnp.abs(flat_grad)**self.norm_type)**(1/self.norm_type)
    
    def apply(self, grad: PyTree) -> PyTree:
        "Scales the gradients to have a norm value of at most `max_norm`"
        norm_val = self.compute_norm(grad)
        scale_val = jnp.where(norm_val<self.max_norm,1,jnp.where(norm_val==0,1,self.max_norm/norm_val))
        return map_pytree(lambda x:x/scale_val,grad)


@dataclass(frozen=True)
class ClipGradValue(BaseTransformation):
    max_value: float 
    "Maximum absolute value of the gradient"
    def apply(self, grad: PyTree) -> PyTree:
        "Clips the gradient to have an absolute value of at most `max_value`"
        return map_pytree(lambda x:jnp.maximum(-self.max_value,jnp.minimum(self.max_value,x)),grad)
    
@dataclass(frozen=True)
class CompositeTransformation(ABC):
    "A class to apply multiple gradient transformations in order"
    pipeline: list[BaseTransformation]
    "The gradient transformations to use, in order"

    def apply(self,grad: PyTree) -> PyTree:
        "Applies the transformations in `pipeline` in order, returning the resulting gradient"
        for transformation in self.pipeline:
            grad = transformation(grad)
        return grad 
