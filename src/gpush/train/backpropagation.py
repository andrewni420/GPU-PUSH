from __future__ import annotations 
from abc import ABC, abstractmethod
from ..utils import map_pytree, PyTree
from dataclasses import dataclass, field, KW_ONLY
from .gradient import BaseTransformation
from .loss import BaseLoss
from typing import Union, Optional, Callable 
from functools import cached_property, wraps
from jax import jit, value_and_grad,vmap, Array
import jax.lax as lax 
import jax.numpy as jnp 
import numpy as np

FnOutput = tuple[Array,PyTree[Array]]
GradOutput = tuple[FnOutput,Array]
Params = PyTree[Array]
Input = PyTree[Array]
Target = Array

@dataclass(frozen=True)
class BackPropagation(ABC):
    "A class to implement simple backpropagation"
    
    fn: Callable[[Params,Input],FnOutput]
    "The function to perform backpropagation on. Only applies to a single time step for RNNs. Use `BackPropagation.val` for a scanned version when using RNNs."
    loss: BaseLoss
    "The loss function"
    transform: Optional[BaseTransformation] = field(kw_only=True,default=None)
    "Optional transforms to apply to the gradient"

    def __call__(self,params: Params, input: Input, target: Target) -> GradOutput:
        "Returns ((value, auxiliary), grad)"
        return self.grad(params, input, target) 
    
    def transform_gradient(self, gradient: Callable):
        "Transform the gradient using `self.transform`"
        @wraps(gradient)
        def wrapper(*args, **kwargs):
            ((val,aux),ret) = gradient(*args,**kwargs)
            ret = ret if self.transform is None else self.transform(ret)
            return ((val,aux),ret)
        return wrapper 
    
    @cached_property
    def val(self) -> Callable[[Params,Input],FnOutput]:
        return self.fn

    @cached_property
    def grad(self) -> Callable[[Params,Input,Target],GradOutput]:
        "Returns a jitted function that, when called on the params, input, and hidden state, returns a tuple ((value, auxiliary), grad)"

        # Tack on the loss function
        def fn_wrapper(params,input,target):
            res,aux = self.fn(params,input)
            return self.loss(params, res, target),aux
        
        # Take the gradient
        gradient = value_and_grad(fn_wrapper, has_aux=True)
        # Transform
        gradient = self.transform_gradient(gradient)
        # jit
        return jit(gradient)
    
@dataclass(frozen=True)
class BPTT(BackPropagation):
    hidden_params: np.ndarray
    "A list of the indices of the parameters corresponding to hidden states, which are updated at each evaluation"
    reg_loss: BaseLoss = None
    "Auxiliary regularization loss function"
    time_first: bool = False 
    "Is the first input dimension batch or time?"
    
    def stop_grad(self,params):
        return [lax.cond(self.hidden_params[i]==1,lax.stop_gradient(e),e) for i,e in enumerate(params)]
        
    @cached_property
    def val(self) -> Callable[[Params,Input],FnOutput]:
        "A function that computes the value of the function on given parameters and inputs."
        def fn_wrapper(params: Params, input: Input) -> FnOutput:
            "A wrapper to scan the function through the inputs and compute the loss"
            # Setup
            batch_size = input[0].shape[1] if self.time_first else input[0].shape[0]
            params = map_pytree(lambda x:lax.broadcast(x,(batch_size,)),params)
            input = input if self.time_first else map_pytree(lambda x:jnp.swapaxes(x,0,1),input)

            def scan_fn(carry,input):
                "Wrapper to transform the input function into a scannable function"
                val,aux = self.fn(carry,input)
                return aux,val 
            def scan_all(carry,input):
                carry, output = lax.scan(scan_fn,carry,xs=input)
                output = output if self.time_first else jnp.swapaxes(output,0,1)
                return output,carry
            return scan_all(params,input)
        return fn_wrapper
    
    @cached_property
    def grad(self) -> Callable[[Params,Input,Target],GradOutput]:
        """Returns a jitted function that, when called on the params, input, and target, returns a tuple ((value, auxiliary), grad)"""
        
        def fn_wrapper(params, input,target):
            "A wrapper to scan the function through the inputs and compute the loss"
            # Setup
            batch_size = input[0].shape[1] if self.time_first else input[0].shape[0]
            params = map_pytree(lambda x:lax.broadcast(x,(batch_size,)),params)
            input = input if self.time_first else map_pytree(lambda x:jnp.swapaxes(x,0,1),input)
            target = target if self.time_first else jnp.swapaxes(target,0,1)

            def scan_fn(carry,input):
                "Wrapper to transform the input function into a scannable function"
                val,aux = self.fn(carry,input)
                return aux,val 
            def scan_all(carry,input):
                "Scans a single iteration of truncated bptt, composed of k1 timesteps, using `scan_fn`"
                # Scan through without backprop
                carry, output = lax.scan(scan_fn,carry,xs=input)
                loss = self.loss(carry, output,target)
                return loss,carry
            aux_loss = 0 if self.reg_loss is None else self.reg_loss(params,None,None)
            loss, carry = scan_all(params,input)
            return loss+aux_loss, carry
        
        # Take the gradient
        gradient = value_and_grad(fn_wrapper, has_aux=True)

        # Transform the gradient
        gradient = self.transform_gradient(gradient)

        return jit(gradient)  


@dataclass(frozen=True)
class TBPTT(BackPropagation):
    k1: int 
    "How many timesteps to run for"
    k2: int 
    "How many timesteps to apply backpropagation through time for"
    hidden_params: np.ndarray
    "A list of the indices of the parameters corresponding to hidden states, which are updated at each evaluation"
    reg_loss: BaseLoss = None
    "Auxiliary regularization loss function"
    time_first: bool = False 
    "Is the first input dimension batch or time?"
    
    def __post_init__(self):
        if self.k2<self.k1:
            raise ValueError("TBPTT k2 cannot be smaller than k1")
        
    def stop_grad(self,params):
        return [lax.cond(self.hidden_params[i]==1,lax.stop_gradient(e),e) for i,e in enumerate(params)]
        
    @cached_property
    def val(self) -> Callable[[Params,Input],FnOutput]:
        "A function that computes the value of the function on given parameters and inputs."
        def fn_wrapper(params: Params, input: Input) -> FnOutput:
            "A wrapper to scan the function through the inputs and compute the loss"
            # Setup
            batch_size = input[0].shape[1] if self.time_first else input[0].shape[0]
            params = map_pytree(lambda x:lax.broadcast(x,(batch_size,)),params)
            input = input if self.time_first else map_pytree(lambda x:jnp.swapaxes(x,0,1),input)

            def scan_fn(carry,input):
                "Wrapper to transform the input function into a scannable function"
                val,aux = self.fn(carry,input)
                return aux,val 
            def scan_all(carry,input):
                carry, output = lax.scan(scan_fn,carry,xs=input)
                output = output if self.time_first else jnp.swapaxes(output,0,1)
                return output,carry
            return scan_all(params,input)
        return fn_wrapper
    
    @cached_property
    def grad(self) -> Callable[[Params,Input,Target],GradOutput]:
        """Returns a jitted function that, when called on the params, input, and target, returns a tuple ((value, auxiliary), grad)"""

        k_initial = self.k1-self.k2 
        
        def fn_wrapper(params, input,target):
            "A wrapper to scan the function through the inputs and compute the loss"
            # Setup
            batch_size = input[0].shape[1] if self.time_first else input[0].shape[0]
            params = map_pytree(lambda x:lax.broadcast(x,(batch_size,)),params)
            input = input if self.time_first else map_pytree(lambda x:jnp.swapaxes(x,0,1),input)
            target = target if self.time_first else jnp.swapaxes(target,0,1)
            input = map_pytree(lambda x:jnp.reshape(x,shape=(-1,self.k1)+x.shape[1:]),input)
            target = jnp.reshape(target,shape=(-1,self.k1)+target.shape[1:])

            def scan_fn(carry,input):
                "Wrapper to transform the input function into a scannable function"
                val,aux = self.fn(carry,input)
                return aux,val 
            def scan_k1(carry,input):
                "Scans a single iteration of truncated bptt, composed of k1 timesteps, using `scan_fn`"
                # k1-k2 timesteps without backprop, k2 timesteps with backprop
                input_initial = map_pytree(lambda x:x[:k_initial],input)
                input_backprop = map_pytree(lambda x:x[k_initial:self.k1],input)

                # Scan through without backprop
                carry, _ = lax.scan(scan_fn,carry,xs=input_initial)
                carry = self.stop_grad(carry)
                # Scan with backprop
                carry, output = lax.scan(scan_fn,carry,xs=input_backprop)
                return carry,output
            def scan_all(carry,input):
                "Scans through the entire input array, using `scan_k1` at each iteration. Returns (loss,carry)"
                carry, output = lax.scan(scan_k1,carry,xs=input)
                # Do we want to reshape the targets or the output?
                # output = jnp.reshape(output,shape=(-1,)+output.shape[2:])
                # output = jnp.swapaxes(output,0,1) if self.time_first else output
                loss = self.loss(carry, output,target)
                return loss,carry 
            aux_loss = 0 if self.reg_loss is None else self.reg_loss(params,None,None)
            loss, carry = scan_all(params,input)
            return loss+aux_loss, carry
        
        # Take the gradient
        gradient = value_and_grad(fn_wrapper, has_aux=True)

        # Transform the gradient
        gradient = self.transform_gradient(gradient)

        return jit(gradient)  

