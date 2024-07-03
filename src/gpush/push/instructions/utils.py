from ..instruction import SimpleInstruction, SimpleExprInstruction
from ..limiter import Limiter 
from typing import Callable, Union 
from abc import ABC, abstractmethod
from copy import copy 
from warnings import warn
from functools import update_wrapper

Stacks = Union[list[str], dict[str,int], str]

class InstructionWrapper(Callable):
    "A wrapper around an instruction that stores additional values"
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn 
        self.args = args 
        self.kwargs = kwargs 

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    def apply(self, fn):
        return fn(self.fn, *self.args, **self.kwargs)
    
class WrapperCreator():
    "Calling this object gives a decorator that wraps a function into an `InstructionWrapper` object."
    def __init__(self, 
                 name_creator: Callable[[str],str] = None, 
                 base_name: str = None, 
                 input_stacks: Stacks= None, 
                 output_stacks: Stacks = None, 
                 code_blocks: int = 0, 
                 docstring: str = None, 
                 validator: Callable = None, 
                 limiter: Limiter = None, 
                 signature: Callable = None,
                 non_expr_stacks: set[str] = None,
                 **kwargs):
        kwargs.update({"name_creator":name_creator, 
                       "base_name":base_name, 
                       "input_stacks":input_stacks, 
                       "output_stacks":output_stacks, 
                       "code_blocks":code_blocks, 
                       "docstring":docstring, 
                       "validator":validator, 
                       "limiter":limiter, 
                       "signature":signature, 
                       "non_expr_stacks":non_expr_stacks})
        self.default = kwargs 
    def __call__(self, **kwargs):
        default = copy(self.default)
        default.update(kwargs)
        def wrap(fn):
            return update_wrapper(InstructionWrapper(fn, **default),fn)
        return wrap
    
def transform_stacks(fn, stacks, non_expr_stacks=set()):
    "Applies `fn` to all of the stack names in `stacks`"
    if isinstance(stacks,tuple):
        return tuple((s if s in non_expr_stacks else fn(s)) for s in stacks)
    elif isinstance(stacks, dict):
        return {(k if k in non_expr_stacks else fn(k)):v for k,v in stacks.items()}
    else:
        return stacks if stacks in non_expr_stacks else fn(stacks)

def create_instructions(fn: Callable, 
                        name_creator: Callable[[str],str] = None, 
                        base_name: str = None, 
                        input_stacks: Stacks = None, 
                        output_stacks: Stacks = None, 
                        code_blocks: int = 0, 
                        docstring: str = None, 
                        validator: Callable = None, 
                        limiter: Limiter = None, 
                        signature: Callable = None,
                        non_expr_stacks: set[str] = None,
                        is_child_stack: Callable = None):
    """Default function to process instructions. Creates an eager function. If limiter is not None, creates limited versions of the functions. If signature is not None, 
    creates graph versions of the functions."""
    non_expr_stacks = set() if non_expr_stacks is None else non_expr_stacks
    in_expr_stacks = transform_stacks(lambda x:f"{x}_expr", input_stacks, non_expr_stacks=non_expr_stacks)
    out_expr_stacks = transform_stacks(lambda x:f"{x}_expr", output_stacks, non_expr_stacks=non_expr_stacks)
    graph_fn = lambda *args,**kwargs: fn(*args, **{k.replace("_expr",""):v for k,v in kwargs.items()})
    instructions = []
    base_name = base_name or fn.__name__
    base_name = base_name if name_creator is None else name_creator(base_name)
    i1 = SimpleInstruction(f"{base_name}", fn, input_stacks, output_stacks, code_blocks, docstring=docstring, validator=validator)
    instructions.append(i1)
    if signature:
        i2 = SimpleExprInstruction(f"{base_name}_graph", graph_fn, signature, in_expr_stacks, out_expr_stacks, code_blocks, docstring=docstring, validator=validator)
        instructions.append(i2)
    if limiter:
        i3 = SimpleInstruction(f"{base_name}_limit", limiter.limit(fn), input_stacks, output_stacks, code_blocks, docstring=docstring, validator=validator)
        instructions.append(i3)
        if signature:
            i4 = SimpleExprInstruction(f"{base_name}_graph_limit", limiter.limit(graph_fn), signature, in_expr_stacks, out_expr_stacks, code_blocks, docstring=docstring, validator=validator)
            instructions.append(i4)
    return instructions


def simple_instruction(fn: Callable, 
                       name_creator: Callable[[str],str] = None, 
                       base_name: str = None, 
                       input_stacks: Stacks = None, 
                       output_stacks: Stacks = None, 
                       code_blocks: int = 0, 
                       docstring: str = None, 
                       validator: Callable = None, 
                       limiter: Limiter = None, 
                       **kwargs):
    base_name = base_name or fn.__name__
    base_name = base_name if name_creator is None else name_creator(base_name)
    if len([k for k in kwargs if kwargs[k] is not None])>0:
       warn(f"simple_instruction {base_name} ignoring additional kwargs: {[k for k in kwargs if kwargs[k] is not None]}")
    if limiter is not None:
        fn = limiter.limit(fn)
    return [SimpleInstruction(base_name,fn,input_stacks,output_stacks,code_blocks,docstring=docstring, validator=validator)]

def process_name_input(input: Union[str,tuple[str],dict[str,int]]):
    if isinstance(input,tuple):
        return input[0] 
    elif isinstance(input,dict):
        return next(iter(input))
    else:
        return input 


def default_polymorphic_name_creator(name: str, input: str, output: str):
        input,output = process_name_input(input),process_name_input(output)
        graph="_graph" if ("expr" in input or "expr" in output) else ""
        input = input.replace("_expr","") if isinstance(input,str) else input 
        output = output.replace("_expr","") if isinstance(output,str) else output 
        if input==output:
            return f"{input}_{name}{graph}"
        return f"{input}_{output}_{name}{graph}"

def polymorphic_instruction(fn: Callable, 
                       name_creator: Callable[[str,Stacks,Stacks],str] = None, 
                       base_name: str = None, 
                       input_stacks: list[Stacks]= None, 
                       output_stacks: list[Stacks] = None, 
                       code_blocks: int = 0, 
                       docstring: str = None, 
                       validator: Callable = None, 
                       limiter: Limiter = None, 
                       **kwargs):
    base_name = base_name or fn.__name__
    output_stacks = input_stacks if output_stacks is None else output_stacks
    if len([k for k in kwargs if kwargs[k] is not None])>0:
       warn(f"polymorphic_instruction {base_name} ignoring additional kwargs: {[k for k in kwargs if kwargs[k] is not None]}")
    if limiter is not None:
        fn = limiter.limit(fn)
    name_creator = default_polymorphic_name_creator if name_creator is None else name_creator
    return [SimpleInstruction(name_creator(base_name,i,o),fn,i,o,code_blocks,docstring=docstring, validator=validator) for i,o in zip(input_stacks,output_stacks)]
