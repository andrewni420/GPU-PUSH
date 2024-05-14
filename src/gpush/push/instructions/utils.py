from ..instruction import SimpleInstruction, SimpleExprInstruction
from ..limiter import Limiter 
from typing import Callable, Union 
from abc import ABC, abstractmethod
from copy import copy 

class InstructionWrapper():
    "A wrapper around an instruction that stores additional values"
    def __init__(self, fn, *args, **kwargs):
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
                 input_stacks: Union[list[str], dict[str,int], str]= None, 
                 output_stacks: Union[list[str], dict[str,int], str] = None, 
                 code_blocks: int = 0, 
                 docstring: str = None, 
                 validator: Callable = None, 
                 limiter: Limiter = None, 
                 signature: Callable = None,
                 **kwargs):
        kwargs.update({"name_creator":name_creator, "base_name":base_name, "input_stacks":input_stacks, "output_stacks":output_stacks, "code_blocks":code_blocks, "docstring":docstring, "validator":validator, "limiter":limiter, "signature":signature})
        self.default = kwargs 
    def __call__(self, **kwargs):
        default = copy(self.default)
        default.update(kwargs)
        def wrap(fn):
            return InstructionWrapper(fn, **default)
        return wrap
    
def transform_stacks(fn, stacks):
    "Applies `fn` to all of the stack names in `stacks`"
    if isinstance(stacks,tuple):
        return tuple(fn(s) for s in stacks)
    elif isinstance(stacks, dict):
        return {fn(k):v for k,v in stacks.items()}
    else:
        return fn(stacks)

def create_instructions(fn, 
                        name_creator: Callable[[str],str] = None, 
                        base_name: str = None, 
                        input_stacks: Union[list[str], dict[str,int], str]= None, 
                        output_stacks: Union[list[str], dict[str,int], str] = None, 
                        code_blocks: int = 0, 
                        docstring: str = None, 
                        validator: Callable = None, 
                        limiter: Limiter = None, 
                        signature: Callable = None):
    """Default function to process instructions. Creates an eager function. If limiter is not None, creates limited versions of the functions. If signature is not None, 
    creates graph versions of the functions."""
    in_expr_stacks = transform_stacks(lambda x:f"{x}_expr", input_stacks)
    out_expr_stacks = transform_stacks(lambda x:f"{x}_expr", output_stacks)
    graph_fn = lambda *args,**kwargs: fn(*args, **{k.replace("_expr",""):v for k,v in kwargs.items()})
    instructions = []
    base_name = base_name or fn.__name__
    base_name = base_name if name_creator is None else name_creator(base_name)
    i1 = SimpleInstruction(f"{base_name}_eager", fn, input_stacks, output_stacks, code_blocks, docstring=docstring, validator=validator)
    instructions.append(i1)
    if signature:
        i2 = SimpleExprInstruction(f"{base_name}_graph", graph_fn, signature, in_expr_stacks, out_expr_stacks, code_blocks, docstring=docstring, validator=validator)
        instructions.append(i2)
    if limiter:
        i3 = SimpleInstruction(f"{base_name}_eager_limit", limiter.limit(fn), input_stacks, output_stacks, code_blocks, docstring=docstring, validator=validator)
        instructions.append(i3)
        if signature:
            i4 = SimpleExprInstruction(f"{base_name}_graph_limit", limiter.limit(graph_fn), signature, in_expr_stacks, out_expr_stacks, code_blocks, docstring=docstring, validator=validator)
            instructions.append(i4)
    return instructions



