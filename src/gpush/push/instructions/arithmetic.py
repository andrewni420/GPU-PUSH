from ..instruction import SimpleInstruction
from ..instruction_set import GLOBAL_INSTRUCTIONS
from ..state import PushState
from .utils import WrapperCreator, simple_instruction
from ..limiter import DEFAULT_SIZE_LIMITER
from ..signature import broadcast_signature
import jax.numpy as jnp 
import jax.lax as lax 


float_wrap = WrapperCreator(input_stacks={"float":2}, output_stacks="float", limiter=DEFAULT_SIZE_LIMITER)
int_wrap = WrapperCreator(input_stacks={"int":2}, output_stacks="int", limiter=DEFAULT_SIZE_LIMITER)

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_wrap()
def float_add(**kwargs):
    x,y = kwargs["float"]
    return x+y

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_wrap()
def int_add(**kwargs):
    x,y = kwargs["int"]
    return x+y

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_wrap()
def float_sub(**kwargs):
    x,y = kwargs["float"]
    return x - y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_wrap()
def int_sub(**kwargs):
    x,y = kwargs["int"]
    return x - y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_wrap()
def float_mul(**kwargs):
    x,y = kwargs["float"]
    return x * y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_wrap()
def int_mul(**kwargs):
    x,y = kwargs["int"]
    return x * y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_wrap()
def float_div(**kwargs):
    x,y = kwargs["float"]
    return jnp.where(y==0,0,lax.div(x,y))

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_wrap()
def int_div(**kwargs):
    x,y = kwargs["int"]
    return jnp.where(y==0,0,lax.div(x,y))

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_wrap(input_stacks = "int", output_stacks = "float")
def int_to_float(x):
    return float(x)

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_wrap(input_stacks = "float", output_stacks = "int")
def float_to_int(x):
    return int(x)

