from ..instruction import SimpleInstruction
from ..instruction_set import GLOBAL_INSTRUCTIONS
from ..state import PushState
from .utils import WrapperCreator
from ..limiter import DEFAULT_SIZE_LIMITER
from ..signature import broadcast_signature
import jax.numpy as jnp 
import jax.lax as lax 



float_jax_wrap = WrapperCreator(input_stacks={"float_jax":2}, output_stacks="float_jax", limiter=DEFAULT_SIZE_LIMITER, signature=broadcast_signature)
int_jax_wrap = WrapperCreator(input_stacks={"int_jax":2}, output_stacks="int_jax", limiter=DEFAULT_SIZE_LIMITER, signature=broadcast_signature)

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap()
def float_jax_add(**kwargs):
    x,y = kwargs["float_jax"]
    return x+y

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_jax_wrap()
def int_jax_add(**kwargs):
    x,y = kwargs["int_jax"]
    return x+y

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap()
def float_jax_sub(**kwargs):
    x,y = kwargs["float_jax"]
    return x - y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_jax_wrap()
def int_jax_sub(**kwargs):
    x,y = kwargs["int_jax"]
    return x - y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap()
def float_jax_mul(**kwargs):
    x,y = kwargs["float_jax"]
    return x * y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_jax_wrap()
def int_jax_mul(**kwargs):
    x,y = kwargs["int_jax"]
    return x * y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap()
def float_jax_div(**kwargs):
    x,y = kwargs["float_jax"]
    return jnp.where(y==0,0,lax.div(x,y))

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_jax_wrap()
def int_jax_div(**kwargs):
    x,y = kwargs["int_jax"]
    return jnp.where(y==0,0,lax.div(x,y))

