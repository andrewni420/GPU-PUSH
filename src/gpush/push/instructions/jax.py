from ..instruction import SimpleInstruction
from ..instruction_set import GLOBAL_INSTRUCTIONS, ACTIVATION_INSTRUCTIONS
from ..state import PushState
from .utils import WrapperCreator
from ..limiter import DEFAULT_SIZE_LIMITER
from ..signature import broadcast_signature, mmul_signature, cast_signature
import jax.numpy as jnp 
import jax.lax as lax 
import jax.nn as nn
from functools import partial 



float_jax_wrap = WrapperCreator(input_stacks={"float_jax":2}, output_stacks="float_jax", limiter=DEFAULT_SIZE_LIMITER, signature=broadcast_signature)
int_jax_wrap = WrapperCreator(input_stacks={"int_jax":2}, output_stacks="int_jax", limiter=DEFAULT_SIZE_LIMITER, signature=broadcast_signature)
cast_to_float = partial(cast_signature,cast_to="float")

######################################
######## Arithmetic Functions ########
######################################

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

##################################
######## Binary Functions ########
##################################

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=mmul_signature)
def float_jax_mmul(**kwargs):
    x,y = kwargs["float_jax"]
    return x @ y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@int_jax_wrap(signature=mmul_signature)
def int_jax_mmul(**kwargs):
    x,y = kwargs["int_jax"]
    return x @ y 

######################################
######## Activation Functions ########
######################################
@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_relu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.relu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_relu6(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.relu6(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_sigmoid(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.sigmoid(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_softplus(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.softplus(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_silu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.silu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_leaky_relu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.leaky_relu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_elu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.elu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_celu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.leaky_relu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_selu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.selu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_tanh(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.tanh(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_glu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.glu(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_gelu(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.gelu(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_squareplus(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.squareplus(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_mish(**kwargs):
    x = kwargs["float_jax"][0]
    return nn.mish(x)





