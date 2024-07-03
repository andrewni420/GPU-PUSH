from ..instruction import SimpleInstruction
from ..instruction_set import GLOBAL_INSTRUCTIONS, ACTIVATION_INSTRUCTIONS
from ..state import PushState
from .utils import WrapperCreator
from ..limiter import DEFAULT_SIZE_LIMITER
from ..signature import broadcast_signature, mmul_signature, cast_signature, conv_signature, pool_signature, flatten_signature
from ...utils import preprocess
import jax.numpy as jnp 
import jax.lax as lax 
import jax.nn as nn
from functools import partial, wraps, update_wrapper



float_jax_wrap = WrapperCreator(input_stacks={"float_jax":2}, output_stacks="float_jax", limiter=DEFAULT_SIZE_LIMITER, signature=broadcast_signature)
int_jax_wrap = WrapperCreator(input_stacks={"int_jax":2}, output_stacks="int_jax", limiter=DEFAULT_SIZE_LIMITER, signature=broadcast_signature)
cast_to_float = partial(cast_signature,cast_to="float")
activation_wrap = WrapperCreator(input_stacks={"float_jax":1}, output_stacks="float_jax", limiter=DEFAULT_SIZE_LIMITER, signature=cast_to_float)

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

# @GLOBAL_INSTRUCTIONS.unpack_register()
# @int_jax_wrap(signature=mmul_signature)
# def int_jax_mmul(**kwargs):
#     x,y = kwargs["int_jax"]
#     return x @ y 

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=conv_signature)
def float_jax_conv_vanilla(**kwargs):
    """Default n-dimensional convolution operator, with stride = 1, padding = 0, and dilation = 0.
    \nAssumes input shapes of (CHW...) and (OIHW...), with no batch dimension."""
    x,y = kwargs["float_jax"]
    n = y.ndim-2
    return lax.conv_general_dilated(x[None,...],y,(1,)*n,((0,0),)*n).squeeze(0)

def conv_preprocessor(*args,**kwargs):
    "Preprocesses arguments from the int stack into optional arguments for convolution"
    # print(f"conv preprocesser {kwargs}")
    stride, padding, lhs, rhs = kwargs["int"]
    kwargs = {k:v for k,v in kwargs.items() if k!="int"}
    kwargs["stride"] = stride if stride>=1 else 1
    kwargs["padding"] = padding if padding>=0 else "SAME"
    kwargs["lhs"] = lhs if lhs>=1 else 1
    kwargs["rhs"] = rhs if rhs>=1 else 1
    return args,kwargs 

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(input_stacks = {"float_jax":2,"int":4}, 
                signature=preprocess(conv_preprocessor,conv_signature),
                non_expr_stacks = {"int"})
@partial(preprocess,conv_preprocessor)
def float_jax_conv(**kwargs):
    """General-ish convolution operator. Takes elements from the int stack to specify the stride, padding, lhs dilation, and rhs dilation along all spatial axes. 
    \nIs protected in the sense that invalid arguments are replaced with default values.
    \nHowever, in contrast to `float_jax_conv_vanilla`, the default padding is "SAME" to make it easier to do residual connections.
    \nAssumes input shapes of (CHW...) and (OIHW...), with no batch dimension."""
    x,y = kwargs["float_jax"]
    stride = kwargs["stride"]
    padding = kwargs["padding"]
    lhs = kwargs["lhs"]
    rhs = kwargs["rhs"]
    n = y.ndim-2
    return lax.conv_general_dilated(x[None,...],y,(stride,)*n,padding if isinstance(padding,str) else ((padding,padding),)*n, (lhs,)*n,(rhs,)*n).squeeze(0)

######################################
######## Activation Functions ########
######################################
@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_relu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.relu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_relu6(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.relu6(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_sigmoid(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.sigmoid(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_softplus(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.softplus(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_silu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.silu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_leaky_relu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.leaky_relu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_elu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.elu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_celu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.leaky_relu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_selu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.selu(x) 

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_tanh(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.tanh(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=cast_to_float)
def float_jax_glu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.glu(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_gelu(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.gelu(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_squareplus(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.squareplus(x)

@ACTIVATION_INSTRUCTIONS.unpack_register()
@GLOBAL_INSTRUCTIONS.unpack_register()
@activation_wrap()
def float_jax_mish(**kwargs):
    [x] = kwargs["float_jax"]
    return nn.mish(x)

####################################
######## Reducing Functions ########
####################################

def vanilla_pooling_preprocessor(*args, **kwargs):
    w,h = kwargs["int"]
    kwargs = {k:v for k,v in kwargs.items() if k!="int"}
    kwargs["dimensions"]=[w,h]
    return args,kwargs

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=pool_signature, input_stacks={"float_jax":1,"int":2})
@partial(preprocess,vanilla_pooling_preprocessor)
def float_jax_max_pool_vanilla(dimensions=None, **kwargs):
    [x] = kwargs["float_jax"]
    return lax.reduce_window(x,-jnp.inf,lax.max,dimensions)

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(signature=conv_signature)
@partial(preprocess,vanilla_pooling_preprocessor)
def float_jax_sum_pool_vanilla(dimensions=None, **kwargs):
    [x] = kwargs["float_jax"]
    return lax.reduce_window(x,0,lax.sum,dimensions)

#####################################
######## Reshaping Functions ########
#####################################

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(input_stacks = {"float_jax":1}, signature=flatten_signature)
def float_jax_flatten_vanilla(**kwargs):
    "Default flattening operation that completely flattens the array. Assumes no batch dimension."
    [x] = kwargs["float_jax"]
    return jnp.reshape(x,[-1])

def flatten_preprocessor(*args, **kwargs):
    [ndim] = kwargs["int"]
    kwargs = {k:v for k,v in kwargs.items() if k!="int"}
    kwargs["ndim"]=ndim 
    return args,kwargs

@GLOBAL_INSTRUCTIONS.unpack_register()
@float_jax_wrap(input_stacks = {"float_jax":1, "int":1}, signature=flatten_signature)
@partial(preprocess,flatten_preprocessor)
def float_jax_flatten(ndim = 1,**kwargs):
    "Flattens into an ndim-dimensional array by coalescing all further dimensions into the ndim'th dimension"
    [x] = kwargs["float_jax"]
    return lax.reshape(x,x.shape[:ndim-1]+[-1])

