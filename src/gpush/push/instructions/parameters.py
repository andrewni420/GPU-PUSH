from ..instruction import SimpleInstruction
from ..instruction_set import GLOBAL_INSTRUCTIONS
from ..dag.shape import Shape 
from ..state import PushState
from .utils import WrapperCreator
from ..limiter import DEFAULT_SIZE_LIMITER
from ..signature import broadcast_signature
import jax.numpy as jnp 
import jax.lax as lax 


param_wrap = lambda i:WrapperCreator(input_stacks={"int":i}, output_stacks="float_expr")



@GLOBAL_INSTRUCTIONS.unpack_register
@param_wrap()
def float_add(**kwargs):
    shape = Shape(*kwargs["int"])
    return 
