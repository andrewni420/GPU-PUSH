from ..instruction import SimpleInstruction
from ..instruction_set import GLOBAL_INSTRUCTIONS
from ..state import PushState
from ..program import CodeBlock
from .utils import simple_instruction, WrapperCreator, polymorphic_instruction, default_polymorphic_name_creator


data_stacks = ["float","int","float_jax","float_jax_expr","int_jax","int_jax_expr"]
wrap = WrapperCreator()
control_wrap = WrapperCreator(input_stacks=("exec","int"), output_stacks="exec", code_blocks=1)


def copy_block(codeblock: CodeBlock):
    if isinstance(codeblock,list):
        return [copy_block(instr) for instr in codeblock]
    else:
        return codeblock.copy()
    
@GLOBAL_INSTRUCTIONS.unpack_register(simple_instruction)
@control_wrap()
def exec_do_times(codeblock: CodeBlock, n: int):
    return [copy_block(codeblock) for _ in range(n)]

@GLOBAL_INSTRUCTIONS.unpack_register(polymorphic_instruction)
@wrap(input_stacks = [{k:1} for k in data_stacks], output_stacks=[{k:2} for k in data_stacks])
def dup(**kwargs):
    k = next(iter(kwargs))
    return {k:kwargs[k]*2}

@GLOBAL_INSTRUCTIONS.unpack_register(polymorphic_instruction)
@wrap(input_stacks=[{s:2} for s in data_stacks])
def swap(**kwargs):
    return {k:v[::-1] for k,v in kwargs.items()}

# def dup_times_name_creator(name, input, output):
#     input = input[0] if isinstance(input,tuple) else next(iter(input))
#     graph="_graph" if ("expr" in input or "expr" in output) else ""
#     if input==output:
#         print(f"{input}_{name}{graph}")
#         return f"{input}_{name}{graph}"
#     return f"{input}_{output}_{name}{graph}"

# @GLOBAL_INSTRUCTIONS.unpack_register(polymorphic_instruction)
# @dup_wrap(input_stacks=[("float","int"),{"int":2},("float_jax","int"),("float_jax_expr","int"),("int_jax","int"),("int_jax_expr","int")],
#           output_stacks = ["float","int","float_jax","float_jax_expr","int_jax","int_jax_expr"],
#           name_creator = dup_times_name_creator)
# def dup_times(x,n):
#     return [x]*n






