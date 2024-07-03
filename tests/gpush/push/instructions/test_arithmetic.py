import gpush.push.instructions.arithmetic as arithmetic
from gpush.push.instructions.utils import InstructionWrapper,create_instructions
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.state import PushState
from gpush.push.dag.expr import Function, Input, Parameter
from gpush.push.dag.dag import Dag
from copy import deepcopy
import pytest

@pytest.mark.parametrize("stack", ["float", "int"])
@pytest.mark.parametrize("name", ["add", "mul", "sub", "div"])
@pytest.mark.parametrize("x", [-10,-3,-1,0,1,3,10])
@pytest.mark.parametrize("y", [-10,-3,-1,0,1,3,10])
def test_arithmetic(stack, name, x, y):
    name = f"{stack}_{name}"
    fn = getattr(arithmetic,name)
    res = fn(**{stack: [x,y]})
    init_state = PushState(float=[], int=[])
    init_state[stack]=[x,y]
    init_graph_state = PushState(float_expr=[], int_expr=[])
    init_graph_state[f"{stack}_expr"]=[Input(0,0),Input(1,1)]
    init_graph_state.nsteps=2
    output_stack = fn.kwargs["output_stacks"]

    keys = [f"{name}"]
    if fn.kwargs.get("limiter",None) is not None:
        keys.append(f"{name}_limit")
    for k in keys:
        assert k in GLOBAL_INSTRUCTIONS

    if f"{name}" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}"](deepcopy(init_state))
        assert out_state.nsteps==1 
        assert out_state[output_stack] == [res]

    if f"{name}_limit" in GLOBAL_INSTRUCTIONS:
        limit_res = fn.kwargs["limiter"](fn(**{stack: [x,y]}))
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_limit"](deepcopy(init_state))
        assert out_state.nsteps==1 
        assert out_state[output_stack] == [limit_res]


@pytest.mark.parametrize("stack,x", [("float", 0.3), ("int", 3), ("float", 1), ("int", -3.5), ("int", 10.2)])
def test_cast(stack, x):
    name = "float_to_int" if stack=="float" else "int_to_float"
    fn = getattr(arithmetic,name)
    res = fn(x)
    init_state = PushState(float=[], int=[])
    init_state[stack]=[x]
    init_graph_state = PushState(float_expr=[], int_expr=[])
    init_state[f"{stack}_expr"]=[Input(0,0)]
    init_graph_state.nsteps=2
    output_stack = fn.kwargs["output_stacks"]

    keys = [f"{name}"]
    if fn.kwargs.get("limiter",None) is not None:
        keys.append(f"{name}_limit")
    if fn.kwargs.get("signature",None) is not None:
        keys.append(f"{name}_graph_limit")
        if fn.kwargs.get("limiter",None) is not None:
            keys.append(f"{name}_graph_limit")
    for k in keys:
        assert k in GLOBAL_INSTRUCTIONS

    if f"{name}" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}"](deepcopy(init_state))
        assert out_state.nsteps==1 
        assert out_state[output_stack] == [res]

    if f"{name}_limit" in GLOBAL_INSTRUCTIONS:
        limit_res = fn.kwargs["limiter"](fn(x))
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_limit"](deepcopy(init_state))
        assert out_state.nsteps==1 
        assert out_state[output_stack] == [limit_res]

