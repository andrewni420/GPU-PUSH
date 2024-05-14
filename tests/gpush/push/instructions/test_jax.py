from gpush.push.instruction import *
import gpush.push.instructions.jax as jax
from gpush.push.instructions.utils import InstructionWrapper
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.state import PushState
from gpush.push.dag.expr import Function, Input, Parameter
from gpush.push.dag.dag import Dag
from inspect import getmembers, isfunction
from copy import deepcopy
import pytest

@pytest.mark.parametrize("stack", ["float_jax", "int_jax"])
@pytest.mark.parametrize("name", ["add", "mul", "sub", "div"])
@pytest.mark.parametrize("x", [-10,-3,-1,0,1,3,10])
@pytest.mark.parametrize("y", [-10,-3,-1,0,1,3,10])
def test_jax_arithmetic(stack, name, x, y):
    name = f"{stack}_{name}"
    fn = getattr(jax,name)
    res = fn(**{stack: [x,y]})
    init_state = PushState(float_jax=[], int_jax=[])
    init_state[stack]=[x,y]
    init_graph_state = PushState(float_jax_expr=[], int_jax_expr=[])
    init_graph_state[f"{stack}_expr"]=[Input(0,0,dtype=stack.split("_")),Input(1,1,dtype=stack.split("_"))]
    init_graph_state.nsteps=2
    output_stack = fn.kwargs["output_stacks"]

    keys = [f"{name}_eager"]
    if fn.kwargs.get("limiter",None) is not None:
        keys.append(f"{name}_eager_limit")
    if fn.kwargs.get("signature",None) is not None:
        keys.append(f"{name}_graph_limit")
        if fn.kwargs.get("limiter",None) is not None:
            keys.append(f"{name}_graph_limit")
    for k in keys:
        assert k in GLOBAL_INSTRUCTIONS

    if f"{name}_eager" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_eager"](deepcopy(init_state))
        assert out_state.nsteps==1 
        assert out_state[output_stack] == [res]

    if f"{name}_eager_limit" in GLOBAL_INSTRUCTIONS:
        limit_res = fn.kwargs["limiter"](fn(**{stack: [x,y]}))
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_eager_limit"](deepcopy(init_state))
        assert out_state.nsteps==1
        assert out_state[output_stack] == [limit_res]

    if f"{name}_graph" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_graph"](deepcopy(init_graph_state))
        assert out_state.nsteps==3
        output = out_state[f"{output_stack}_expr"][0]
        assert output.shape==Shape()
        assert Dag(output).eval([],[x,y]) == res

    if f"{name}_graph_limit" in GLOBAL_INSTRUCTIONS:
        limit_res = fn.kwargs["limiter"](fn(**{stack: [x,y]}))
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_graph_limit"](deepcopy(init_graph_state))
        assert out_state.nsteps==3
        output = out_state[f"{output_stack}_expr"][0]
        assert output.shape==Shape()
        assert Dag(output).eval([],[x,y]) == res