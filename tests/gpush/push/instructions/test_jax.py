from gpush.push.instruction import *
import gpush.push.instructions.jax as jax
from gpush.push.instructions.utils import InstructionWrapper
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.state import PushState, make_state
from gpush.push.dag.expr import Function, Input, Parameter
from gpush.push.dag.dag import Dag
from gpush.push.dag.shape import Shape, SizePlaceholder
from inspect import getmembers, isfunction
from copy import deepcopy
import pytest
import jax.numpy as jnp 

def assert_fn_variants(name):
    """Gets the names of all the defined variants of the function. 
    For example, float_jax_mul -> [float_jax_mul, float_jax_mul_limit, float_jax_mul_graph, float_jax_mul_graph_limit]
    Also ensures that all of the defined variants are present in the global instructions"""
    fn = getattr(jax,name)
    keys = [f"{name}"]
    if fn.kwargs.get("limiter",None) is not None:
        keys.append(f"{name}_limit")
    if fn.kwargs.get("signature",None) is not None:
        keys.append(f"{name}_graph_limit")
        if fn.kwargs.get("limiter",None) is not None:
            keys.append(f"{name}_graph_limit")
    for k in keys:
        assert k in GLOBAL_INSTRUCTIONS

def assert_single_stack_instruction(stack, name, *inputs):
    "Utility function to assert that a function that uses only a single stack works"
    name = f"{stack}_{name}"
    fn = getattr(jax,name)
    res = fn(**{stack: list(inputs)})
    limit_res = fn.kwargs["limiter"](fn(**{stack: list(inputs)})) if ("limiter" in fn.kwargs and fn.kwargs["limiter"] is not None) else None 
    init_graph_state = make_state(list(inputs), "graph",stacks={f"{stack}_expr":[Input(i,i,shape=e.shape) for i,e in enumerate(inputs)]})
    init_graph_state.nsteps=len(inputs)
    init_state = make_state([],"eager",stacks={stack:list(inputs)})

    assert_fn_variants(name)

    if f"{name}" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}"](deepcopy(init_state))
        assert out_state.nsteps==1 
        assert out_state[stack] == [res]

    if f"{name}_limit" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_limit"](deepcopy(init_state))
        assert out_state.nsteps==1
        assert out_state[stack] == [limit_res]

    if f"{name}_graph" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_graph"](deepcopy(init_graph_state))
        assert out_state.nsteps==len(inputs)+1
        output = out_state[f"{stack}_expr"][0]
        assert output.shape==res.shape
        out_res,out_params = Dag(output).eval([],list(inputs))
        assert out_res == res
        assert out_params == []

    if f"{name}_graph_limit" in GLOBAL_INSTRUCTIONS:
        out_state = GLOBAL_INSTRUCTIONS[f"{name}_graph_limit"](deepcopy(init_graph_state))
        assert out_state.nsteps==len(inputs)+1
        output = out_state[f"{stack}_expr"][0]
        assert output.shape==res.shape
        out_res,out_params = Dag(output).eval([],list(inputs))
        assert out_res == res
        assert out_params == []

@pytest.mark.parametrize("stack", ["float_jax", "int_jax"])
@pytest.mark.parametrize("name", ["add", "mul", "sub", "div"])
@pytest.mark.parametrize("x", [-10,-3,-1,0,1,3,10])
@pytest.mark.parametrize("y", [-10,-3,-1,0,1,3,10])
def test_arithmetic(stack, name, x, y):
    assert_single_stack_instruction(stack,name,jnp.array(x),jnp.array(y))

@pytest.mark.parametrize("stack", ["float_jax"])
@pytest.mark.parametrize("name", ["cos", "sin", "relu", "relu6", "sigmoid", "tanh", "softplus", "silu", "leaky_relu", "elu", "celu", "selu", "gelu", "squareplus", "mish"])
@pytest.mark.parametrize("x", [-10.,-3.,-1.4, -1.,-0.5, -0.1,0.,0.1, 0.5,1., 1.4, 3.,10.])
def test_one_to_one(stack, name, x):
    assert_single_stack_instruction(stack,name,jnp.array(x))

@pytest.mark.parametrize("prefix", ["float_jax", "int_jax"])
@pytest.mark.parametrize("param_first", [True, False])
def test_param_update(prefix, param_first):
    stack = f"{prefix}_expr"
    name = f"{prefix}_param_update"
    inputs = [jnp.array(1),jnp.array([2,3]), jnp.array([[-1,2]])]
    i0 = Input(0,0,Shape())
    i1 = Input(1,1,Shape(2))
    i2 = Input(2,2,Shape(1,2))
    f1 = Function(3,jnp.add,(i0,i1),shape=Shape(2))
    f2 = Function(4,jnp.multiply,(i2,f1),shape=Shape(1,2))
    p1 = Parameter(5,1,Shape(1,2))

    start_stacks = {stack:[p1,f2]} if param_first else {stack:[f2,p1]}

    init_graph_state = make_state(list(inputs), "graph",stacks=start_stacks, params=[None, {"shape":Shape(1,2), "dtype":"float"}, None])
    init_graph_state.nsteps=6

    assert name in GLOBAL_INSTRUCTIONS
    out_state = GLOBAL_INSTRUCTIONS[name](deepcopy(init_graph_state))
    assert out_state.nsteps==7
    output = out_state[stack][0]
    assert output.id==6
    assert output.shape==(1,2)
    params = [None, None, None]
    out_res,out_params = Dag(output).eval([None, None, None],list(inputs))
    assert jnp.allclose(out_res,jnp.array([[-3,8]]))
    assert out_params == [None, out_res, None]
    assert params == [None, None, None]

@pytest.mark.parametrize("prefix", ["float", "int"])
@pytest.mark.parametrize("idx", [-10,-5,-4,-3,-2,-1,0,1,2,3,4,10])
def test_param_from_idx(prefix, idx):
    stack = f"{prefix}_jax_expr"
    name = f"{prefix}_jax_param_from_idx"
    params = [{"shape": Shape(), "dtype": "float"},
              {"shape": Shape(3,4), "dtype": "float"},
              {"shape": Shape(SizePlaceholder(),4), "dtype": "int"},
              {"shape": Shape(3,SizePlaceholder().link(4)), "dtype": "float"}]
    
    init_graph_state = make_state([], "graph", stacks={"int":[idx]}, params=params)

    assert name in GLOBAL_INSTRUCTIONS

    out_state = GLOBAL_INSTRUCTIONS[name](deepcopy(init_graph_state))
    assert out_state.nsteps==1
    if -5<idx<4 and prefix==params[idx]["dtype"]:
        [output] = out_state[stack]
        assert len(out_state["int"])==0
        assert isinstance(output,Parameter)
        assert output.id==0
        assert output.shape==params[idx]["shape"]
        assert output.dtype==params[idx]["dtype"]
        assert output.param_idx == idx
    else:
        keys = set(out_state.keys())
        keys.update(init_graph_state.keys())
        for k in keys:
            assert out_state[k]==init_graph_state[k]