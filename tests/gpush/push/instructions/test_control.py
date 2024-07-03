from gpush.push.instruction import *
import gpush.push.instructions.jax as jax
import gpush.push.instructions.arithmetic as arithmetic
import gpush.push.instructions.control as control
from gpush.push.instructions.utils import InstructionWrapper
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.state import PushState
from gpush.push.dag.expr import Function, Input, Parameter
from gpush.push.dag.dag import Dag
from inspect import getmembers, isfunction
from copy import deepcopy
import pytest

@pytest.fixture
def do_times():
    assert "exec_do_times" in GLOBAL_INSTRUCTIONS 
    return GLOBAL_INSTRUCTIONS["exec_do_times"]

@pytest.mark.parametrize("exec", [ParamInstruction("p1",Shape(),"float","float_jax"), [GLOBAL_INSTRUCTIONS["int_add"],GLOBAL_INSTRUCTIONS["float_sub"]], GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"], []])
@pytest.mark.parametrize("int", [0, 1, 2, 5, -1])
def test_do_times(do_times, exec, int):
    init_state = PushState(exec=[exec], int=[int])
    res = do_times(init_state)
    assert res.nsteps==1 
    assert res["exec"]==[[exec for _ in range(int)]]
    assert res["int"]==[]


@pytest.mark.parametrize("stack",["int","int_jax","float_jax"])
@pytest.mark.parametrize("val", [(0,1,2,3),tuple(),("A","B","C")])
def test_dup(stack, val):
    init_state = PushState(**{stack:list(val)})
    instr = GLOBAL_INSTRUCTIONS[f"{stack}_dup"]
    final_state = instr(init_state)
    assert final_state.nsteps==1 
    assert final_state[stack]==((list(val)+[val[-1]]) if len(val)>0 else [])

@pytest.mark.parametrize("stack",["int","int_jax","float_jax"])
@pytest.mark.parametrize("val,res", [((0,1,2,3),(0,1,3,2)), 
                                     ((1,),(1,)), 
                                     ((1,2),(2,1)), 
                                     (tuple(),tuple()), 
                                     (("A","B","C"),("A","C","B"))])
def test_swap(stack, val,res):
    init_state = PushState(**{stack:list(val)})
    instr = GLOBAL_INSTRUCTIONS[f"{stack}_swap"]
    final_state = instr(init_state)
    assert final_state.nsteps==1 
    assert final_state[stack]==list(res)
