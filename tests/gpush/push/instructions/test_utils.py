from gpush.push.instructions.arithmetic import *
from gpush.push.instruction import *
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.state import PushState
from gpush.push.dag.dag import Dag
from gpush.push.instructions.utils import WrapperCreator, create_instructions
from gpush.push.limiter import LambdaLimiter,SizeLimiter
from copy import copy,deepcopy
import pytest
import jax.numpy as jnp 

def assert_eval_result(init_state: PushState, instr: str, final_state: PushState):
    instruction = GLOBAL_INSTRUCTIONS[instr]
    assert instruction(deepcopy(init_state))==final_state

@pytest.mark.parametrize("default", [{"hi":1,"bye":3,"adios":2}, {"bar":[1,2,3],"zero":(1,2),"foo":"bar"}])
@pytest.mark.parametrize("update", [{"hi":1,"asvab":2}, {"bar":[1,2,3],"one":4}, {"adios":0,"bar":-1,"foo":"foo"}])
def test_creator(default, update):
    creator = WrapperCreator(**default)

    @creator(**update)
    def test(*args,**kwargs):
        return args,kwargs 
    default = copy(default)
    default.update(update)
    for k in default.keys():
        assert test.kwargs[k]==default[k]

    def dictify(fn, **kwargs):
        kwargs["fn"]=fn 
        return kwargs 
    default.update({"fn":test.fn})
    dictified = test.apply(dictify)
    assert dictified["fn"].__name__=="test" 

@pytest.mark.parametrize("limiter", [None, lambda x:x+1, lambda x:2*x,lambda x:min(max(x,0),1), SizeLimiter(low=-2,high=5)])
@pytest.mark.parametrize("signature", [None, lambda x:(1,1), lambda x:(2,2)])
def test_create_instr(limiter, signature):
    input_stacks = "int"
    output_stacks = "int"
    def instr(x):
        return x+1
    l = limiter 
    if l is not None and not isinstance(l,SizeLimiter):
        l = LambdaLimiter(l)
    instructions = create_instructions(instr,input_stacks=input_stacks, output_stacks=output_stacks,limiter=l, signature=signature)
    inputs = [-10,-3, -1, -0.5, 0, 0.5, 1, 3, 10]
    if signature:
        names = ["instr", "instr_graph"]
        names+= ["instr_limit", "instr_graph_limit"] if limiter else []
    else:
        names = ["instr"]
        names+= ["instr_limit"] if limiter else []
    assert [i.name for i in instructions]==names 

    def output_test(f,input):
        res = input+1
        if "limit" in f.name:
            res = res if limiter is None else limiter(res)
        if "graph" in f.name:
            state = PushState({"int_expr":[Input(0)]})
            state.nsteps+=1
            output = f(state)["int_expr"][0]
            assert Dag(output).eval([],[input]) == res
            return 
        assert f(PushState({"int":[input]}))["int"]==[res]
    
    for f in instructions:
        for i in inputs:
            output_test(f,i)








