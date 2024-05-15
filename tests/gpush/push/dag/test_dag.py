from .test_expr import make_tree
from gpush.push.dag.dag import Dag
from gpush.push.dag.expr import Function, Parameter, Input
import jax.numpy as jnp 
import jax.lax as lax 

def test_eval():
    """Test whether evaluating a dag gives the expected results\n
    The leaf nodes p2, p3, i1, and i3 do not appear in the intermediate calculations, as they are not present in the tree."""

    params = [1,-4,50,2,3]
    input = [9,34,0,1,-34]

    (tree, parameters, inputs, functions, _functions), (params, input, (otree, oparameters, oinputs, ofunctions, _ofunctions)) = make_tree()
    
    for p,op in zip(parameters,oparameters):
        assert Dag(p).eval(params, input) == Dag(p)(params, input) == op 
    for i,oi in zip(inputs,oinputs):
        assert Dag(i).eval(params, input) == Dag(i)(params, input) == oi 
    for f,of in zip(functions,ofunctions):
        assert Dag(f).eval(params, input) == Dag(f)(params, input) == of 
    for k in _functions.keys():
        assert Dag(_functions[k]).eval(params, input) == Dag(_functions[k])(params, input) == _ofunctions[k]

    dag = Dag(tree)
    op0,op1,op2,op3,op4 = oparameters
    oi0,oi1,oi2,oi3,oi4 = oinputs
    assert dag.eval(params, input, return_intermediate=True) == dag(params, input, return_intermediate=True) == (otree, [op0,op1,op4,oi0,oi2,oi4] + ofunctions)

def test_eval_2():
    """Test a tree of jax operations"""
    params = [jnp.array([1,2,3,4,5,6,7,8,9,10.]), jnp.ones((3,4)), jnp.ones((3,3,3)), jnp.array([[1,2,2.],[3,4,5], [9,8,7]])]# Shapes = (10,), (3,4), (3,3,3), (3,3)
    input = [jnp.array([[1],[2],[3],[4.]]), jnp.ones((2,1,2)), jnp.zeros((2,4)), jnp.array([[[1,2.], [3,4]], [[5,6], [7,8]]])]# Shape = (4,1), (2,1,2), (2,4), (2,2,2)

    p0,p1,p2,p3 = parameters = [Parameter(i,param_idx=i) for i in range(4)]
    i0,i1,i2,i3 = inputs =  [Input(i+4, input_idx=i) for i in range(4)]

    # Functions of leaves
    def div(x,y):
        x,y = jnp.broadcast_arrays(x,y)
        return lax.div(x,y)
    
    f0 = Function(id = 10, fn = lambda x,y:x+y, children = (i1,i3))# shape = (2,2,2)
    f1 = Function(id = 14, fn = div, children = (p2, p3))# Shape = (3,3,3)
    f2 = Function(id = 35, fn = lambda arg0=None, arg1=None, arg2=None: arg0[0]@arg1[0], children = {"arg0": [p3], "arg1": [p1], "arg2": []})# Shape = (3,3,4)
    f3 = Function(id = 55, fn = lambda x:x*10, children = (p0,))# Shape = (10,)
    f4 = Function(id = 60, fn = lambda x,y,z:x@y+z.T, children = (f1,f2,i0))# Shape = (3,3,4)

    o0 = input[1]+input[3]
    o1 = div(params[2],params[3])
    o2 = params[3]@params[1]
    o3 = params[0]*10
    o4 = o1@o2+input[0].T

    tree = f4 
    functions = [f0,f1,f2,f3,f4]
    otree = o4 
    ofunctions = [o0,o1,o2,o3,o4]

    for p,op in zip(parameters,params):
        eval_res = Dag(p).eval(params, input)
        call_res = Dag(p)(params, input)
        assert jnp.allclose(eval_res,op)
        assert jnp.allclose(call_res,op)
    for i,oi in zip(inputs,input):
        eval_res = Dag(i).eval(params, input)
        call_res = Dag(i)(params, input)
        assert jnp.allclose(eval_res,oi)
        assert jnp.allclose(call_res,oi)
    for f,of in zip(functions,ofunctions):
        eval_res = Dag(f).eval(params, input)
        call_res = Dag(f)(params, input)
        assert jnp.allclose(eval_res,of)
        assert jnp.allclose(call_res,of)

    dag = Dag(tree)
    eval_res = dag.eval(params, input)
    call_res = dag(params, input)
    assert jnp.allclose(eval_res,otree)
    assert jnp.allclose(call_res,otree)

