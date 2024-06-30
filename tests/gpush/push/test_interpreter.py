from gpush.push.interpreter import * 
from gpush.push.instructions.jax import *
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.instruction import InputInstruction, ParamInstruction
from gpush.push.compiler import PlushyCompiler
from gpush.push.state import PushState
from gpush.push.dag.shape import Shape
from gpush.push.dag.dag import Dag
import jax.numpy as jnp
import jax.random as random 
import jax 
from jax import grad, jit

def test_linreg():
    """Tests whether we can use GPush to do linear regression"""
    # Test single evaluation
    program = [InputInstruction("in1",0,"float_jax_expr"), 
               ParamInstruction("p1",Shape(3,1),"float","float_jax_expr"), 
               GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
               ParamInstruction("p2", Shape(), "float", "float_jax_expr"),
               GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[]).initialize([],[{"shape":Shape(3,3),"dtype":"float"}])
    expr = Interpreter().run(program, state=start_state)["float_jax_expr"][0]
    dag = Dag(expr)
    arr1 = jnp.array([[1.],[2.],[3.]])
    arr2 = jnp.array([[1.,2.,3.],[4.,5.,6.],[5.,2.,6.]])
    arr3 = jnp.array(4)
    output = dag.eval([arr1, arr3],[arr2])
    assert jnp.allclose(output, arr2@arr1+arr3)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,4)
    inputs = random.normal(subkeys[0],shape=(10,3))
    targets = jnp.reshape(inputs[:,0]+0.2*inputs[:,1]-0.5*inputs[:,2]+1.3, [-1,1])
    w = random.normal(subkeys[1], shape=(3,1))
    b = random.normal(subkeys[2], shape=tuple())

    # Loss / grad functions
    def loss(output, target):
        return jnp.mean((output-target)**2)
    grad_fn = dag.grad(loss)
    
    def actual(params, input):
        [w,b] = params 
        return input@w+b 
    actual_grad = jit(grad(lambda params, input, output: loss(output, actual(params,input))))

    # Training loop
    for i in range(1000):
        dw,db = grad_fn([w,b],[inputs], targets)
        actual_dw,actual_db = actual_grad([w,b],inputs,targets)
        
        output = dag.eval([w,b], [inputs])
        assert jnp.allclose(output, actual([w,b],inputs))
        assert jnp.allclose(dw,actual_dw)
        assert jnp.allclose(db,actual_db)
        
        w-=dw*0.1
        b-=db*0.1 
    
    # Does training converge to the right values?
    final_loss = loss(dag.eval([w,b], [inputs]), targets)
    assert final_loss<0.001
    assert jnp.allclose(w,jnp.array([[1],[0.2],[-0.5]]))
    assert jnp.allclose(b,jnp.array(1.3))

def test_mlp():
    """Tests whether we can use GPush to train a multilayer perceptron"""
    # Test single evaluation
    program = [InputInstruction("in1",0,"float_jax_expr"), 
               # Layer 1
               ParamInstruction("p1",Shape(3,5),"float","float_jax_expr"), 
               GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
               ParamInstruction("p2", Shape(5), "float", "float_jax_expr"),
               GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
               # Layer 2
               ParamInstruction("p3",Shape(5,4),"float","float_jax_expr"), 
               GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
               ParamInstruction("p4", Shape(4), "float", "float_jax_expr"),
               GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
               # Layer 3
               ParamInstruction("p5",Shape(4,1),"float","float_jax_expr"), 
               GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
               ParamInstruction("p6", Shape(), "float", "float_jax_expr"),
               GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[]).initialize([],[{"shape":Shape(3,3),"dtype":"float"}])
    expr = Interpreter().run(program, state=start_state)["float_jax_expr"][0]
    dag = Dag(expr)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,8)
    inputs = random.normal(subkeys[0],shape=(10,3))
    targets = jnp.reshape(inputs[:,0]+0.2*inputs[:,1]**2-0.5*inputs[:,2]**3+1.3, [-1,1])
    
    # Initialization
    w1 = random.normal(subkeys[1], shape=(3,5))
    b1 = random.normal(subkeys[2], shape=(5,))
    w2 = random.normal(subkeys[3], shape=(5,4))
    b2 = random.normal(subkeys[4], shape=(4,))
    w3 = random.normal(subkeys[5], shape=(4,1))
    b3 = random.normal(subkeys[6], shape=tuple())
    params = [w1,b1,w2,b2,w3,b3]

    # Loss / grad functions
    def loss(output, target):
        return jnp.mean((output-target)**2)
    grad_fn = dag.grad(loss)
    
    def actual(params, input):
        [w1,b1, w2,b2, w3,b3] = params 
        x = input 
        x = x@w1+b1 
        x = x@w2+b2 
        x = x@w3 + b3 
        return x 
    actual_grad = jit(grad(lambda params, input, output: loss(actual(params,input), output)))

    # Training loop
    for i in range(300):
        dparams = grad_fn(params,[inputs], targets)
        actual_dparams = actual_grad(params,inputs,targets)
        
        output = dag.eval(params, [inputs])
        assert jnp.allclose(output, actual(params,inputs))
        for dp,actual_dp in zip(dparams,actual_dparams):
            assert jnp.allclose(dp,actual_dp)

        params = [p-0.01*dp for p,dp in zip(params, dparams)]
    
    # Does training converge to the right values?
    final_loss = loss(dag.eval(params, [inputs]), targets)
    assert final_loss<0.02
