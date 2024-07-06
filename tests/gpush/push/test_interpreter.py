from gpush.push.interpreter import * 
from gpush.push.instructions.jax import *
import gpush.push.instructions.control as control
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.instruction import InputInstruction, ParamInstruction, ParamBuilderInstruction, LiteralInstruction, CodeBlockClose, ArrayLiteralInstruction
from gpush.push.compiler import PlushyCompiler
from gpush.push.state import PushState
from gpush.push.dag.shape import Shape, SizePlaceholder
from gpush.push.dag.dag import Dag
from gpush.train.backpropagation import BackPropagation,TBPTT, BPTT
from gpush.train.loss import MSELoss,CrossEntropyLoss
import jax.numpy as jnp
import jax.random as random 
import jax 
from jax import grad, jit, vmap
import pytest 
from gpush.utils import map_pytree
import numpy as np 

def pytree_equals(tree1,tree2):
    "Tests whether two pytrees have equal array entries"
    success = map_pytree(lambda x:True,tree1)
    return map_pytree(lambda x,y: jnp.allclose(x,y),tree1,tree2)==success 

def training_loop(fn, grad_fn, params, inputs, targets, n, lr):
    fn, actual_fn = fn 
    grad_fn, actual_grad_fn = grad_fn
    for _ in range(n):
        (loss,aux), dparams = grad_fn(params,[inputs], targets)
        print(f"iter {_} loss {loss}")
        actual_dparams = actual_grad_fn(params,inputs,targets)
        
        output,aux_ = fn(params, [inputs])
        assert jnp.allclose(output, actual_fn(params,inputs))
        assert all([jnp.allclose(a,b) for a,b in zip(aux,aux_)])
        for dp,actual_dp in zip(dparams,actual_dparams):
            assert jnp.allclose(dp,actual_dp)

        params = [p-lr*dp for p,dp in zip(params, dparams)]
    return params 

linreg_program = [InputInstruction("in1",0,"float_jax_expr"), 
                  ParamInstruction("p1",Shape(3,1),"float","float_jax_expr"), 
                  GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                  ParamInstruction("p2", Shape(), "float", "float_jax_expr"),
                  GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]

@pytest.fixture 
def actual_linreg():
    def linreg(params, input):
        [w,b] = params 
        return input@w+b 
    return linreg

@pytest.mark.slow
@pytest.mark.parametrize("program",[linreg_program])
def test_linreg(program, actual_linreg):
    """Tests whether we can use GPush to do linear regression"""
    # Test single evaluation
    
    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[]).initialize([],[{"shape":Shape(3),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    assert len(final_state["float_jax_expr"])==1
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr)
    arr1 = jnp.array([[1.],[2.],[3.]])
    arr2 = jnp.array([[1.,2.,3.],[4.,5.,6.],[5.,2.,6.]])
    arr3 = jnp.array(4)
    output, aux = dag.fn([arr1, arr3],[arr2])
    assert jnp.allclose(output, arr2@arr1+arr3)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,3)
    inputs = random.normal(subkeys[0],shape=(10,3))
    targets = jnp.reshape(inputs[:,0]+0.2*inputs[:,1]-0.5*inputs[:,2]+1.3, [-1,1])
    w,b = final_state.init_params(subkeys[1])

    # Loss / grad functions
    backprop = BackPropagation(dag.fn,MSELoss())
    def loss(output, target):
        return jnp.mean((output-target)**2)
    actual_grad = jit(grad(lambda params, input, output: loss(output, actual_linreg(params,input))))

    res, aux = dag.fn([w,b], [inputs])
    initial_loss = loss(res, targets)
    assert pytree_equals(aux,[w,b])

    # Training loop
    w,b = training_loop([dag.fn, actual_linreg], 
                        [backprop, actual_grad], 
                        [w,b], 
                        inputs, 
                        targets, 
                        1000, 
                        0.1)
    
    # Does training converge to the right values?
    res, aux = dag.fn([w,b], [inputs])
    final_loss = loss(res, targets)
    assert pytree_equals(aux,[w,b])
    assert final_loss<initial_loss
    assert final_loss<0.001
    assert jnp.allclose(w,jnp.array([[1],[0.2],[-0.5]]))
    assert jnp.allclose(b,jnp.array(1.3))


mlp_program_1 = [InputInstruction("in1",0,"float_jax_expr"), 
                 # Layer 1
                 ParamInstruction("p1",Shape(3,5),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p2", Shape(5), "float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 # Layer 2
                 ParamInstruction("p3",Shape(5,10),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p4", Shape(10), "float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 # Layer 3
                 ParamInstruction("p5",Shape(10,4),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p6", Shape(4), "float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 # Layer 4
                 ParamInstruction("p7",Shape(4,1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p8", Shape(), "float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]] 
"""Manually built (5,10,4,1) mlp, with all shapes explicitly specified"""

mlp_program_2 = [InputInstruction("in1",0,"float_jax_expr"), 
                 # Number of hidden nodes for each layer in reverse order
                 LiteralInstruction(f"1",1,"int"),
                 LiteralInstruction(f"4",4,"int"),
                 LiteralInstruction(f"10",10,"int"),
                 LiteralInstruction(f"5",5,"int"),
                 # Number of layers
                 LiteralInstruction(f"3",3,"int"),
                 # Build layers
                 GLOBAL_INSTRUCTIONS["exec_do_times"],
                 ParamBuilderInstruction("mmul_builder", Shape(SizePlaceholder(), None),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamBuilderInstruction("add_builder", Shape(SizePlaceholder()),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 CodeBlockClose(),
                 
                 # Output layer
                 ParamBuilderInstruction("mmul_builder", Shape(SizePlaceholder(), None),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamBuilderInstruction("add_builder", Shape(SizePlaceholder()),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
"Procedurally built (5,10,4,1) mlp, with only the number of hidden nodes and the number of layers specified"

@pytest.fixture
def actual_mlp():
    def mlp(params, input):
        [w1,b1, w2,b2, w3,b3, w4,b4] = params 
        x = input 
        x = nn.relu(x@w1+b1)
        x = nn.relu(x@w2+b2)
        x = nn.relu(x@w3 + b3)
        x = x@w4 + b4
        return x 
    return mlp 

@pytest.mark.slow
@pytest.mark.parametrize("program", [mlp_program_1])
def test_mlp(program, actual_mlp):
    """Tests whether we can use GPush to train a multilayer perceptron"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int = []).initialize([],[{"shape":Shape(3),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    assert len(final_state["float_jax_expr"])==1
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,3)
    inputs = random.normal(subkeys[0],shape=(10,3))
    targets = jnp.reshape(inputs[:,0]+0.2*inputs[:,1]**2-0.5*inputs[:,2]**3+1.3, [-1,1])
    
    # Initialization
    params = final_state.init_params(subkeys[1])

    # Loss / grad functions
    backprop = BackPropagation(dag.fn,MSELoss())
    def loss(output, target):
        return jnp.mean((output-target)**2)
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_mlp(params,input), output)))

    res, aux = dag.fn(params, [inputs])
    initial_loss = loss(res, targets)
    assert pytree_equals(aux,params)

    # Training loop
    params = training_loop([dag.fn, actual_mlp], 
                           [backprop, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           200, 
                           0.01)
    
    # Does training converge to the right values?
    res,aux = dag.fn(params, [inputs])
    final_loss = loss(res, targets)
    assert pytree_equals(aux,params)
    assert final_loss<initial_loss
    assert final_loss<0.5

# 3x10x10 input -> 6x6x6 -> 6x2x2 -> 24 -> 1
cnn_program_1 = [InputInstruction("in1",0,"float_jax_expr"),
                 ParamInstruction("p1",Shape(6,3,5,5),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_conv_vanilla_graph"],
                 ParamInstruction("p2",Shape(6,1,1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 ParamInstruction("p3",Shape(6,6,5,5),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_conv_vanilla_graph"],
                 ParamInstruction("p4",Shape(6,1,1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_flatten_vanilla_graph"],
                 ParamInstruction("p5",Shape(24,5),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p6",Shape(),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
"Manually built CNN, with all parameter sizes specified"

cnn_program_2 = [InputInstruction("in1",0,"float_jax_expr"),
                 # CNN Parameters in reverse 
                 LiteralInstruction(f"6",6,"int"),
                 LiteralInstruction(f"5",5,"int"),
                 LiteralInstruction(f"5",5,"int"),
                 LiteralInstruction(f"6",6,"int"),
                 LiteralInstruction(f"5",5,"int"),
                 LiteralInstruction(f"5",5,"int"),
                 # Number of CNN layers
                 LiteralInstruction(f"2",2,"int"),
                 GLOBAL_INSTRUCTIONS["exec_do_times"],
                 ParamBuilderInstruction("conv_builder", Shape(None, SizePlaceholder(), None, None),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_vanilla_graph"],
                 ParamBuilderInstruction("add_builder", Shape(SizePlaceholder(), 1,1),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 CodeBlockClose(),

                 # Flatten
                 GLOBAL_INSTRUCTIONS["float_jax_flatten_vanilla_graph"],

                 # FCN Parameters in reverse 
                 LiteralInstruction(f"5",5,"int"),
                 # Number of FCN layers
                 LiteralInstruction(f"1",1,"int"),
                 # Build layers
                 GLOBAL_INSTRUCTIONS["exec_do_times"],
                 ParamBuilderInstruction("mmul_builder", Shape(SizePlaceholder(), None),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamBuilderInstruction("add_builder", Shape(SizePlaceholder()),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
"Procedurally built CNN, with parameter sizes taken from the int stack"

@pytest.fixture
def actual_cnn():
    def cnn(params, input):
        [w1,b1, w2,b2, w3,b3] = params
        x = input 
        x = nn.relu(lax.conv_general_dilated(x,w1,(1,1),"VALID")+b1)
        x = nn.relu(lax.conv_general_dilated(x,w2,(1,1),"VALID")+b2)
        
        x = jnp.reshape(x,[x.shape[0],-1])
        x = x@w3 + b3 
        return x 
    return cnn 

@pytest.mark.slow
@pytest.mark.parametrize("program", [cnn_program_1, cnn_program_2])
def test_cnn(program, actual_cnn):
    """Tests whether we can use GPush to train a CNN"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int=[]).initialize([],[{"shape":Shape(3,10,10),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    assert len(final_state["float_jax_expr"])==1
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,8)
    inputs = random.normal(subkeys[0],shape=(10,3,10,10))
    targets = random.choice(subkeys[1], jnp.array([0,1,2,3,4]),shape=(10,))
    
    # Initialization
    params = final_state.init_params(subkeys[2])

    # Loss / grad functions
    backprop = BackPropagation(dag.fn, CrossEntropyLoss())
    def loss(output, target):
        output = nn.log_softmax(output,axis=-1)
        loss = jnp.take_along_axis(output,target[:,None],-1)
        return -jnp.mean(loss)
    
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_cnn(params,input), output)))

    res, aux = dag.fn(params, [inputs])
    initial_loss = loss(res, targets)
    assert pytree_equals(aux,params)

    # Training loop
    params = training_loop([dag.fn, actual_cnn], 
                           [backprop, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           500, 
                           0.01)
    
    # Does training converge to the right values?
    res, aux = dag.fn(params, [inputs])
    final_loss = loss(res, targets)
    assert pytree_equals(aux,params)
    assert final_loss<initial_loss
    assert final_loss<0.01

resnet_program_1 = [InputInstruction("in1",0,"float_jax_expr"),
                 
                 # Default parameters
                 LiteralInstruction("30",30,"int"),
                 GLOBAL_INSTRUCTIONS["exec_do_times"],
                 LiteralInstruction("1",1,"int"),
                 CodeBlockClose(),

                 # ResNet layer1: (3,10,10) -> (4,5,5)
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamInstruction("p1",Shape(4,3,3,3),"float","float_jax_expr"), 
                 LiteralInstruction("2",2,"int"), # stride=2
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("1",1,"int"),
                 LiteralInstruction("1",1,"int"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamInstruction("p2",Shape(4,1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_swap_graph"],
                 ParamInstruction("p3",Shape(4,3,1,1),"float","float_jax_expr"), 
                 LiteralInstruction("2",2,"int"), # stride=2
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("1",1,"int"),
                 LiteralInstruction("1",1,"int"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamInstruction("p4",Shape(4,1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],

                 # ResNet layer2: (4,5,5) -> (4,5,5)
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamInstruction("p5",Shape(4,4,3,3),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamInstruction("p6",Shape(4,1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],

                 # ResNet layer 3: (4,5,5) -> (6,3,3)
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamInstruction("p7",Shape(6,4,3,3),"float","float_jax_expr"), 
                 LiteralInstruction("2",2,"int"), # stride=2
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("1",1,"int"),
                 LiteralInstruction("1",1,"int"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamInstruction("p8",Shape(6,1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_swap_graph"],
                 ParamInstruction("p9",Shape(6,4,1,1),"float","float_jax_expr"), 
                 LiteralInstruction("2",2,"int"), # stride=2
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("1",1,"int"),
                 LiteralInstruction("1",1,"int"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamInstruction("p10",Shape(6,1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],

                 # ResNet layer4: (6,3,3) -> (6,3,3)
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamInstruction("p11",Shape(6,6,3,3),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamInstruction("p12",Shape(6,1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],

                 # Flatten
                 GLOBAL_INSTRUCTIONS["float_jax_flatten_vanilla_graph"],

                 # FCN layer 1: (54,) -> (5,)
                 ParamInstruction("p13", Shape(54, 5),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p14", Shape(5),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
"Manually built ResNet, with all parameter sizes specified"

resnet_program_2 = [InputInstruction("in1",0,"float_jax_expr"),
                 # Number of filters + stride in reverse 
                 LiteralInstruction("2",2,"int"),
                 LiteralInstruction(f"6",6,"int"),
                 LiteralInstruction("2",2,"int"),
                 LiteralInstruction(f"4",4,"int"),

                 # Number of ResNet layers
                 LiteralInstruction(f"2",2,"int"),
                 GLOBAL_INSTRUCTIONS["exec_do_times"],

                 # Strided block
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamBuilderInstruction("p1",Shape(None,SizePlaceholder(),3,3),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["int_dup"],
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamBuilderInstruction("p2",Shape(SizePlaceholder(),1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_swap_graph"],
                 ParamBuilderInstruction("p3",Shape(SizePlaceholder(),SizePlaceholder(),1,1),"float","float_jax_expr"), 
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"),
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamBuilderInstruction("p4",Shape(SizePlaceholder(),1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],

                 # Number of unstrided blocks per layer
                 LiteralInstruction("1",1,"int"),
                 GLOBAL_INSTRUCTIONS["exec_do_times"],
                 # unstrided block

                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamBuilderInstruction("p5",Shape(SizePlaceholder(),SizePlaceholder(),3,3),"float","float_jax_expr"),
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"),
                 LiteralInstruction("-1",-1,"int"), 
                 GLOBAL_INSTRUCTIONS["float_jax_conv_graph"],
                 ParamBuilderInstruction("p6",Shape(SizePlaceholder(),1,1),"float","float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_relu_graph"],
                 CodeBlockClose(),

                 CodeBlockClose(),

                 # Flatten
                 GLOBAL_INSTRUCTIONS["float_jax_flatten_vanilla_graph"],

                 # FCN Parameters in reverse 
                 LiteralInstruction(f"5",5,"int"),
                 # Number of FCN layers
                 LiteralInstruction(f"1",1,"int"),
                 # Build layers
                 GLOBAL_INSTRUCTIONS["exec_do_times"],
                 ParamBuilderInstruction("mmul_builder", Shape(SizePlaceholder(), None),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamBuilderInstruction("add_builder", Shape(SizePlaceholder()),"float", "float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]]
"Procedurally built ResNet, with parameter sizes taken from the int stack"

@pytest.fixture
def actual_resnet():
    def resnet(params, input):
        [w00,b00,w01,b01,w1,b1,w20,b20,w21,b21,w3,b3,w4,b4] = params
        x = input 
        temp = x 
        x = lax.conv_general_dilated(x,w00,(2,2),"SAME")+b00
        temp = lax.conv_general_dilated(temp,w01,(2,2),"SAME")+b01
        x = nn.relu(x+temp)

        temp = x
        x = lax.conv_general_dilated(x,w1,(1,1),"SAME")+b1
        x = nn.relu(x+temp)

        temp = x
        x = lax.conv_general_dilated(x,w20,(2,2),"SAME")+b20
        temp = lax.conv_general_dilated(temp,w21,(2,2),"SAME")+b21
        x = nn.relu(x+temp)

        temp = x
        x = lax.conv_general_dilated(x,w3,(1,1),"SAME")+b3
        x = nn.relu(x+temp)
        
        x = jnp.reshape(x,[x.shape[0],-1])
        x = x@w4 + b4 
        return x 
    return resnet

@pytest.mark.slow
@pytest.mark.parametrize("program", [resnet_program_1, resnet_program_2])
def test_resnet(program, actual_resnet):
    """Tests whether we can use GPush to train a CNN"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int=[]).initialize([],[{"shape":Shape(3,10,10),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    assert len(final_state["float_jax_expr"])==1
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,8)
    inputs = random.normal(subkeys[0],shape=(10,3,10,10))
    targets = random.choice(subkeys[1], jnp.array([0,1,2,3,4]),shape=(10,))
    
    # Initialization
    params = final_state.init_params(subkeys[2])

    # Loss / grad functions
    backprop = BackPropagation(dag.fn, CrossEntropyLoss())
    def loss(output, target):
        output = nn.log_softmax(output,axis=-1)
        loss = jnp.take_along_axis(output,target[:,None],-1)
        return -jnp.mean(loss)
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_resnet(params,input), output)))

    res, aux = dag.fn(params, [inputs])
    initial_loss = loss(res, targets)
    assert pytree_equals(aux,params)

    # Training loop
    params = training_loop([dag.fn, actual_resnet], 
                           [backprop, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           500, 
                           0.01)
    
    # Does training converge to the right values?
    res, aux = dag.fn(params, [inputs])
    final_loss = loss(res, targets)
    assert pytree_equals(aux,params)
    assert final_loss<initial_loss
    assert final_loss<0.01


rnn_program_1 = [ParamInstruction("p1",Shape(100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 ParamInstruction("p2",Shape(100,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p1",Shape(100),"float","float_jax_expr"), 
                 InputInstruction("in1",0,"float_jax_expr"), 
                 ParamInstruction("p1",Shape(1,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_tanh_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_param_update"],
                 ParamInstruction("p1",Shape(100,1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("p1",Shape(1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]] 

@pytest.fixture
def actual_rnn():
    def step(params,input):
        [h0,w_recur,b_recur,w_in,w_out,b_out] = params
        # Order of execution needs to be the exact same as in the program, or rounding errors will accumulate
        r1 = h0@w_recur 
        r2 = input@w_in 
        r3 = b_recur+r2 
        h0 = r1+r3
        h0 = nn.tanh(h0)
        output = h0@w_out +b_out 
        return [h0] + params[1:], output
    def rnn(params,input):
        batch_size = input.shape[0]
        params = map_pytree(lambda x:lax.broadcast(x,(batch_size,)),params)
        input = jnp.swapaxes(input,0,1)
        carry,output = lax.scan(vmap(step,in_axes=[0,0]),params,xs=input)
        # print(output)
        return jnp.swapaxes(output,0,1)
    return jit(rnn)

@pytest.mark.slow
@pytest.mark.parametrize("program", [rnn_program_1])
def test_rnn(program, actual_rnn):
    """Tests whether we can use GPush to train an RNN"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int=[]).initialize([],[{"shape":Shape(1,),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    assert len(final_state["float_jax_expr"])==1
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr, recurrent = True)

    hidden_params = np.zeros((len(final_state.params),),dtype="int")
    hidden_indices = dag.root.hidden_params()
    for i in hidden_indices:
        hidden_params[i]=1

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,8)
    inputs = random.normal(subkeys[0],shape=(2,25,1))*0.025+0.01
    targets = jnp.cumsum(inputs,axis=1)
    
    # Initialization
    params = final_state.init_params(subkeys[1])

    # Loss / grad functions
    backprop = BPTT(dag.fn, MSELoss(), hidden_params)
    def loss(output, target):
        return jnp.mean((output-target)**2)
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_rnn(params,input), output)))

    res, aux = backprop.val(params, [inputs])
    res_ = actual_rnn(params,inputs)
    assert jnp.allclose(res,res_)
    initial_loss = loss(res, targets)
    assert pytree_equals(aux[1:],params[1:])

    # Training loop
    params = training_loop([backprop.val, actual_rnn], 
                           [backprop, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           200, 
                           0.01)
    
    # Does training converge to the right values?
    res, aux = backprop.val(params, [inputs])
    final_loss = loss(res, targets)
    assert pytree_equals(aux[1:],params[1:])
    assert final_loss<initial_loss
    assert final_loss<0.005

gru_program_1 = [ParamInstruction("h0",Shape(100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"], # 4x h0
                 ParamInstruction("uz",Shape(100,100),"float","float_jax_expr"), # h0,h0,h0,h0,uz
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"], # h0,h0,h0,h0@uz
                 ParamInstruction("bz",Shape(100),"float","float_jax_expr"),  # h0,h0,h0,h0@uz,bz
                 InputInstruction("in1",0,"float_jax_expr"), 
                 ParamInstruction("wz",Shape(1,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"], # h0,h0,h0,h0@uz,bz, input@wz
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"], # h0,h0,h0,h0@uz,bz + input@wz
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"], # h0,h0,h0,h0@uz + (bz + input@wz)
                 GLOBAL_INSTRUCTIONS["float_jax_sigmoid_graph"], # This makes h0,h0,h0,z
                 ArrayLiteralInstruction("1",jnp.array(1.),"float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_swap_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_sub_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"],# dup 
                 ArrayLiteralInstruction("1",jnp.array(1.),"float_jax_expr"),
                 GLOBAL_INSTRUCTIONS["float_jax_swap_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_sub_graph"],# 1-z = h0,h0,h0,1-z,z
                 GLOBAL_INSTRUCTIONS["float_jax_rot_graph"],# rot = h0,h0,z,h0,1-z
                 GLOBAL_INSTRUCTIONS["float_jax_mul_graph"],# = h0,h0,z,1-z(h0)
                 GLOBAL_INSTRUCTIONS["float_jax_rot_graph"],# rot = h0,h0(1-z),h0,z
                 GLOBAL_INSTRUCTIONS["float_jax_swap_graph"], # h0,h0(1-z),z,h0
                 GLOBAL_INSTRUCTIONS["float_jax_dup_graph"], # h0,h0(1-z),z,h0,h0
                 ParamInstruction("ur",Shape(100,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("br",Shape(100),"float","float_jax_expr"), 
                 InputInstruction("in1",0,"float_jax_expr"), 
                 ParamInstruction("wr",Shape(1,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_sigmoid_graph"], # This makes h0, h0(1-z),z,h0,r
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"], # h0,h0(1-z),z,r+h0
                 ParamInstruction("uh",Shape(100,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("bh",Shape(100),"float","float_jax_expr"), 
                 InputInstruction("in1",0,"float_jax_expr"), 
                 ParamInstruction("wh",Shape(1,100),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],
                 GLOBAL_INSTRUCTIONS["float_jax_tanh_graph"], # This makes h0,h0(1-z),z,hhat
                 GLOBAL_INSTRUCTIONS["float_jax_mul_graph"], # h0,hhat,z
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"],

                 GLOBAL_INSTRUCTIONS["float_jax_param_update"],
                 ParamInstruction("w_out",Shape(100,1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_mmul_graph"],
                 ParamInstruction("b_out",Shape(1),"float","float_jax_expr"), 
                 GLOBAL_INSTRUCTIONS["float_jax_add_graph"]] 

@pytest.fixture
def actual_gru():
    def step(params,input):
        [h0,uz,bz,wz,ur,br,wr,uh,bh,wh,w_out,b_out] = params
        # Order of execution needs to be the exact same as in the program, or rounding errors will accumulate
        z1 = h0@uz 
        z2 = input@wz 
        z3 = bz+z2 
        z4 = z1+z3 
        z = nn.sigmoid(z4)
        r1 = h0@ur
        r2 = input@wr 
        r3 = br+r2 
        r4 = r1+r3 
        r = nn.sigmoid(r4)
        h1 = h0+r 
        h2 = h1@uh 
        h3 = input@wh 
        h4 = bh+h3 
        h5 = h2 + h4
        hhat = nn.tanh(h5)
        
        h7 = jnp.array(1.)-z
        z_ = jnp.array(1.)-h7
        h6 = z_*hhat 
        h8 = h0*h7 
        h0 = h8 + h6
        o1 = h0@w_out 
        output = o1 +b_out 
        return [h0] + params[1:], output
    def gru(params,input):
        batch_size = input.shape[0]
        params = map_pytree(lambda x:lax.broadcast(x,(batch_size,)),params)
        input = jnp.swapaxes(input,0,1)
        carry,output = lax.scan(vmap(step,in_axes=[0,0]),params,xs=input)
        # print(output)
        return jnp.swapaxes(output,0,1)
    return jit(gru)

@pytest.mark.slow
@pytest.mark.parametrize("program", [gru_program_1])
def test_gru(program, actual_gru):
    """Tests whether we can use GPush to train an RNN"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int=[]).initialize([],[{"shape":Shape(1,),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    assert len(final_state["float_jax_expr"])==1
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr, recurrent = True)

    hidden_params = np.zeros((len(final_state.params),),dtype="int")
    hidden_indices = dag.root.hidden_params()
    for i in hidden_indices:
        hidden_params[i]=1

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,8)
    inputs = random.normal(subkeys[0],shape=(2,25,1))*0.025+0.01
    targets = jnp.cumsum(inputs,axis=1)
    
    # Initialization
    params = final_state.init_params(subkeys[1])

    # Loss / grad functions
    backprop = BPTT(dag.fn, MSELoss(), hidden_params)
    def loss(output, target):
        return jnp.mean((output-target)**2)
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_gru(params,input), output)))

    res, aux = backprop.val(params, [inputs])
    res_ = actual_gru(params,inputs)
    assert jnp.allclose(res,res_)
    initial_loss = loss(res, targets)
    assert pytree_equals(aux[1:],params[1:])

    # Training loop
    params = training_loop([backprop.val, actual_gru], 
                           [backprop, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           200, 
                           0.01)
    
    # Does training converge to the right values?
    res, aux = backprop.val(params, [inputs])
    final_loss = loss(res, targets)
    assert pytree_equals(aux[1:],params[1:])
    assert final_loss<initial_loss
    assert final_loss<0.005

# attention_program_1 = []