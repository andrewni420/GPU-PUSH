from gpush.push.interpreter import * 
from gpush.push.instructions.jax import *
import gpush.push.instructions.control as control
from gpush.push.instruction_set import GLOBAL_INSTRUCTIONS
from gpush.push.instruction import InputInstruction, ParamInstruction, ParamBuilderInstruction, LiteralInstruction, CodeBlockClose
from gpush.push.compiler import PlushyCompiler
from gpush.push.state import PushState
from gpush.push.dag.shape import Shape, SizePlaceholder
from gpush.push.dag.dag import Dag
import jax.numpy as jnp
import jax.random as random 
import jax 
from jax import grad, jit
import pytest 

def training_loop(fn, grad_fn, params, inputs, targets, n, lr):
    fn, actual_fn = fn 
    grad_fn, actual_grad_fn = grad_fn
    for _ in range(n):
        dparams = grad_fn(params,[inputs], targets)
        actual_dparams = actual_grad_fn(params,inputs,targets)
        
        output = fn(params, [inputs])
        assert jnp.allclose(output, actual_fn(params,inputs))
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

@pytest.mark.parametrize("program",[linreg_program])
def test_linreg(program, actual_linreg):
    """Tests whether we can use GPush to do linear regression"""
    # Test single evaluation
    
    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[]).initialize([],[{"shape":Shape(3),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
    expr = final_state["float_jax_expr"][0]
    dag = Dag(expr)
    arr1 = jnp.array([[1.],[2.],[3.]])
    arr2 = jnp.array([[1.,2.,3.],[4.,5.,6.],[5.,2.,6.]])
    arr3 = jnp.array(4)
    output = dag.eval([arr1, arr3],[arr2])
    assert jnp.allclose(output, arr2@arr1+arr3)

    # Test learning
    key = random.key(152)
    key, *subkeys = random.split(key,3)
    inputs = random.normal(subkeys[0],shape=(10,3))
    targets = jnp.reshape(inputs[:,0]+0.2*inputs[:,1]-0.5*inputs[:,2]+1.3, [-1,1])
    w,b = final_state.init_params(subkeys[1])

    # Loss / grad functions
    def loss(output, target):
        return jnp.mean((output-target)**2)
    grad_fn = dag.grad(loss)
    
    
    actual_grad = jit(grad(lambda params, input, output: loss(output, actual_linreg(params,input))))

    # Training loop
    w,b = training_loop([dag.eval, actual_linreg], 
                        [grad_fn, actual_grad], 
                        [w,b], 
                        inputs, 
                        targets, 
                        1000, 
                        0.1)
    
    # Does training converge to the right values?
    final_loss = loss(dag.eval([w,b], [inputs]), targets)
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

@pytest.mark.parametrize("program", [mlp_program_1])
def test_mlp(program, actual_mlp):
    """Tests whether we can use GPush to train a multilayer perceptron"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int = []).initialize([],[{"shape":Shape(3),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
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
    def loss(output, target):
        return jnp.mean((output-target)**2)
    grad_fn = dag.grad(loss)
    
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_mlp(params,input), output)))

    # Training loop
    params = training_loop([dag.fn, actual_mlp], 
                           [grad_fn, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           200, 
                           0.01)
    
    # Does training converge to the right values?
    final_loss = loss(dag.fn(params, [inputs]), targets)
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

@pytest.mark.parametrize("program", [cnn_program_1, cnn_program_2])
def test_cnn(program, actual_cnn):
    """Tests whether we can use GPush to train a CNN"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int=[]).initialize([],[{"shape":Shape(3,10,10),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
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
    def loss(output, target):
        output = nn.log_softmax(output,axis=-1)
        loss = jnp.take_along_axis(output,target[:,None],-1)
        return -jnp.mean(loss)
    grad_fn = dag.grad(loss)
    
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_cnn(params,input), output)))

    initial_loss = loss(dag.fn(params, [inputs]), targets)

    # Training loop
    params = training_loop([dag.fn, actual_cnn], 
                           [grad_fn, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           500, 
                           0.01)
    
    # Does training converge to the right values?
    final_loss = loss(dag.fn(params, [inputs]), targets)
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

@pytest.mark.parametrize("program", [resnet_program_1, resnet_program_2])
def test_resnet(program, actual_resnet):
    """Tests whether we can use GPush to train a CNN"""
    # Test single evaluation

    program = PlushyCompiler()(program)
    start_state = PushState(float_jax_expr=[], exec=[], int=[]).initialize([],[{"shape":Shape(3,10,10),"dtype":"float"}])
    final_state = Interpreter().run(program, state=start_state)
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
    def loss(output, target):
        output = nn.log_softmax(output,axis=-1)
        loss = jnp.take_along_axis(output,target[:,None],-1)
        return -jnp.mean(loss)
    grad_fn = dag.grad(loss)
    
    
    actual_grad = jit(grad(lambda params, input, output: loss(actual_resnet(params,input), output)))

    initial_loss = loss(dag.fn(params, [inputs]), targets)

    # Training loop
    params = training_loop([dag.fn, actual_resnet], 
                           [grad_fn, actual_grad], 
                           params, 
                           inputs, 
                           targets, 
                           500, 
                           0.01)
    
    # Does training converge to the right values?
    final_loss = loss(dag.fn(params, [inputs]), targets)
    assert final_loss<0.01

# attention_program_1 = []