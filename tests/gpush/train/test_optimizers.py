from gpush.train.optimizers import *
import pytest
from dataclasses import dataclass, field
from typing import Callable
import numpy as np  
import jax.numpy as jnp 
import jax 
from jax import jit

@pytest.mark.parametrize("val", [0, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1])
def test_constant_schedule(val):
    s = ConstantSchedule(val)
    assert s.init_val==val
    assert s.step(1,10)==val

@pytest.mark.parametrize("args", [(1,2,3), (None, "hi",34.1), (1,), tuple(), (tuple(),(1,2),{"this": 0})])
@pytest.mark.parametrize("kwargs", [{}, {"this":"is", "not":"a", "test":"."}, {"one":2, "":4}])
def test_lambda_schedule(args, kwargs):
    def identity(i, cur_val, *args, **kwargs):
        return (i,cur_val,args,kwargs)
    s = LambdaSchedule(0,identity,args,kwargs)
    assert s.init_val==0
    assert s.step(1,10)==(1,10,args,kwargs)

@pytest.mark.parametrize("val", [0, 1E-5, 1E-4, 1E-3, 1E-2, 0.1, 1])
@pytest.mark.parametrize("max_steps", [0, 1, 10, 25, 4392])
@pytest.mark.parametrize("step", [-1, 0, 1, 9, 11, 1249, 10492])
@pytest.mark.parametrize("cur_val", [None,1,0.1])
def test_cosine_schedule(val,max_steps,step, cur_val):
    if max_steps>0:
        s = CosineSchedule(val,max_steps)
        if step<=max_steps and step>=0:
            assert np.allclose(float(s.step(step,cur_val)), float(np.cos(step*np.pi/(max_steps*2))*val), atol=1E-5)


@pytest.mark.parametrize("steps",[((0,0.1),), ((0,0.1),(10,0.01)), ((10,0.01),(20,0.02),(100,0.001)), tuple(), ((10000,5),(20000,10))])
@pytest.mark.parametrize("step",[-1,0,1,6,13,26,104,15024])
@pytest.mark.parametrize("cur_val", [None,1,0.1])
def test_stepwise_linear_schedule(steps, step, cur_val):
    if len(steps)<1:
        with pytest.raises(Exception):
            s = StepwiseLinearSchedule(0,steps)
        return
    s = StepwiseLinearSchedule(0 if steps[0][0]>0 else steps[0][1],steps)
    steps = ((0,0),)+steps if steps[0][0]>0 else steps
    if step>=0:
        if len(steps)==0:
            res = cur_val
        elif step<steps[0][0]:
            res = steps[0][1]
        elif step>=steps[-1][0]:
            res = steps[-1][1]
        else:
            idx = min([i for i,(idx,val) in enumerate(steps) if idx>step])
            res = steps[idx-1][1]+(steps[idx][1]-steps[idx-1][1])*(step-steps[idx-1][0])/(steps[idx][0]-steps[idx-1][0])
        assert jnp.allclose(res,s.step(step,cur_val))

@pytest.mark.parametrize("params,grad",[([jnp.zeros((1,)), jnp.ones((2,3)), jnp.array([[1000.1,2.2,3],[4,1,2.4],[1,205.2,2]])], [jnp.ones((1,)), jnp.zeros((2,3)), jnp.array([[0.4,1.3,2.4],[4,1.2,4],[100.5,-204.1,0.00001]])]), 
                                        ([jnp.array(10.)], [jnp.array(-4.1)]), 
                                        ([],[])])
@pytest.mark.parametrize("lr", [ConstantSchedule(1.),StepwiseLinearSchedule(1.,((0,1.),(100,0.001))), CosineSchedule(0.1,100), LambdaSchedule(1.,lambda i,cur_val,arg1,kwarg1=0:cur_val*i+(arg1/kwarg1), (10.,),{"kwarg1":-9.})])
@pytest.mark.parametrize("step", [0,1,10,100])
def test_sgd(params, grad, lr, step):
    res = [p-lr.step(step,lr.init_val)*g for p,g in zip(params,grad)]
    sgd = SGD(lr)
    state = sgd.init(params)
    state2 = sgd.update(step, grad, state)
    res_params = sgd.get_params(state2)

    assert all([jnp.allclose(rp,r) for rp,r in zip(res_params,res)])
    assert state == {"params":params, "lr":lr.init_val} 
    assert isinstance(state["lr"],float) or jnp.ndim(state["lr"])==0
    assert state2["lr"] == lr.step(step,lr.init_val)
    assert isinstance(state2["lr"],float) or jnp.ndim(state2["lr"])==0
    assert sgd.get_params(state2)==state2["params"]

def linreg():
    def sigmoid(x):
        return 0.5 * (jnp.tanh(x / 2) + 1)
    def predict(W, b, inputs):
        return sigmoid(jnp.dot(inputs, W) + b)
    inputs = jnp.array([[0.52, 1.12,  0.77],
                    [0.88, -1.08, 0.15],
                    [0.52, 0.06, -1.30],
                    [0.74, -2.49, 1.39]])
    targets = jnp.array([True, True, False, True])
    def loss(W, b):
        preds = predict(W, b, inputs)
        label_probs = preds * targets + (1 - preds) * (1 - targets)
        return -jnp.sum(jnp.log(label_probs))
    return loss


@pytest.mark.parametrize("params, loss_tolerance, loss_fn, tolerance, res",[([jnp.ones((1,))], 0.001, lambda x:jnp.mean(jnp.array([(p**2).mean() for p in x])), 0.1,[jnp.zeros((1,))]), 
                                                 ([jnp.array([2.,0.1,1.4, 0.001, -1.2])], 0.1, lambda x:jnp.mean(jnp.array([((p-1.3)**2).mean() for p in x])), 0.5,[jnp.zeros((5,))+1.3]), 
                                                 ([jnp.zeros((3,)), jnp.zeros(tuple())], 0.3, lambda x: linreg()(*x), None, None)])
@pytest.mark.parametrize("lr", [ConstantSchedule(0.1),StepwiseLinearSchedule(0.1,((0,0.1),(100,0))), CosineSchedule(0.1,100), LambdaSchedule(0.1,lambda i,cur_val,arg1,kwarg1=0:jnp.where(i==0,0.1,jnp.where(i==arg1,cur_val/kwarg1,cur_val)), (40,),{"kwarg1":10})])
@pytest.mark.parametrize("opt_factory", [Adam,SGD,Momentum])
def test_convergence(params, loss_tolerance, loss_fn, tolerance, res, lr,opt_factory):
    grad_fn = jax.grad(loss_fn)
    opt = opt_factory(lr)

    @jit
    def update(i,cur_state):
        params = opt.get_params(cur_state)
        grad = grad_fn(params)
        return opt.update(i,grad,cur_state)

    state = opt.init(params)
    for i in range(100):
        state = update(i,state)
        print(loss_fn(state["params"]))
        print(state)
        if isinstance(lr,LambdaSchedule):
            assert jnp.allclose(state["lr"], (0.1 if i<40 else 0.01))
        elif isinstance(lr,ConstantSchedule):
            assert jnp.allclose(state["lr"],0.1)
        elif isinstance(lr,StepwiseLinearSchedule):
            assert jnp.allclose(state["lr"],0.1-i/1000)
        else:
            assert jnp.allclose(state["lr"],jnp.cos(i*jnp.pi/200)*0.1)
    
    assert loss_fn(opt.get_params(state))<loss_tolerance
    if res is not None:
        assert all([jnp.allclose(r,p,atol=tolerance) for r,p in zip(res,opt.get_params(state))])
    

@pytest.mark.parametrize("params,velocity,grad",[([jnp.zeros((1,)), jnp.ones((2,3)), jnp.array([[1000.1,2.2,3],[4,1,2.4],[1,205.2,2]])], [jnp.ones((1,)), jnp.zeros((2,3)), jnp.array([[-20.1,22.2,203],[-4,1.001,-2.4],[1.91,-25.2,-4]])], [jnp.zeros((1,)), jnp.ones((2,3)), jnp.array([[0.4,1.3,2.4],[4,1.2,4],[100.5,-204.1,0.00001]])]), 
                                        ([jnp.array(10.)],[jnp.array(2.2)], [jnp.array(-4.1)]), 
                                        ([],[],[])])
@pytest.mark.parametrize("lr", [ConstantSchedule(1.),StepwiseLinearSchedule(1.,((0,1.),(100,0.001))), CosineSchedule(0.1,100), LambdaSchedule(1.,lambda i,cur_val,arg1,kwarg1=0:cur_val*i+(arg1/kwarg1), (10.,),{"kwarg1":-9.})])
@pytest.mark.parametrize("mass", [ConstantSchedule(0.9),StepwiseLinearSchedule(0.99,((0,0.99),(100,0.5)))])
@pytest.mark.parametrize("step", [0,1,10,100])
def test_momentum(params, velocity, grad, lr, mass, step):
    vel = [v*mass.step(step,mass.init_val)+g for v,g in zip(velocity,grad)]
    res = [x-lr.step(step,lr.init_val)*v for x,v in zip(params,vel)]
    opt = Momentum(lr,mass)
    state = opt.init(params)
    assert all([jnp.allclose(m,0) for m in state["momentum"]])
    state["momentum"]=velocity
    state2 = opt.update(step, grad, state)
    res_params = opt.get_params(state2)

    assert all([jnp.allclose(rp,r) for rp,r in zip(res_params,res)])
    assert state == {"params":params, "lr":lr.init_val, "momentum":velocity, "mass": mass.init_val} 
    assert isinstance(state["lr"],float) or jnp.ndim(state["lr"])==0
    assert isinstance(state["mass"],float) or jnp.ndim(state["mass"])==0
    assert state2["lr"] == lr.step(step,lr.init_val)
    assert state2["mass"] == mass.step(step,mass.init_val)
    assert isinstance(state2["lr"],float) or jnp.ndim(state2["lr"])==0
    assert isinstance(state2["mass"],float) or jnp.ndim(state2["mass"])==0
    assert opt.get_params(state2)==state2["params"]

@pytest.mark.parametrize("params,velocity,var,grad",[([jnp.zeros((1,)), jnp.ones((2,3)), jnp.array([[1000.1,2.2,3],[4,1,2.4],[1,205.2,2]])], [jnp.ones((1,)), jnp.zeros((2,3)), jnp.array([[-20.1,22.2,203],[-4,1.001,-2.4],[1.91,-25.2,-4]])], [jnp.zeros((1,)), jnp.ones((2,3)), jnp.array([[1.4,10.2,2.03],[4,20.001,2.4],[19.1,2.223,10.2]])], [jnp.zeros((1,)), jnp.ones((2,3)), jnp.array([[0.4,1.3,2.4],[4,1.2,4],[100.5,-204.1,0.00001]])]), 
                                        ([jnp.array(10.)],[jnp.array(2.2)],[jnp.array(14.4)], [jnp.array(-4.1)]), 
                                        ([],[],[],[])])
@pytest.mark.parametrize("lr", [ConstantSchedule(1.),StepwiseLinearSchedule(1.,((0,1.),(100,0.001))), CosineSchedule(0.1,100), LambdaSchedule(1.,lambda i,cur_val,arg1,kwarg1=0:cur_val*i+(arg1/kwarg1), (10.,),{"kwarg1":-9.})])
@pytest.mark.parametrize("b1", [ConstantSchedule(0.9),StepwiseLinearSchedule(0.99,((0,0.99),(100,0.5)))])
@pytest.mark.parametrize("b2", [ConstantSchedule(0.99),StepwiseLinearSchedule(0.999,((0,0.999),(100,0.9)))])
@pytest.mark.parametrize("eps", [ConstantSchedule(1E-4),ConstantSchedule(1E-8)])
@pytest.mark.parametrize("step", [0,1,10,100])
def test_adam(params, velocity,var,grad, lr, b1,b2,eps,step):
    # Compute actual result
    b1_ = b1.step(step,b1.init_val)
    b2_ = b2.step(step,b2.init_val)
    lr_ = lr.step(step,lr.init_val)
    eps_ = eps.step(step,eps.init_val)

    velocity_ = [((1-b1_)*g+b1_*v)for g,v in zip(grad,velocity)]
    var_ = [((1-b2_)*jnp.square(g)) + b2_*v for g,v in zip(grad,var)]
    mhat_ = [m/(1-jnp.asarray(b1_)**(step+1)) for m in velocity_]
    vhat_ = [v/(1-jnp.asarray(b2_)**(step+1)) for v in var_]
    res = [x-lr_*v/(jnp.sqrt(v_)+eps_) for x,v,v_ in zip(params,mhat_,vhat_)]

    state2_actual = {"params":res, "velocity":velocity_,"var":var_,"lr":lr_,"b1":b1_,"b2":b2_,"eps":eps_}

    # Compute result
    opt = Adam(lr,b1,b2,eps)
    # Test initialization
    state = opt.init(params)
    assert all([jnp.allclose(m,0) for m in state["velocity"]])
    assert all([jnp.allclose(m,0) for m in state["var"]])
    state["velocity"]=velocity
    state["var"]=var 
    state2 = opt.update(step, grad, state)
    res_params = opt.get_params(state2)

    # Test immutability
    assert state == {"params":params, "velocity":velocity, "var":var, "lr":lr.init_val, "b1":b1.init_val,"b2":b2.init_val,"eps":eps.init_val} 
    # Test hyperparameters shapes
    for k in ["lr", "b1","b2","eps"]:
        assert isinstance(state[k],float) or jnp.ndim(state[k])==0
    hyperparams = {"lr":lr,"b1":b1,"b2":b2,"eps":eps}
    for k in ["lr", "b1","b2","eps"]:
        assert isinstance(state2[k],float) or jnp.ndim(state2[k])==0
        assert state2[k] == hyperparams[k].step(step,hyperparams[k].init_val)
    # Test correctness
    for k in state2:
        if not isinstance(state2[k],list) and (isinstance(state2[k],float) or jnp.ndim(state[k])==0):
            assert state2[k]==state2_actual[k]
        else:
            assert all([jnp.allclose(r,r_) for r,r_ in zip(state2[k],state2_actual[k])])
    assert all([jnp.allclose(r,rp) for r,rp in zip(res,res_params)])
    # Test get_params
    assert opt.get_params(state2)==state2["params"]

