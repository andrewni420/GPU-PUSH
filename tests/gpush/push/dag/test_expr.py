from gpush.push.dag.expr import Function, Parameter, Input
import gpush.push.dag.expr as expr 
from gpush.push.dag.shape import Shape 

def identity(*args, **kwargs):
    "Identity function returns its args/kwargs unchanged"
    return args,kwargs

def make_tree():
    """Builds an example tree to test on. Returns the tree root, the constants, the inputs, the functions, and the invisible functions that
    should not appear in the tree. In addition, c2, c3, i1, and i3 should also not appear in the tree.\n
    Also builds the expected outputs from evaluating the tree, and returns them as well.\n
    Returns (`structure`, `output`) where \n
        `structure` is (`root`, `constants`, `inputs`, `functions`, `invisible functions`)\n
        `output` is (`params`, `input`, `tree outputs`) where\n
        `tree outputs` is (`root output`, `constants output`, `inputs output`, `functions output`, `invisible functions output`)"""
    # Leaves
    p0,p1,p2,p3,p4 = parameters = [Parameter(i,param_idx=i) for i in range(5)]
    i0,i1,i2,i3,i4 = inputs =  [Input(i+5, input_idx=i) for i in range(5)]

    # Functions of leaves
    f0 = Function(id = 10, fn = identity, children = (p0, p4, i2))
    f0_ = Function(id = 12, fn = identity, children = (p0, p4, i2))
    f1 = Function(id = 14, fn = identity, children = tuple())
    f1_ = Function(id = 15, fn = identity, children = dict())
    f2 = Function(id = 17, fn = identity, children = (p1,))
    f3 = Function(id = 19, fn = identity, children = (p0, i0))
    f4 = Function(id = 24, fn = identity, children = (i0, i2))
    f4_ = Function(id = 25, fn = identity, children = (i0, i2))

    # Functions of functions
    f5 = Function(id = 28, fn = identity, children = (f0, i0))
    f6 = Function(id = 32, fn = identity, children = (f4, i2))
    f7_ = Function(id = 34, fn = identity, children = (f4, f0, f3))
    f7 = Function(id = 35, fn = identity, children = {"arg0": [f4, f0], "arg1": [f3], "arg2": []})
    f8 = Function(id = 37, fn = identity, children = (i4, f7))
    f9 = Function(id = 40, fn = identity, children = (f8,))
    f10 = Function(id = 50, fn = identity, children = (f1,p1,i4))

    # Root node
    root = Function(id = 100, fn = identity, children = {"arg0": [f10], "arg1": [f2, i0, f9], "arg2": [f6, f5]})

    functions = [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,root]
    _functions = {0:f0_,1: f1_, 4:f4_,7:f7_}

    # Expected outputs 
    params = [1,-4,50,2,3]
    input = [9,34,0,1,-34] 

    # Constant outputs
    op0,op1,op2,op3,op4 = oparameters = params 
    oi0,oi1,oi2,oi3,oi4 = oinputs = input 

    # Function outputs
    of0 = ((op0,op4,oi2), dict())
    of0_ = ((op0,op4,oi2), dict())
    of1 = (tuple(), dict())
    of1_ = (tuple(), dict())
    of2 = ((op1,), dict())
    of3 = ((op0, oi0), dict())
    of4 = ((oi0, oi2), dict())
    of4_ = ((oi0, oi2), dict())

    # Functions of functions outputs
    of5 = ((of0, oi0), dict())
    of6 = ((of4, oi2), dict())
    of7 = (tuple(), {"arg0": [of4, of0], "arg1": [of3], "arg2": []})
    of7_ = ((of4, of0, of3), dict())
    of8 = ((oi4, of7), dict())
    of9 = ((of8,), dict())
    of10 = ((of1, op1, oi4), dict())

    # Root output
    oroot = (tuple(), {"arg0": [of10], "arg1": [of2, oi0, of9], "arg2": [of6, of5]})

    ofunctions = [of0,of1,of2,of3,of4,of5,of6,of7,of8,of9,of10,oroot]
    _ofunctions = {0:of0_,1: of1_, 4:of4_,7:of7_}

    return (root, parameters, inputs, functions, _functions), (params, input, (oroot, oparameters, oinputs, ofunctions, _ofunctions))

def test_gather():
    "Test whether `gather()` correctly returns the nodes in the tree"
    (tree, parameters, inputs, functions, _functions), _ = make_tree()
    expressions = tree.gather()
    target_expressions = [parameters[0], parameters[1], parameters[4], inputs[0], inputs[2], inputs[4]] + functions 
    assert expressions==target_expressions

    expr_ids = tree.gather(mapper = lambda x:x.id*2+10)
    assert expr_ids == [(e.id*2+10) for e in target_expressions]

    expr_9 = tree.gather(reducer = lambda x:x[9])
    assert expr_9 == target_expressions[9] 

    map_reduce = tree.gather(mapper = lambda x:f"{x.dtype} {x.id}", reducer=lambda x:f"({x[-3]}) ({x[4]})")
    assert map_reduce == "(float 40) (float 7)"

def test_map():
    "Test whether `map_dfs()` and `update_ids` perform the correct operations"
    (tree, parameters, inputs, functions, _functions), _ = make_tree()
    expressions = tree.gather()
    target_ids = [(e.id*2+10) for e in expressions]
    def update(x):
        x.foo = x.id*2+10
    tree.map_dfs(update)
    assert [e.foo for e in tree.gather()]==target_ids

    def update(x, idx=0):
        x._id[idx] = x._id[idx]*2+10 
    tree.update_ids(update)
    assert [e.id for e in tree.gather()]==target_ids

def test_normalize():
    "Test whether `normalize()` correctly updates the tree to have `id`s from 1 to len(tree)"
    (tree, parameters, inputs, functions, _functions), _ = make_tree()
    expressions = tree.normalize()
    assert [e.id for e in expressions]==list(range(len(expressions)))
    expressions_ = tree.gather()
    assert [e.id for e in expressions_]==list(range(len(expressions)))
    assert expressions_ == expressions

def test_initial_eval():
    "Test whether evaluating expressions with an empty cache gives the expected results"

    (tree, parameters, inputs, functions, _functions), (params, input, _) = make_tree()
    cache = [None]*len(tree.gather())
    tree.normalize()

    # Test constants
    for p,p_ in zip(parameters,params):
        assert p.eval(params,input,cache)==p_
    # Test inputs
    for i,i_ in zip(inputs, input):
        assert i.eval(params,input,cache)==i_ 
    # Test functions. f1 has no inputs, so it will not evaluate to `None`
    for i,f in enumerate(functions):
        if i==1:
            assert f.eval(params,input, cache)==(tuple(), dict())
        else:
            assert f.eval(params, input, cache) is None 
    # Test invisible functions. f1_ has no inputs, so it will not evaluate to `None`
    for i,f in _functions.items():
        if i==1:
            assert f.eval(params,input, cache)==(tuple(), dict())
        else:
            assert f.eval(params,input, cache) is None

