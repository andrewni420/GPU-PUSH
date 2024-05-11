from .test_expr import make_tree
from gpush.push.dag.dag import Dag

def test_eval():
    """Test whether evaluating a dag gives the expected results\n
    The leaf nodes c2, c3, i1, and i3 do not appear in the intermediate calculations, as they are not present in the tree."""

    params = [1,-4,50,2,3]
    input = [9,34,0,1,-34]

    (tree, constants, inputs, functions, _functions), (params, input, (otree, oconstants, oinputs, ofunctions, _ofunctions)) = make_tree()
    
    for c,oc in zip(constants,oconstants):
        assert Dag(c).eval(params, input) == oc 
    for i,oi in zip(inputs,oinputs):
        assert Dag(i).eval(params, input) == oi 
    for f,of in zip(functions,ofunctions):
        assert Dag(f).eval(params, input) == of 
    for k in _functions.keys():
        assert Dag(_functions[k]).eval(params, input) == _ofunctions[k]

    dag = Dag(tree)
    oc0,oc1,oc2,oc3,oc4 = oconstants
    oi0,oi1,oi2,oi3,oi4 = oinputs
    assert dag.eval(params, input, return_intermediate=True) == (otree, [oc0,oc1,oc4,oi0,oi2,oi4] + ofunctions)