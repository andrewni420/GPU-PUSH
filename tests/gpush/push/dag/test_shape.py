from gpush.push.dag.shape import Shape
import gpush.push.dag.shape as shape
import pytest 
import random 

@pytest.mark.parametrize("input1", [(1,2,3),(0,12,6,47,20,4),(9,),tuple()])
@pytest.mark.parametrize("input2", [(1,2,3),(0,12,6,47,20,4),(9,),tuple()])
def test_add(input1, input2):
    "Tests whether the plus operator works on `Shape` objects"
    shape = Shape(*input1)
    shape+=Shape(*input2)
    assert isinstance(shape,Shape) and shape==Shape(*input1, *input2)

@pytest.mark.parametrize("input1", [(1,2,3),(0,12,6,47,20,4),(9,),tuple()])
@pytest.mark.parametrize("input2",list(range(10)))
def test_times(input1, input2):
    "Tests whether the times operator works on `Shape` objects"
    shape = Shape(*input1)
    shape*=input2 
    assert isinstance(shape,Shape) and shape==Shape(*(input1*input2))

@pytest.mark.parametrize("entries", [(1,2,3),(0,12,6,47,20,4),(9,),tuple()])
@pytest.mark.parametrize("index",list(range(-10,10)))
def test_plain_subscript(entries,index):
    """Test whether plain subscription works on `Shape` objects"""
    if -len(entries)<=index<len(entries):
        assert Shape(*entries)[index]==entries[index]
    else:
        with pytest.raises(Exception):
            Shape(*entries)[index]


@pytest.mark.parametrize("x", [(0,12,6,47,20,4),(9,),tuple()])
@pytest.mark.parametrize("y",[(0,6,47,20,4), (0,12,6,47,20), (1,12,6,47,20,4), (0,12,6,47,20,4), (9,), tuple()])
def test_equality(x,y):
    "Test whether the equality operator works on `Shape` objects"
    if x==y:
        assert Shape(*x)==Shape(*y)
    else:
        assert Shape(*x)!=Shape(*y)

@pytest.mark.parametrize("entries", [(1,2,3),(0,12,6,47,20,4),(9,),tuple()])
@pytest.mark.parametrize("index",list(range(-5,5)))
@pytest.mark.parametrize("val",[-100,3])
def test_clojure(entries, index, val):
    "Test whether the clojure operations `assoc`, `dissoc`, `cons`, and `conj` work on `Shape` objects"
    if -len(entries)<=index<len(entries):
        shape = Shape(*entries)
        assoc_shape = shape.assoc(index,val)
        dissoc_shape = shape.dissoc(index)
        dissoc_shape_2 = Shape(*(entries[:index%len(entries)] + entries[index%len(entries)+1:]))
        conj_shape = shape.conj(val)
        cons_shape = shape.cons(val)
        assert shape[index]==entries[index]
        assert assoc_shape[index]==val
        assert dissoc_shape==dissoc_shape_2
        assert cons_shape==Shape(val,*entries)
        assert conj_shape==Shape(*(entries+(val,)))
    else:
        with pytest.raises(IndexError):
            Shape(*entries).assoc(index,val)
        with pytest.raises(IndexError):
            Shape(*entries).dissoc(index)

@pytest.mark.parametrize("shape1,shape2,res", [((1,2,3), (2,3), (1,2,3)), (tuple(), (3,), (3,)),
                                                ((1,), (4,), (4,)), ((2,3), (3,3), None),
                                                ((2,1,3), (2,4,3), (2,4,3)), ((2,2,3), (2,3,3), None),
                                                ((3,), (4,), None), (tuple(), tuple(), tuple())])
def test_broadcast(shape1, shape2, res):
    "Test whether broadcasting works correctly on `Shape` objects"
    shape1 = Shape(*shape1)
    shape2 = Shape(*shape2)
    
    if res is None:
        assert shape1.broadcast(shape2) is None 
        assert shape2.broadcast(shape1) is None 
        assert shape.broadcast(shape1,shape2) is None
    else:
        res = Shape(*res)
        assert shape.broadcast(shape1,shape2) == shape1.broadcast(shape2) == shape2.broadcast(shape1) == res 

@pytest.mark.parametrize("shape1,shape2,res", [((1,2,3), (3,2), (1,2,2)), (tuple(), (1,3), pytest.raises(ValueError)),
                                                ((1,), (4,), None), ((2,3), (3,3), (2,3)),
                                                ((2,1,3), (2,3,4), (2,1,4)), ((2,2,3), (2,4,3), None),
                                                ((3,), (3,), tuple()), (tuple(), tuple(), pytest.raises(ValueError)),
                                                ((2,1,3), (3,3,4), None), ((2,3,4), (4,), (2,3))])
def test_mmul(shape1, shape2, res):
    "Test whether matrix multiplication of shapes works correctly"
    shape1 = Shape(*shape1)
    shape2 = Shape(*shape2)
    if res is None:
        assert shape1.mmul(shape2) is None 
        assert shape.mmul(shape1, shape2) is None 
    else:
        if isinstance(res,tuple):
            assert shape.mmul(shape1, shape2) == shape1.mmul(shape2) == Shape(*res) 
        else:
            with res as r:
                assert shape.mmul(shape1, shape2) == shape1.mmul(shape2) == Shape(*r) 




