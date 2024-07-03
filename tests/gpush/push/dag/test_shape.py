from gpush.push.dag.shape import Shape, SizePlaceholder
import gpush.push.dag.shape as shape
import pytest 
import random 
import jax.numpy as jnp 
import jax.lax as lax 


@pytest.mark.parametrize("input1", [-10, -3, -1, 0, 1, 3, 10, None])
@pytest.mark.parametrize("input2", [-10, -3, -1, 0, 1, 3, 10, None])
def test_placeholder(input1, input2):
    "Tests for correct arithmetic operations on `SizePlaceholder` objects"
    p1 = SizePlaceholder()
    p1.value = input1

    p2 = SizePlaceholder()
    p2.value = input2 

    if input1 is not None and input2 is not None:
        assert input1+input2 == p1+input2 == input1+p2 == p1+p2
        assert input1*input2 == p1*input2 == input1*p2 == p1*p2

        assert(isinstance(p1+input2, int))
        assert(isinstance(input2+p1, int))
        assert(isinstance(p1+p2, int))
        assert(isinstance(p2+p1, int))
    else:
        add = p1+p2 
        assert(isinstance(add,SizePlaceholder))
        assert(add.value is None)
        if input1 is not None:
            add = input1+p2 
            assert(isinstance(add,SizePlaceholder))
            assert(add.value is None)
        if input2 is not None:
            add = p1+input2 
            assert(isinstance(add,SizePlaceholder))
            assert(add.value is None)

def test_unboxing():
    "Tests unboxing functionality"
    shape = Shape(SizePlaceholder(), 1, 2)

    assert not shape.is_set()
    with pytest.raises(RuntimeError):
        shape.unbox()
    
    shape[0].value = 10
    assert shape.is_set()
    assert shape.unbox()==Shape(10,1,2)

    shape = Shape(SizePlaceholder(), SizePlaceholder(), SizePlaceholder())

    assert not shape.is_set()
    with pytest.raises(RuntimeError):
        shape.unbox()
    
    shape[0].value = 10
    assert not shape.is_set()
    with pytest.raises(RuntimeError):
        shape.unbox()
    
    shape[1].value = 20
    shape[2].value = 30
    assert shape.is_set()
    assert shape.unbox()==Shape(10,20,30)


@pytest.mark.parametrize("partition", [(10,), (3,3,4), (1,)*10, (1,1,8), (1,2,5,2)])
def test_link(partition):
    """Tests whether placeholders that are linked together will have the same value"""
    s = [SizePlaceholder() for _ in range(10)]
    cur_idx = 0
    for p in partition:
        for i in range(p-1):
            s[cur_idx+i].link(s[cur_idx+i+1])
        cur_idx+=p

    cur_idx=0
    for i,p in enumerate(partition):
        s[cur_idx].value=i 
        cur_idx+=p 
    
    cur_idx=0
    for i,p in enumerate(partition):
        for j in range(p):
            assert s[cur_idx+j].value==i 
        cur_idx+=p 

def test_link_error():
    "Tests whether errors are raised appropriately when linking placeholders"
    s = [SizePlaceholder() for _ in range(10)]
    s[0].value = 1
    s[0].link(s[1])
    assert s[1].value==1 
    s[2].link(s[1])
    assert s[2].value==1
    s[3].link(s[4])
    s[4].link(s[0])
    assert s[3].value==s[4].value==1

    s[5].value=2
    s[6].value=3
    with pytest.raises(ValueError):
        s[5].link(s[6])
    with pytest.raises(ValueError):
        s[1].link(s[6])
    with pytest.raises(ValueError):
        s[0].link(s[1],s[6])

    s[7].value=4
    s[8].value=4
    s[9].value=4 
    s[7].link(s[8],s[9])

def test_set_placeholder():
    "Tests the `Shape.set_placeholder()` function"
    shape = Shape(SizePlaceholder(), 1, 2)
    shape2 = Shape(3,1,2)
    shape.set_placeholders(shape2)
    assert shape[0].value==3

    shape = Shape(SizePlaceholder(), 1, 2)
    shape[:-2].set_placeholders(shape2[:-2])
    assert shape[0].value==3

    shape = Shape(SizePlaceholder(), 1, 2)
    shape2 = Shape(SizePlaceholder(), 1, 2)
    shape2[0].value = 3
    shape.set_placeholders(shape2)
    assert shape[0].value==3

    shape = Shape(SizePlaceholder(), 1, 2)
    shape[0].value=2
    with pytest.raises(ValueError):
        shape.set_placeholders(shape2)
    with pytest.raises(ValueError):
        shape[:-2].set_placeholders(shape2[:-2])

    shape2=Shape(SizePlaceholder(),1,2)
    shape2[0].value=3
    with pytest.raises(ValueError):
        shape.set_placeholders(shape2)

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


@pytest.mark.parametrize("x", [(0,12,6,47,20,4),(9,),tuple(), (SizePlaceholder(),1,2)])
@pytest.mark.parametrize("y",[(0,6,47,20,4), (0,12,6,47,20), (1,12,6,47,20,4), (0,12,6,47,20,4), (9,), tuple(), (SizePlaceholder(),1,2)])
def test_equality(x,y):
    "Test whether the equality operator works on `Shape` objects"
    if x==y:
        assert Shape(*x)==Shape(*y)
        assert not (Shape(*x)!=Shape(*y))
    else:
        assert Shape(*x)!=Shape(*y)
        assert not (Shape(*x)==Shape(*y))

@pytest.mark.parametrize("s1,s2,res", [([(1,1)],[(1,1)],True), ([(1,2)],[(1,1)],False), ([(0,1)],[(1,1)],False), ([(0,1), (1,2)],[(0,1), (1,2)],True), ([(0,2), (2,4)],[(0,1), (2,4)],False)])
def test_placeholder_equality(s1,s2,res):
    shape1 = Shape(SizePlaceholder(),SizePlaceholder(),SizePlaceholder())
    shape2 = Shape(SizePlaceholder(),SizePlaceholder(),SizePlaceholder())
    for i,v in s1:
        shape1[i].value=v 
    for i,v in s2:
        shape2[i].value=v 
    if res:
        assert shape1==shape2 
        assert not (shape1!=shape2)
    else:
        assert shape1!=shape2 
        assert not (shape1==shape2)


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
                                                ((3,), (4,), None), (tuple(), tuple(), tuple()),
                                                ((SizePlaceholder(),),(4,),(4,)), ((SizePlaceholder(),), (SizePlaceholder(),), (SizePlaceholder(),)), 
                                                ((SizePlaceholder(),2,3),(4,),None), ((2,3,SizePlaceholder()),(4,),(2,3,4)), 
                                                ((SizePlaceholder(),2,3),(3,),(SizePlaceholder(),2,3)),
                                                ((SizePlaceholder(),SizePlaceholder(),SizePlaceholder()),(3,1,SizePlaceholder()),(3,1,SizePlaceholder()))])
def test_broadcast(shape1, shape2, res):
    "Test whether broadcasting works correctly on `Shape` objects"

    s1 = Shape(*shape1)
    s2 = Shape(*shape2)

    if res is None:
        assert s1.broadcast(s2) is None 
        assert s1.broadcast(s2) is None 
        assert shape.broadcast(s1,s2) is None
    else:
        res = Shape(*res)
        assert shape.broadcast(s1,s2) == s1.broadcast(s2) == s2.broadcast(s1) == res 

@pytest.mark.parametrize("shape1,shape2,res", [((1,2,3), (3,2), (1,2,2)), (tuple(), (1,3), pytest.raises(ValueError)),
                                                ((1,), (4,), None), ((2,3), (3,3), (2,3)),
                                                ((2,1,3), (2,3,4), (2,1,4)), ((2,2,3), (2,4,3), None),
                                                ((3,), (3,), tuple()), (tuple(), tuple(), pytest.raises(ValueError)),
                                                ((2,1,3), (3,3,4), None), ((2,3,4), (4,), (2,3)),
                                                ((SizePlaceholder(),),(4,),tuple()), ((SizePlaceholder(),3), (3,SizePlaceholder(),), (SizePlaceholder(), SizePlaceholder())), 
                                                ((SizePlaceholder(),2,3),(3,3),(SizePlaceholder(),2,3)), ((2,3,SizePlaceholder()),(4,2),(2,3,2)), 
                                                ((SizePlaceholder(),SizePlaceholder(),SizePlaceholder()),(3,1,SizePlaceholder()),(3,SizePlaceholder(),SizePlaceholder()))])
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

@pytest.mark.parametrize("shape1,shape2", [((2,3,10,10),(4,3,3,3)), ((4,5,10,10),(4,5,1,1)),
                                           ((2,3,3,3),(4,3,4,4)), ((2,3,7,3),(4,3,4,2)),
                                           ((2,3,10,10),(4,3,3,3)), ((2,3,5,10),(4,3,3,6))])
@pytest.mark.parametrize("stride", [(1,1), (2,1), (3,1), (1,3), (3,3)])
@pytest.mark.parametrize("padding", [((0,0),(0,0)), ((2,0),(0,2)), ((2,1),(3,2)), "SAME"])
@pytest.mark.parametrize("lhs,rhs", [((1,1),(1,1)), ((2,1),(1,3)), ((1,3),(4,1)), ((2,3),(1,4))])
def test_conv_2d(shape1, shape2, stride, padding, lhs, rhs):
    n = len(shape1)
    if not (len(stride)==len(lhs)==len(rhs)==n-2):
        pytest.skip()
    if padding!="SAME" and len(padding)!=n-2:
        pytest.skip() 
    if padding=="SAME" and any(i>1 for i in lhs):
        pytest.skip()
    s1 = Shape(*shape1)
    s2 = Shape(*shape2)
    arr1 = jnp.zeros(shape1)
    arr2 = jnp.zeros(shape2)
    arr = lax.conv_general_dilated(arr1, arr2, stride, padding, lhs_dilation=lhs, rhs_dilation=rhs)
    s = shape.conv(s1, s2, stride=stride, padding=padding, lhs_dilation=lhs, rhs_dilation=rhs)
    actual_shape = arr.shape if all([i>0 for i in arr.shape]) else None 
    if actual_shape is None:
        assert actual_shape==s 
    else:
        assert actual_shape==tuple(s)

@pytest.mark.parametrize("shape1,shape2", [((2,3,10,10,10),(4,3,3,3,3)), ((4,5,10),(4,5,1)),
                                           ((2,3,5,6,7),(4,3,3,2,4)), ((2,3,5),(4,3,1))])
@pytest.mark.parametrize("stride", [(1,), (3,), (1,1,1), (3,1,2)])
@pytest.mark.parametrize("padding", [((0,0),), ((2,3),), ((0,0),(0,0),(0,0)), ((2,1),(3,2),(5,6)), "SAME"])
@pytest.mark.parametrize("lhs,rhs", [((1,),(1,)), ((2,),(3,)), ((1,1,1), (1,1,1)), ((1,3,2),(4,1,3))])
def test_conv_nd(shape1, shape2, stride, padding, lhs, rhs):
    n = len(shape1)
    if not (len(stride)==len(lhs)==len(rhs)==n-2):
        pytest.skip()
    if padding!="SAME" and len(padding)!=n-2:
        pytest.skip() 
    if padding=="SAME" and any(i>1 for i in lhs):
        pytest.skip()
    s1 = Shape(*shape1)
    s2 = Shape(*shape2)
    arr1 = jnp.zeros(shape1)
    arr2 = jnp.zeros(shape2)
    arr = lax.conv_general_dilated(arr1, arr2, stride, padding, lhs_dilation=lhs, rhs_dilation=rhs)
    s = shape.conv(shape1, shape2, stride=stride, padding=padding, lhs_dilation=lhs, rhs_dilation=rhs)
    actual_shape = arr.shape if all([i>0 for i in arr.shape]) else None 
    if actual_shape is None:
        assert actual_shape==s 
    else:
        assert actual_shape==tuple(s)

@pytest.mark.parametrize("padding",["SAME",["SAME"],[(0,0)]])
@pytest.mark.parametrize("shape1,shape2,res1,res2", [((SizePlaceholder(),3,10,10),(4,3,3,3), (SizePlaceholder(),4,8,8),(SizePlaceholder(),4,10,10)),
                                               ((2,3,10,10),(SizePlaceholder(),3,3,3), (2,SizePlaceholder(),8,8),(2,SizePlaceholder(),10,10)),
                                               ((2,3,SizePlaceholder(),10),(4,3,3,3), None, (2,4,SizePlaceholder(),10)),
                                               ((2,3,10,10),(4,3,3,SizePlaceholder()), None,(2,4,10,SizePlaceholder()))])
def test_conv_placeholder(padding, shape1, shape2, res1, res2):
    n = len(shape1)
    s1 = Shape(*shape1)
    s2 = Shape(*shape2)
    lhs = rhs = stride = (1,)*(n-2)
    if isinstance(padding,list):
        padding = padding*(n-2)
    res = res2 if (padding=="SAME" or (isinstance(padding, list) and padding[0]=="SAME")) else res1
    s = shape.conv(s1, s2, stride=stride, padding=padding, lhs_dilation=lhs, rhs_dilation=rhs)
    if res is None:
        assert s==res
    else:
        for s_,r_ in zip(s,res):
            assert type(s_)==type(r_)
        assert s==res

