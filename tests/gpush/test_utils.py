import jax.numpy as jnp
from gpush.utils import * 
import pprint

def test_split_pytree():
    "Tests the ability of split_pytree to unbatch batched pytrees into separate pytrees"
    arr1 = jnp.array([1,2,3,4])
    arr2 = jnp.array([[2],[43],[12]])
    pytree1 = {"one": 1,
               (1,2): {"two": 2,
                       "three": (3,4)},
               frozenset({"24",14}): [{"four":4},1,3]}
    pytree2 = {"one": "b",
               (1,2): {"two": "a",
                       "three": (0.1,"c")},
               frozenset({"24",14}): [{"four":arr1},"m","a"]}
    pytree3 = {"one": 4,
               (1,2): {"two": 212,
                       "three": (342,231)},
               frozenset({"24",14}): [{"four":"three"},-12,arr2]}
    
    pytree = {"one": [1,"b",4],
               (1,2): {"two": [2,"a",212],
                       "three": ([3,0.1,342],[4,"c",231])},
               frozenset({"24",14}): [{"four":[4,arr1,"three"]},[1,"m",-12],[3,"a",arr2]]}
    
    assert pytree == map_pytree(lambda *args, **kwargs:[*args,*kwargs.values()], pytree1,pytree2,lmao=pytree3)
    assert split_pytree(pytree) == [pytree1, pytree2, pytree3]


def test_map_pytree():
    "Tests the ability of map_pytree to accumulate and clip gradients"
    pytree1 = {"one": jnp.array([1.,2,3,4]),
               (1,2): {"two": jnp.array([[1,2,3.,4],[2,4,5,2]])},
               frozenset({"24",14}): [{"four":jnp.array([1])},jnp.array([[4,3.],[2,3]]),jnp.array([1.,42,1,1])]}
    pytree1_ = {"one": jnp.array([1.,2,3,4]),
               (1,2): {"two": jnp.array([[1,2,3.,4],[2,4,5,2]])},
               frozenset({"24",14}): [{"four":jnp.array([1])},jnp.array([[4,3.],[2,3]]),jnp.array([1.,5,1,1])]}
    pytree2 = {"one": jnp.array([-10,-21,3,40.]),
               (1,2): {"two": jnp.array([[1.2,20,-23,-4],[-2,0,51,1002]])},
               frozenset({"24",14}): [{"four":jnp.array([-10])},jnp.array([[43,32],[-12.,13]]),jnp.array([1.12,-2.2,1.1,-1])]}
    pytree2_ = {"one": jnp.array([-5,-5,3,5.]),
               (1,2): {"two": jnp.array([[1.2,5,-5,-4],[-2,0,5,5]])},
               frozenset({"24",14}): [{"four":jnp.array([-5])},jnp.array([[5,5],[-5.,5]]),jnp.array([1.12,-2.2,1.1,-1])]}
    
    pytree = {"one": jnp.array([-4,-3,6,9.]),
               (1,2): {"two": jnp.array([[2.2,7,-2,0],[0,4,10,7]])},
               frozenset({"24",14}): [{"four":jnp.array([-4])},jnp.array([[9,8],[-3,8]]),jnp.array([2.12,2.8,2.1,0])]}
    
    pytree_min = {"one": None,
               (1,2): {"two": 1},
               frozenset({"24",14}): [{"four":jnp.array([-10])},jnp.array([[3,6],[-12.,0]]),jnp.array([1.12,-2.9,-1.1,1.121])]}
    pytree_max = {"one": jnp.array([1,1,4,5.]),
               (1,2): None,
               frozenset({"24",14}): [{"four":jnp.array([10])},jnp.array([[3,7],[-2.,9]]),jnp.array([1.45,3.2,10.1,3])]}
    pytree_minmax = {"one": jnp.array([-3,-3,4,5.]),
               (1,2): {"two": jnp.array([[2.2,3,1,1],[1,3,3,3]])},
               frozenset({"24",14}): [{"four":jnp.array([-4])},jnp.array([[3,7],[-3,8]]),jnp.array([1.45,2.8,2.1,1.121])]}
    
    success = {"one": True,
               (1,2): {"two": True},
               frozenset({"24",14}): [{"four":True},True,True]}
    
    assert success == map_pytree(lambda x,y: jnp.allclose(x,y), pytree1, pytree1)
    assert success == map_pytree(lambda x,y: jnp.allclose(x,y), pytree2, pytree2)

    assert success == map_pytree(lambda x=None,y=None: jnp.allclose(x,y), x=pytree1, y=pytree1)
    assert success == map_pytree(lambda x,y=None: jnp.allclose(x,y), pytree2, y=pytree2)

    assert success == map_pytree(lambda x,y,z=1: jnp.allclose(x,y) and (z is None), pytree2, pytree2, z=None)
    assert success == map_pytree(lambda x,y,z=1: jnp.allclose(x,y) and (z == 1), pytree2, pytree2)
    
    pytree1_res = map_pytree(lambda x:jnp.maximum(-5,jnp.minimum(5,x)), pytree1)
    pytree2_res = map_pytree(lambda x:jnp.maximum(-5,jnp.minimum(5,x)), pytree2)
    assert success == map_pytree(lambda x,y: jnp.allclose(x,y), pytree1_, pytree1_res)
    assert success == map_pytree(lambda x,y: jnp.allclose(x,y), pytree2_, pytree2_res)

    pytree_res = map_pytree(lambda x,y: x+y, pytree1_, pytree2_)
    assert success == map_pytree(lambda x,y: jnp.allclose(x,y), pytree, pytree_res)

    def minmax(x, min=None, max=None):
        min = -3 if min is None else min 
        max = 3 if max is None else max 
        return jnp.maximum(min,jnp.minimum(max,x))

    pytree_minmax_res = map_pytree(minmax, pytree, min=pytree_min,max=pytree_max)
    assert success == map_pytree(lambda x,y: jnp.allclose(x,y), pytree_minmax, pytree_minmax_res)


def test_pytree_list():
    "Tests the ability to flatten and unflatten pytrees"
    arr1 = jnp.array([1,2,3,4])
    arr2 = jnp.array([[2],[43],[12]])
    pytree = {"one": arr2,
               (1,2): {"two": 2,
                       "three": (3,arr1)},
               frozenset({"24",14}): [{"four":2.2},1.3,"abc"]}
    
    pytree_ = {"one": None,
               (1,2): {"two": None,
                       "three": (None,None)},
               frozenset({"24",14}): [{"four":None},None,None]}

    res = [arr2, 2, 3, arr1, 2.2, 1.3, "abc"]
    
    assert pytree_to_list(pytree)==res 
    assert pytree_from_list(pytree_,res) == pytree_from_list(pytree,res) == pytree