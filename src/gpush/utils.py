from typing import Callable, Any, Union, Optional, TypeVar
from functools import wraps 

T = TypeVar("T")
PyTree = Union[None, T, list["PyTree[T]"], dict[Any,"PyTree[T]"],tuple["PyTree[T]"]]

def preprocess(pre_fn: Callable[[Any],tuple[list,dict]], fn: Callable):
    """Returns a wrapped function that uses `pre_fn` to preprocess the args and kwargs before feeding them
    into `fn`"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args, kwargs = pre_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper 

def get_first_subtree(pytrees: list[PyTree], kwarg_trees: dict[Any,PyTree]) -> PyTree:
    if len(pytrees)>0:
        return pytrees[0] 
    valid_keys = list(filter(lambda x:x,kwarg_trees))
    if len(valid_keys)==0:
        raise ValueError("map_pytree requires at least one positional argument, or at least one non-null kwarg")
    return kwarg_trees[valid_keys[0]]

def map_pytree(fn: Callable, *pytrees: PyTree, is_leaf: Optional[Callable] = None, **kwarg_trees: PyTree) -> PyTree:
    """Maps a function across multiple pytrees of the same shape, returning the outputs in a pytree of the same shape. 
    Optionally specify an argument to determine when a list/dict/tuple subclass should be treated as a leaf node
    Raises a ValueError when the given pytrees do not have the same shape."""

    if len(pytrees)==0 and len(kwarg_trees)==0:
        raise ValueError("Must provide at least one pytree to map over")
    if any(tree is None for tree in pytrees):
        if not all(tree is None for tree in pytrees):
            raise ValueError(f"apply_to_pytree got trees of different shapes (None)")
        return fn(*pytrees) 
    
    if (is_leaf is not None and is_leaf(pytrees[0])):
        if not all(is_leaf(tree) for tree in pytrees):
            raise ValueError(f"apply_to_pytree got trees of different shapes (is_leaf: {is_leaf})")
        return fn(*pytrees, **kwarg_trees)
    first_subtree = get_first_subtree(pytrees, kwarg_trees)
    if isinstance(first_subtree,list):
        n = len(first_subtree)
        kwarg_trees = {k:([None]*len(first_subtree) if v is None else v) for k,v in kwarg_trees.items()}
        if not (all((isinstance(tree,list) and len(tree)==n) for tree in pytrees) and all((isinstance(v,list) and len(v)==n) for k,v in kwarg_trees.items())):
            raise ValueError(f"apply_to_pytree got trees of different shapes (list: {n})")
        return [map_pytree(fn,*[tree[i] for tree in pytrees], is_leaf=is_leaf, **{k:v[i] for k,v in kwarg_trees.items()}) for i in range(n)]
    elif isinstance(first_subtree,tuple):
        n = len(first_subtree)
        kwarg_trees = {k:((None,)*len(first_subtree) if v is None else v) for k,v in kwarg_trees.items()}
        if not all((isinstance(tree,tuple) and len(tree)==n) for tree in pytrees):
            raise ValueError(f"apply_to_pytree got trees of different shapes (tuple: {n})")
        return tuple(map_pytree(fn,*[tree[i] for tree in pytrees], is_leaf=is_leaf, **{k:v[i] for k,v in kwarg_trees.items()}) for i in range(n))
    elif isinstance(first_subtree, dict):
        keys = set(first_subtree.keys())
        kwarg_trees = {k:({k_:None for k_ in keys} if v is None else v) for k,v in kwarg_trees.items()}
        if not all((isinstance(tree,dict) and set(tree.keys())==keys) for tree in pytrees):
            raise ValueError(f"apply_to_pytree got trees of different shapes (dict: {keys})")
        return {k:map_pytree(fn,*[tree[k] for tree in pytrees], is_leaf=is_leaf, **{k_:v[k] for k_,v in kwarg_trees.items()}) for k in first_subtree.keys()}
    else:
        return fn(*pytrees, **kwarg_trees)
    
def is_batched_leaves(pytree: PyTree[T], n: Optional[int] = None, is_leaf: Optional[Callable[[PyTree[T]],bool]] = None) -> bool:
    """Tests whether the given pytree is a leaf node of n batched pytrees. 
    A batched leaf node is a length-n list/tuple of leaf elements."""

    # A batch must be a list of tuple of leaf nodes
    if not (isinstance(pytree,list) or isinstance(pytree,tuple)):
        return False 
    # A leaf node can't be a batch of leaf nodes
    if is_leaf is not None and is_leaf(pytree):
        return False 
    # Need to have the right batch size, if specified
    if n is not None and len(pytree)!=n:
        return False 
    # All items in the batch must be leaves
    for tree in pytree:
        if is_leaf is not None:
            if (not is_leaf(tree)) and (isinstance(tree,list) or isinstance(tree,tuple) or isinstance(tree,dict)):
                return False 
        if isinstance(tree,list) or isinstance(tree,tuple) or isinstance(tree,dict):
            return False 
    return True  

def check_n_trees(n: Optional[int], n_: int, trees: Optional[list[Union[list,dict]]], cls:type = list) -> tuple[int,list[Union[list,dict]]]:
    "Helper method to check that the batch dimension is the same, and to define the batch dimension / return value if not yet defined"
    if n is None:
        n = n_ 
    elif n!=n_:
        raise ValueError(f"split_pytree_helper got inconsistent batch sizes {(n,n_)}")
    if trees is None:
        trees = [cls() for _ in range(n)]
    return n,trees

def split_pytree_helper(pytree: PyTree[T], n: Optional[int] = None, is_leaf: Optional[Callable[[PyTree[T]],bool]] = None) -> list[PyTree[T]]:
    "Helper method for split_pytree. Additionally takes the batch dimension, when it has been defined. Raises a ValueError when inconsistent batch dimensions are found."
    if pytree is None:
        raise ValueError("split_pytree_helper cannot split a null pytree")
    elif is_batched_leaves(pytree,n=n,is_leaf=is_leaf):
        return pytree, len(pytree)
    elif isinstance(pytree,list) or isinstance(pytree,tuple):
        trees = None 
        for tree in pytree:
            subtrees, n_ = split_pytree_helper(tree, n, is_leaf=is_leaf)
            n,trees = check_n_trees(n,n_,trees, cls = list)
            for t,st in zip(trees,subtrees):
                t.append(st)
        if isinstance(pytree,tuple):
            trees = [tuple(t) for t in trees]
        return trees,n
    elif isinstance(pytree,dict):
        trees = None 
        for k,v in pytree.items():
            subtrees,n_ = split_pytree_helper(v,n=n,is_leaf=is_leaf)
            n,trees = check_n_trees(n,n_,trees, cls = dict)
            for t,st in zip(trees,subtrees):
                t[k]=st
        return trees,n
    else:
        raise ValueError("split_pytree_helper encountered an unrecognized batch pytree entry which is neither a batch of leaves or a list/tuple/dict thereof")
        
def split_pytree(pytree: PyTree[T], is_leaf: Optional[Callable[[PyTree[T]],bool]] = None) -> list[PyTree[T]]:
    """Given a batched pytree, where instead of leaves we have identical-length lists of n leaves, splits into n 
    unbatched pytrees. 
    Optionally specify a function to determine whether an item is a leaf. This function cannot specify when an object 
    is not a leaf, but can be useful when we have list or dictionary leaf elements.
    This is the inverse operation of `map_pytree(lambda *args:args, *pytrees)`"""
    return split_pytree_helper(pytree, n=None, is_leaf=is_leaf)[0]

def pytree_to_list(pytree: PyTree[T]) -> list[T]:
    "Construct a list of the elments inside a pytree"
    elements = []
    map_pytree(lambda x:elements.append(x),pytree)
    return elements  

def pytree_from_list(pytree: PyTree[T], elements: list[T]) -> PyTree[T]:
    """Construct a pytree of the same shape as the given pytree using the elements in the given list."""
    # Copy to avoid modifying the argument
    elements = [e for e in elements]
    return map_pytree(lambda x:elements.pop(0),pytree)