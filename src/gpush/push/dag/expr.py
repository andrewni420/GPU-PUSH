from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Callable, Any
from .shape import Shape 
import numpy as np
from itertools import chain 


class Expression(ABC):
    frozen = False 
    """A generic expression. Forms the basis of the computational graph / directed acyclic graph.
    Possesses two copies of the id for better id updating. These copies should be identical otherwise."""
    def __init__(self, id: int, shape: Shape = Shape(), children: tuple[Expression] | dict[str,Expression] = tuple(), dtype: str = "float"):
        self._id = [id,id] 
        self.shape = shape
        self.children = children
        for c in self.list_children():
            c.parents.append(self)
        self.parents = []
        self.dtype=dtype 
        self.set_depth()

    @property 
    def id(self):
        if self._id[0]==self._id[1]:
            return self._id[0]
        else:
            raise RuntimeError(f"Expression has two different ids: {self._id}")

    def set_depth(self):
        "Initializes the depth of this expression to be 1 more than the max depth of its children"
        self.depth = 1 + max([c.depth for c in self.list_children()], default=0) 

    def list_children(self) -> List[Expression]:
        "Lists all of the children of this expression. Handles tuple and dictionary children"
        if isinstance(self.children, tuple):
            return self.children 
        return list(chain.from_iterable(self.children.values()))

    def gather(self, mapper: Callable[[Expression], Any] = None, reducer: Callable[[List[Expression]], Any] = None):
        """Uses graph DFS to construct a list of all the expressions in this dag, ordered by increasing `id`.
        \nOptionally applies `mapper` to each expression, and `reducer` to the resultant list."""
        ids = set()
        res = []
        stack = [self]
        while len(stack)>0:
            expr = stack.pop()
            if expr.id not in ids:
                res.append((expr.id, mapper(expr) if mapper else expr))
                ids.add(expr.id)
                stack.extend(expr.list_children())
        
        res = sorted(res, key = lambda x:x[0]) 
        res = [x[1] for x in res]
        res = reducer(res) if reducer else res  
        return res 
    
    def map_dfs(self, fn: Callable[[Expression], Any], idx=None ):
        """Uses graph DFS based on the individual's id to apply fn once to every expression in the dag. 
        Function cannot change the id at the given index, and should not touch the ids in general"""
        explored = set()
        stack = [self]
        while len(stack)>0:
            expr = stack.pop()
            id = expr.id if idx is None else expr._id[idx]
            if id not in explored:
                explored.add(id)
                fn(expr)
                stack.extend(expr.list_children())

    def update_ids(self, fn: Callable[[Expression], Any]):
        """Uses `map_dfs()` to update the ids in a safe way. `fn` should accept an optional argument `idx` denoting which idx is to be updated.
        This function should only modify the id at the given index, and should not touch the other index."""
        self.map_dfs(lambda x:fn(x,idx=1), idx=0)
        self.map_dfs(lambda x:fn(x,idx=0), idx=1)

    def normalize(self) -> List[Expression]:
        """Normalizes all of the `id`s in this dag so that they start from 0 and go up by 1 until the number of total expressions. 
        \nReturns the expressions in order of increasing id"""
        expressions = self.gather()
        id_map = {e.id:i for i,e in enumerate(expressions)}
        def update_id(expr, idx=0):
            expr._id[idx] = id_map.get(expr._id[idx],expr._id[idx])
        self.update_ids(fn = update_id)
        return expressions 
    
    # Propagate shape unboxing functionality upwards
    def is_shape_set(self):
        return self.shape.is_set()
    
    def unbox_shape(self):
        self.shape = self.shape.unbox()
    
    def safe_unbox(self):
        if self.is_shape_set():
            self.unbox_shape()

    def __setattr__(self, name: str, value: Any) -> None:
        if self.frozen:
            raise RuntimeError("Trying to set an attribute on a frozen expression")
        return super().__setattr__(name, value)
    
    def freeze(self):
        if self.frozen:
            return 
        self.parents = tuple(self.parents)
        self._id = tuple(self._id)
        self.frozen = True 
        for c in self.list_children():
            c.freeze()

    @abstractmethod
    def eval(self, params, input, cache):
        pass 

class Parameter(Expression):
    """Returns the parameter at index `index`. Need to specify the shape and dtype of that parameter"""
    def __init__(self, id: int, param_idx: int = 0, shape: Shape = Shape(), dtype: str = "float"):
        super().__init__(id, shape=shape, dtype=dtype)
        self.param_idx = param_idx 

    def eval(self, params, input, cache):
        return params[self.param_idx]
    
class Input(Expression):
    """Returns the input at index `index`. Need to specify the shape and dtype of that input"""
    def __init__(self, id: int, input_idx: int = 0, shape: Shape = Shape(), dtype: str = "float"):
        super().__init__(id, shape=shape, dtype=dtype)
        self.input_idx = input_idx 
    
    def eval(self, params, input, cache):
        return input[self.input_idx]
    
class Function(Expression):
    """Takes in some number of inputs and returns and output. fn is a pure jax function that takes matrices and returns a matrix.
    Need to specify the returned shape and dtype"""
    def __init__(self, id, fn: Callable, children: tuple[Expression] | dict[str,Expression] = tuple(), shape: Shape = Shape(), dtype: str = "float"):
        super().__init__(id, shape=shape, children=children, dtype=dtype)
        self.fn = fn 

    def collect_inputs(self, cache):
        """Collect the cached arguments to this expression's function, in the same shape as `self.children`.
        \nIf any arguments are unavailable, returns `None`"""
        if isinstance(self.children, tuple):
            inputs = []
            for c in self.children:
                if cache[c.id] is None:
                    return None 
                inputs.append(cache[c.id])
            return inputs
        inputs = {}
        for k,v in self.children.items():
            input = []
            for c in v:
                if cache[c.id] is None:
                    return None 
                input.append(cache[c.id])
            inputs[k] = input 
        return inputs 
    
    def eval(self, params, input, cache):
        """Evaluates the function based on the cached values of its children. If not all children were evaluated, returns None"""
        inputs = self.collect_inputs(cache)
        if inputs is None:
            return None 
        if isinstance(self.children,tuple):
            return self.fn(*inputs)
        return self.fn(**inputs)
    




    
