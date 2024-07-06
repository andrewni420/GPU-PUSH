from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Callable, Any, Union
from .shape import Shape 
import numpy as np
from itertools import chain 
from jax import Array
from functools import partial
import numpy as np
import re 

Arguments = Union[tuple,dict]

class Expression(ABC):
    """A generic expression. Forms the basis of the computational graph / directed acyclic graph.
    Possesses two copies of the id for better id updating. These copies should be identical otherwise."""
    frozen: bool = False 
    "Whether this expression has been frozen and can no longer be modified"
    _id: list[int]
    "An internal list of two ids to help in setting ids. Do not modify."
    shape: Shape 
    "The shape of the resultant matrix"
    children: tuple[Expression] | dict[str,Expression]
    "The sub-expressions to be fed as arguments into this expression"
    parents: list[Expression]
    "The expressions that contain this expression as an argument"
    dtype: str 
    "The datatype of the resultant matrix"

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
        "The id of the expression"
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
        `fn` cannot change the id at the given index, and should not touch the ids in general."""
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
        """Uses `map_dfs()` to update the ids in a safe way, by updating the left and then the right ids. 
        \n`fn` should accept an optional argument `idx` denoting which idx is to be updated.
        \nThis function should only modify the id at the given index, and should not touch the other index."""
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
        "Tests whether all SizePlaceholders in this expression's shape have been set"
        return self.shape.is_set()
    
    def unbox_shape(self):
        "Unbox this expression's shape"
        self.shape = self.shape.unbox()
    
    def safe_unbox(self):
        "Unbox, but only if all SizePlaceholders have been set."
        if self.is_shape_set():
            self.unbox_shape()

    def __setattr__(self, name: str, value: Any) -> None:
        if self.frozen:
            raise RuntimeError("Trying to set an attribute on a frozen expression")
        return super().__setattr__(name, value)
    
    def freeze(self):
        "Freeze the expression, rendering it immutable"
        if self.frozen:
            return 
        self.parents = tuple(self.parents)
        self._id = tuple(self._id)
        self.frozen = True 
        for c in self.list_children():
            c.freeze()

    def hidden_params(self):
        "Returns a list of the param indices which are hidden states, and are assigned values during execution"
        return list(self.gather(lambda x:x.param_idx if isinstance(x,ParamUpdate) else None, partial(filter,lambda x:x)))


    @abstractmethod
    def eval(self, params: list[Array], input: list[Array], cache: list[Array]) -> tuple[Array,list[Array]]:
        "Evaluate this expression based on parameters, inputs, and a cache containing the values of other expressions. Returns the output and the updated params"
        pass 

    def __str__(self):
        children = tuple(c.id for c in self.children) if isinstance(self.children,tuple) else {k:[c.id for c in v] for k,v in self.children.items()}
        return f"Expression({self.id}, {self.shape}, {self.dtype}, children={children})"
    def __repr__(self):
        return str(self)

class Parameter(Expression):
    """Returns the parameter at index `index`. Need to specify the shape and dtype of that parameter"""
    param_idx: int 
    "The index of the parameter referenced by this expression"
    def __init__(self, id: int, param_idx: int = 0, shape: Shape = Shape(), dtype: str = "float"):
        super().__init__(id, shape=shape, dtype=dtype)
        self.param_idx = param_idx 

    def eval(self, params: list[Array], input: list[Array], cache: list[Array]) -> tuple[Array,list[Array]]:
        "Returns the parameter at `self.param_idx`"
        # print(f"{self}\n\t inputs {params[self.param_idx].shape} self {self.shape}")
        return params[self.param_idx], params
    def __str__(self):
        return f"Parameter({self.id}, {self.param_idx}, {self.shape}, {self.dtype})"
    
class Input(Expression):
    """Returns the input at index `index`. Need to specify the shape and dtype of that input"""
    input_idx: int 
    "The index of the input referenced by this expression"
    def __init__(self, id: int, input_idx: int = 0, shape: Shape = Shape(), dtype: str = "float"):
        super().__init__(id, shape=shape, dtype=dtype)
        self.input_idx = input_idx 
    
    def eval(self, params: list[Array], input: list[Array], cache: list[Array]) -> tuple[Array,list[Array]]:
        "Returns the input at `self.input_idx`"
        # print(f"{self}\n\t inputs {input[self.input_idx].shape} self {self.shape}")
        return input[self.input_idx], params
    def __str__(self):
        return f"Input({self.id}, {self.input_idx}, {self.shape}, {self.dtype})"
    
class Literal(Expression):
    """Returns a literal value"""
    value: Array 
    "The value returned by this expression"
    def __init__(self, id: int, val: Array):
        super().__init__(id, shape=val.shape, dtype=re.search("[a-zA-Z]+",str(val.dtype))[0])
        self.value=val 

    def eval(self, params: list[Array], input: list[Array], cache: list[Array]) -> tuple[Array,list[Array]]:
        "Returns the expression's literal value"
        # print(f"{self}\n\t inputs {params[self.param_idx].shape} self {self.shape}")
        return self.value, params
    def __str__(self):
        return f"Literal({self.id}, {self.param_idx}, {self.shape}, {self.dtype})"
    
def default_arg_reconstructor(children: Arguments, other_args: Arguments) -> Arguments:
    if other_args is None:
        return children 
    if isinstance(children,tuple):
        return children+other_args
    else:
        return dict(**children,**other_args)

class Function(Expression):
    """Takes in some number of inputs and returns and output. fn is a pure jax function that takes matrices and returns a matrix.
    Need to specify the returned shape and dtype"""

    fn: Callable 
    "The function represented by this expression"
    other_args: Union[tuple,dict]
    "Other, non-children arguments of the function"
    arg_reconstructor: Callable
    """A function to take the evaluated children along with the other arguments, and reconstruct the 
    original argument order to use for calling `fn`.
    If not specified, defaults to concatenating tuples and merging dictionaries."""
    
    def __init__(self, 
                 id: int, 
                 fn: Callable, 
                 children: tuple[Expression] | dict[str,Expression] = tuple(), 
                 shape: Shape = Shape(), 
                 dtype: str = "float",
                 other_args: Arguments = None,
                 arg_reconstructor: Callable[[Arguments, Arguments], Arguments] =None,
                 hidden_state: bool = False):
        super().__init__(id, shape=shape, children=children, dtype=dtype)
        self.fn = fn 
        self.other_args = other_args 
        self.arg_reconstructor = arg_reconstructor

    def collect_inputs(self, cache: list) -> Arguments:
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
    
    def eval(self, params: list[Array], input: list[Array], cache: list[Array]) -> tuple[Array,list[Array]]:
        """Evaluates the function based on the cached values of its children. If not all children were evaluated, returns None"""
        inputs = self.collect_inputs(cache)
        if inputs is None:
            return None 
        reconstructor = default_arg_reconstructor if self.arg_reconstructor is None else self.arg_reconstructor
        inputs = inputs if self.other_args is None else reconstructor(inputs,self.other_args)
        # input_shapes = tuple(c.shape for c in inputs) if (isinstance(inputs,tuple) or isinstance(inputs,list)) else {k:[c.shape for c in v] for k,v in inputs.items()}
        # print(f"{self}\n\t inputs {input_shapes} self {self.shape}")
        if isinstance(self.children,tuple):
            output = self.fn(*inputs)
        else:
            output = self.fn(**inputs)

        
        return output, params
    def __str__(self):
        children = tuple(c.id for c in self.children) if isinstance(self.children,tuple) else {k:[c.id for c in v] for k,v in self.children.items()}
        return f"Function({self.id}, {self.shape}, {self.dtype}, children={children})"

class ParamUpdate(Function):
    """Updates a parameter at index 'param_idx' using the value at 'id' in the cache."""

    def __init__(self, id: int, children: Union[dict[str,Expression],tuple[Expression]], param_idx: int = 0, shape: Shape = Shape(), dtype: str = "float"):
        super().__init__(id, lambda x:x, shape=shape, children=children, dtype=dtype)
        self.param_idx = param_idx
        self.children = children
        if len(children)!=1:
            raise ValueError("ParamUpdate takes exactly one child")

    def eval(self, params: list[Array], input: list[Array], cache: list[Array]) -> tuple[Array,list[Array]]:
        """Updates the parameter at self.param_idx using the value from the cache."""
        inputs = self.collect_inputs(cache)
        if inputs is None:
            return None 
        cached_value = inputs[0]

        # Update the parameter at self.param_idx using the cached value
        params = params[:]
        params[self.param_idx] = cached_value
        return cached_value, params
    def __str__(self):
        return f"ParamUpdate({self.id}, {self.param_idx}, {self.shape}, {self.dtype}, children={[c.id for c in self.children]})"

    
