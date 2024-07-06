from .expr import Expression 
from copy import deepcopy
from jax import jit, grad, value_and_grad, Array, vmap 
from typing import Callable, TypeVar


class Dag:
    """A wrapper class around `Expression` to facilitate execution of the computational graph."""
    root: Expression 
    "The expression at the root of the computational graph"
    expressions: list[Expression]
    "A list of all of the sub-expressions, each corresponding to a subgraph"
    _fn: Callable
    "A jit-ed vectorized `_eval` returning the final value only"
    _grad: dict[tuple[Callable,bool],Callable] 
    "A cache of jit-ed gradients/value_and_grads, one for each combination of the loss function and whether to return the loss"
    recurrent: bool = False 
    "Does this Dag represent a recurrent neural network?"

    def __init__(self, root:Expression, recurrent: bool = False ):
        self.root = deepcopy(root)
        self.root.map_dfs(lambda x:x.set_depth())
        self.expressions = self.root.normalize()
        self.root.freeze()
        self._fn = None 
        self._grad = {} 
        self.recurrent = recurrent

    def __call__(self, params, input, return_intermediate = False):
        "Sugar for `self.eval()`"
        return self.eval(params,input, return_intermediate=return_intermediate)
    
    @property
    def fn(self):
        """A jit-ed vectorized eval, returning the final value and the updated params. 
        If this dag represents an RNN, will also vectorize over the parameters"""
        if self._fn is not None:
            return self._fn
        else:
            axes = [0,0] if self.recurrent else [None, 0]
            self._fn = jit(vmap(lambda params,input: self.eval(params,input),in_axes=axes))
            return self._fn

    def eval(self, params: list[Array], input: list[Array], return_intermediate: bool = False):
        """Evaluate the graph given some parameters and **unbatched** inputs. Optionally returns the values of all intermediate sub-expressions.
        Returns:
            (output, params) if return_intermediate is False, or 
            ((output, intermediates), params) if return_intermediate is True"""
        res,params = self._eval(params, input)
        return ((res[self.root.id],res) if return_intermediate else res[self.root.id]), params

    def _eval(self, params: list[Array], input: list[Array]) -> tuple[list[Array], list[Array]]:
        "Evaluate all intermediate sub-expressions using dfs and caching, and return all of their values, along with the updated parameters"
        params = params[:]
        cache = [None]*len(self.expressions)
        explored = set()
        stack = [self.root]
        while len(stack)>0:
            expr = stack[-1]
            if expr.id in explored:
                stack.pop()
            else:
                res = expr.eval(params, input, cache)
                if res is None:
                    stack.extend(expr.list_children())
                else:
                    output, params = res
                    explored.add(expr.id)
                    cache[expr.id] = output 
                    stack.pop()
        return cache, params

