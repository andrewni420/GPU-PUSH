from .expr import Expression 
from copy import deepcopy
from jax import jit, grad, value_and_grad, Array, vmap 
from typing import Callable 


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

    def __init__(self, root:Expression):
        self.root = deepcopy(root)
        self.root.map_dfs(lambda x:x.set_depth())
        self.expressions = self.root.normalize()
        self.root.freeze()
        self._fn = None 
        self._grad = {} 

    def __call__(self, params, input, return_intermediate = False):
        "Sugar for `self.eval()`"
        return self.eval(params,input,return_intermediate=return_intermediate)
    
    @property
    def fn(self):
        "A jit-ed vectorized eval, returning only the final value"
        if self._fn is not None:
            return self._fn
        else:
            self._fn = jit(vmap(lambda params,input: self._eval(params,input)[self.root.id],in_axes=[None,0]))
            return self._fn
        
    def grad(self, loss_fn: Callable, return_value: bool = False) -> Callable:
        """Returns a jit-ed vectorized function to calculate the gradient with respect to a loss function.
        Caches values to avoid re-jit-ing many times."""
        if (loss_fn,return_value) in self._grad:
            return self._grad[(loss_fn,return_value)]
        
        fn = lambda params, input, target: loss_fn(self.fn(params, input), target)
        
        if return_value:
            func =  jit(value_and_grad(fn))
        else:
            func =  jit(grad(fn))
        
        self._grad[(loss_fn,return_value)] = func 
        return func 

    def eval(self, params, input, return_intermediate = False):
        """Evaluate the graph given some parameters and **unbatched** inputs. Optionally returns the values of all intermediate sub-expressions."""
        res = self._eval(params,input)
        return (res[self.root.id],res) if return_intermediate else res[self.root.id]

    def _eval(self, params, input):
        "Evaluate all intermediate sub-expressions using dfs and caching, and return all of their values"
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
                    explored.add(expr.id)
                    cache[expr.id] = res 
                    stack.pop()
        return cache

