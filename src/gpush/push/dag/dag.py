from .expr import Expression 
from copy import deepcopy
from jax import jit


class Dag:
    def __init__(self, root:Expression):
        self.root = deepcopy(root)
        self.root.map_dfs(lambda x:x.set_depth())
        self.expressions = self.root.normalize()
        self.root.freeze()
        self._fn = None 

    def __call__(self, params, input, return_intermediate = False):
        return self.eval(params,input,return_intermediate=return_intermediate)
    
    def eval(self, params, input, return_intermediate = False):
        if self._fn is None:
            self._fn = jit(lambda params,input: self._eval(params,input))
        res = self._fn(params,input)
        return (res[self.root.id],res) if return_intermediate else res[self.root.id]

    def _eval(self, params, input):
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

