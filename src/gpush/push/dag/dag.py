from .expr import Expression 
from copy import deepcopy

class Dag:
    def __init__(self, root:Expression):
        self.root = deepcopy(root)
        self.expressions = self.root.normalize()

    def eval(self, params, input, return_intermediate = False):
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

        if return_intermediate:
            return cache[self.root.id], cache 
        return cache[self.root.id]

