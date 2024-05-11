

class PushState(dict):
    def observe(self, stacks: dict|tuple) -> dict:
        "Return the top items in the stacks requested. If a tuple if provided, returns the top item from each stack"
        if isinstance(stacks,tuple):
            res = []
            for stack in stacks:
                if len(self[stack])==0:
                    return None 
                res.append(self[stack][-1])
            return tuple(res)
        res = {} 
        for stack,n in stacks.items():
            if len(self[stack]<n):
                return None 
            res[stack] = self[stack][-n:]
        return res
    
    def pop_from_stacks(self, stacks:dict|tuple):
        "Remove items from the stacks. If stacks is a tuple, we assume removing one item per stack"
        if isinstance(stacks,tuple):
            stacks = {stack:1 for stack in stacks}
        for k,v in stacks.items():
            self[k] = self[k][:-v]
        return self 
    
    def push_to_stacks(self, stacks: dict):
        "Add items to the stacks"
        for k,v in stacks.items():
            self[k].extend(v)
        return self 
    
    def size(self) -> int:
        "Number of items in the push state"
        return sum([len(self[k]) for k in self.keys()])
    
