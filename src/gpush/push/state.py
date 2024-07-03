from typing import List, Union
import jax.numpy as jnp
from jax import Array
import jax.random as random
import jax.nn.initializers as init

INITIALIZERS: dict[str,init.Initializer] = {#"constant": init.constant(), 
                "delta_orthogonal": init.delta_orthogonal(),
                "glorot_normal": init.glorot_normal(),
                "glorot_uniform": init.glorot_uniform(),
                "he_normal": init.he_normal(in_axis=0,out_axis=1),
                "he_uniform": init.he_uniform(),
                "lecun_normal": init.lecun_normal(),
                "lecun_uniform": init.lecun_uniform(),
                "normal": init.normal(),
                "ones": lambda *args, **kwargs: init.ones(*args, **kwargs),
                "orthogonal": init.orthogonal(),
                "truncated_normal": init.truncated_normal(),
                "uniform": init.uniform(),
                # "variance_scaling": init.variance_scaling(),
                "zeros": lambda *args, **kwargs: init.zeros(*args, **kwargs)}
"""Various initializers copied over from jax.nn.initializers. Currently does not support parameterized
initializers such as `variance_scaling` or `constant`"""


class PushState(dict):
    """A class to represent the current state of the execution of a Push Program.
    
    Fields:
        nsteps (int): The number of execution steps taken so far
        params (list): A list containing specifications for the parameters created during execution of the program
        input (list): A list of specifications of the inputs to the neural network"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsteps = 0 
        self.params = []
        self.input = [] 

    def initialize(self, params: list, input: list):
        "Set the params and the inputs of the PushState"
        self.params = params
        self.input = input 
        self.nsteps = 0
        return self 
    
    def observe(self, stacks: Union[dict[str,int], tuple]) -> dict:
        "Return the top items in the stacks requested. If a tuple if provided, returns the top item from each stack"
        if stacks is None:
            return {}
        elif isinstance(stacks,str):
            return self[stacks][-1]
        elif isinstance(stacks,tuple):
            res = []
            for stack in stacks:
                if len(self[stack])==0:
                    return None 
                res.append(self[stack][-1])
            return tuple(res)
        else:
            res = {} 
            for stack,n in stacks.items():
                if len(self[stack])<n:
                    return None 
                res[stack] = self[stack][-n:] if n>0 else []
            return res
    
    def pop_from_stacks(self, stacks:Union[dict[str,int],tuple]):
        "Remove items from the stacks. If stacks is a tuple, we assume removing one item per stack"
        if isinstance(stacks,tuple):
            stacks = {stack:1 for stack in stacks}
        for k,v in stacks.items():
            self[k] = self[k][:-v] if v>0 else self[k]
        return self 
    
    def push_to_stacks(self, stacks: dict):
        "Add items to the stacks"
        for k,v in stacks.items():
            self[k].extend(v)
        return self 
    
    def size(self) -> int:
        "Number of items in the push state"
        return sum([len(self[k]) for k in self.keys()])
    
    def step(self):
        self.nsteps+=1 
        return self 
    
    def init_param(self, key: Array, param: dict, default: str = None) -> Array:
        """Initializes a parameter array using a random key.

        Parameters:
            key (Array): The random key
            param (dict): A dictionary containing the shape, datatype, and optionally the initializer.
                If no initializer is specified, uses kaiming initialization
            default (str): The default initializer to use. Otherwise resorts to normal for 1d and kaiming for 2+d
            
        Returns:
            Array: The initialized parameter array"""
        
        if "initializer" not in param:
            initializer = INITIALIZERS[default or "he_normal"] if len(param["shape"])>1 else INITIALIZERS[default or "normal"]
        else:
            initializer = INITIALIZERS[param["initializer"]]
        return initializer(key, param["shape"].unbox(default=1), param["dtype"])
    
    def init_params(self, key: Array, default: str = None) -> list[Array]:
        """Initializes all of the parameters created during execution
        
        Parameters:
            key (Array): The random key
            default (str): The default initializer to use. Otherwise resorts to normal for 1d and kaiming for 2+d
        
        Returns:
            list[Array]: A list of all of the created parameters, corresponding to the parameter specifications in self.params"""
        
        subkeys = random.split(key,len(self.params))
        params = [self.init_param(k,p,default=default) for k,p in zip(subkeys, self.params)]
        return params 
    
STATE_TEMPLATES = {"push": {"exec": [], "int": [], "float": []},
                   "jax": {"exec": [], "int": [], "float": [], "int_jax_expr": [], "float_jax_expr": []},
                   "eager": {"exec": [], "int": [], "float": [], "int_jax": [], "float_jax": []}}
"""Various templates to facilitate the creation of new PushStates."""
    
def make_state(input: list, template: str = None, stacks: dict[str,list] = None, params: list = None) -> PushState:
    """Utility function for making a PushState
    
    Parameters:
        input (list): A list of specifications of the inputs to the neural network
        template (str): The name of an optional template to start from. 
            Currently supports `push` (pure push), `jax` (graph computation), and `eager` (eager computation)
        stacks (dict[str,list]): Overrides to the template, manually specifying additional/modified stacks
    
    Returns:
        PushState: The resulting PushState"""
    
    state = {} if template is None else template 
    state.update(stacks)
    params = [] if params is None else params
    return PushState(**state).initialize(params, input)

