from typing import Callable, Any 
from functools import wraps 

def preprocess(pre_fn: Callable[[Any],tuple[list,dict]], fn: Callable):
    """Returns a wrapped function that uses `pre_fn` to preprocess the args and kwargs before feeding them
    into `fn`"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args, kwargs = pre_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper 
