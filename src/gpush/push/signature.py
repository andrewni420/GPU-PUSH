from .dag.expr import Expression
from .dag.shape import Shape, SizePlaceholder, broadcast, mmul, conv 
from itertools import chain 
from typing import List, Union

def process_args(*args: Expression, **kwargs: Expression) -> tuple[List[Shape],List[str]]:
    "Grab the shapes and datatypes of the arguments, converting unset SizePlaceholders to 1s as necessary"
    args = list(args)
    args.extend(chain.from_iterable(kwargs.values()))

    def get_size(s: Union[SizePlaceholder,int]):
        if isinstance(s,SizePlaceholder):
            return s.value or 1
        return s 
    
    ret = ([Shape(*[get_size(a) for a in arg.shape]) for arg in args], [arg.dtype for arg in args])

    # print(f"RET {ret} args {[arg.shape for arg in args]}")
    
    return ret 


def broadcast_signature(*args: Expression, **kwargs: Expression) -> tuple[Shape, str]:
    "Returns the output shape and datatype of a broadcasted elementwise operation given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    shape = broadcast(*shapes)

    if all([d=="int" for d in dtypes]):
        dtype = "int"
    else:
        dtype = "float"

    if shape is None:
        return None, dtype

    # Set placeholders
    for s in shapes:
        s.set_placeholders(shape)

    return shape, dtype 

def mmul_signature(*args: Expression, **kwargs: Expression) -> tuple[Shape, str]:
    "Returns the output shape and datatype of a matrix multiplication operation given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    shape = mmul(*shapes)
    

    if all([d=="int" for d in dtypes]):
        dtype = "int"
    else:
        dtype = "float"

    if shape is None:
        return None, dtype

    # Link together the sizes corresponding to the "middle" size in matrix multiplication, as they must be equal
    p1 = shapes[0][-1 if len(args[0])>1 else 0]
    p2 = shapes[1][-2 if len(args[1])>1 else 0]
    SizePlaceholder.link_sizes(p1,p2)

    shapes[0][:-2].set_placeholders(shape[:-2])
    shapes[1][:-2].set_placeholders(shape[:-2])

    return shape, dtype 

def conv_signature(*args: Expression, stride=None, padding=None, lhs=None, rhs=None, **kwargs: Expression) -> tuple[Shape, str]:
    "Returns the output shape and datatype of a generalized n-dimensional convolution given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    shape = conv(*shapes, stride=stride, padding=padding, lhs_dilation=lhs, rhs_dilation=rhs)

    if all([d=="int" for d in dtypes]):
        dtype = "int"
    else:
        dtype = "float"

    if shape is None:
        return None, dtype
    
    n = len(shape)

    # Link together the sizes corresponding to the "middle" size in matrix multiplication, as they must be equal
    SizePlaceholder.link_sizes(shapes[0][1],shapes[1][1])

    return shape, dtype  

def aggregate_signature(*args: Expression, axis: Union[int, tuple[int]] = None, **kwargs: Expression):
    "Returns the output shape and datatype of an aggregating operation like `mean()` given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    if len(shapes)>1 or len(dtypes)>1:
        raise ValueError("Can only calculate the aggregate signature of one array at a time")
    
    shape,dtype = shapes[0],dtypes[0]
    
    if axis is None:
        return Shape(),dtype
    
    if isinstance(axis, int):
        if axis>=len(shape):
            return None 
        return shape[:axis]+shape[axis+1:]
    else:
        axis = set(axis)
        ret = [shape[i] for i in range(len(shape)) if i not in axis]
        return Shape(*ret)
    
