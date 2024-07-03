from .dag.expr import Expression
from .dag.shape import Shape, SizePlaceholder, broadcast, mmul, conv 
from itertools import chain 
from typing import List, Union, Callable 
from functools import wraps 

def autopromote(dtypes: list[str]):
    "Automatic promotion of ints to floats"
    if all([d=="int" for d in dtypes]):
        return "int"
    else:
        return "float"

def process_args(*args: Expression, **kwargs: Expression) -> tuple[List[Shape],List[str]]:
    "Grab the shapes and datatypes of the arguments, converting unset SizePlaceholders to 1s as necessary" # ???
    args = list(args)
    args.extend(chain.from_iterable(kwargs.values()))

    def get_size(s: Union[SizePlaceholder,int]):
        # if isinstance(s,SizePlaceholder):
        #     return s.value or 1
        return s 
    
    ret = ([Shape(*[get_size(a) for a in arg.shape]) for arg in args], [arg.dtype for arg in args])
    
    return ret 


def broadcast_signature(*args: Expression, **kwargs: Expression) -> tuple[Shape, str]:
    "Returns the output shape and datatype of a broadcasted elementwise operation given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    shape = broadcast(*shapes)
    # print(f"broadcast shapes {[s.unbox(default=-1) for s in shapes]} shape {shape}")

    # Automatic promotion to float
    dtype = autopromote(dtypes)

    # Incompatible shapes
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
    
    # Automatic promotion to float
    dtype = autopromote(dtypes)

    # Incompatible shapes
    if shape is None:
        return None, dtype

    # Link together the sizes corresponding to the "middle" size in matrix multiplication, as they must be equal
    p1 = shapes[0][-1 if len(shapes[0])>1 else 0] # len(args[0])?
    p2 = shapes[1][-2 if len(shapes[1])>1 else 0] # len(args[1])?
    SizePlaceholder.link_sizes(p1,p2)

    # Set placeholders
    shapes[0][:-2].set_placeholders(shape[:-2])
    shapes[1][:-2].set_placeholders(shape[:-2])

    return shape, dtype 

def conv_signature(*args: Expression, stride=None, padding=None, lhs=None, rhs=None, **kwargs: Expression) -> tuple[Shape, str]:
    "Returns the output shape and datatype of a generalized n-dimensional convolution given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    if len(shapes)!=2:
        raise ValueError("conv_signature takes exactly two shape arguments")
    shape1, shape2 = shapes

    # Add and remove a batch dimension
    shape = conv(shape1.cons(1), shape2, stride=stride, padding=padding, lhs_dilation=lhs, rhs_dilation=rhs)

    final_shape = None if shape is None else shape.unbox(default=-1)
    # Automatic promotion to float
    dtype = autopromote(dtypes)

    # Incompatible shapes
    if shape is None:
        return None, dtype
    shape = shape[1:]

    # Link together the sizes that must be equal
    SizePlaceholder.link_sizes(shape1[0],shape2[1])

    return shape, dtype  

def pool_signature(*args: Expression, dimensions=None, stride=None, padding=None, base_dilation=None, window_dilation=None, **kwargs: Expression) -> tuple[Shape, str]:
    "Returns the output shape and datatype of a pooling operation given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    shape,dtype = shapes[0],dtypes[0]
    if len(dimensions)==0:
        return shape, dtype 
    
    shape = conv(*shapes, Shape(shape[:-len(dimensions)],*dimensions), stride=stride, padding=padding, lhs_dilation=base_dilation, rhs_dilation=window_dilation)

    if shape is None:
        return None, dtype 
    
    return shape, dtype


def aggregate_signature(*args: Expression, axis: Union[int, tuple[int]] = None, **kwargs: Expression):
    "Returns the output shape and datatype of an aggregating operation like `mean()` given the input shapes and datatypes"
    shapes, dtypes = process_args(*args, **kwargs)
    if len(shapes)>1 or len(dtypes)>1:
        raise ValueError("Can only calculate the aggregate signature of one array at a time")
    
    shape,dtype = shapes[0],dtypes[0]
    
    # Aggregate entire array
    if axis is None:
        return Shape(),dtype
    
    # Aggregate single axis
    if isinstance(axis, int):
        if axis>=len(shape):
            return None 
        return shape[:axis]+shape[axis+1:], dtype
    
    # Aggregate multiple axes
    else:
        axis = set(axis)
        ret = [shape[i] for i in range(len(shape)) if i not in axis]
        return Shape(*ret), dtype
    

def cast_signature(*args: Expression, cast_to: str = None, **kwargs: Expression):
    "Returns the same shape and datatype as the input array, and optionally casting it to a different datatype"
    shapes, dtypes = process_args(*args, **kwargs)
    if len(shapes)>1 or len(dtypes)>1:
        raise ValueError("Can only calculate the cast signature of one array at a time")
    
    # Optional type cast
    if cast_to is None:
        dtype = dtypes[0] 
    else:
        dtype = cast_to 
    
    # Same shape
    shape = shapes[0] 

    return shape, dtype 

def flatten_signature(*args: Expression, ndim=1, **kwargs: Expression):
    "Returns the same datatype as the input array, but flattened to an ndim-dimensional array"
    shapes, dtypes = process_args(*args, **kwargs)
    if len(shapes)>1 or len(dtypes)>1:
        raise ValueError("Can only calculate the flattened signature of one array at a time")
    if ndim<1:
        raise ValueError("Final dimensionality must be at least 1")
    
    # Same shape
    shape,dtype = shapes[0],dtypes[0]

    total = 1
    for d in shape[ndim-1:]:
        # Currently does not support size placeholders, as it would require some complex symbolic algebra
        if isinstance(d,SizePlaceholder) and not d.is_set():
            return None, dtype 
        else:
            total*=d 
    
    return shape[:ndim-1].conj(total), dtype 

# def reshape_signature(*args: Expression, cast_to: str = None, **kwargs: Expression):
#     "Returns the same datatype as the input array, with optional reshaping"
#     shapes, dtypes = process_args(*args, **kwargs)
#     if len(shapes)>1 or len(dtypes)>1:
#         raise ValueError("Can only calculate the reshape signature of one array at a time")
    
#     # Optional type cast
#     if cast_to is None:
#         dtype = dtypes[0] 
#     else:
#         dtype = cast_to 
    
#     # Same shape
#     shape = shapes[0] 

#     return shape, dtype 


