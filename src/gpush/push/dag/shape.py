from __future__ import annotations
from typing import List, Union 
from dataclasses import dataclass
from copy import deepcopy

def broadcast(*shapes: Shape):
    """Computes the resultant shape from broadcasting together the two given shapes, following the same rules as numpy. 
    Returns `None` if the shapes are not compatible
    
    Parameters:
        shapes (Shape): The shapes to be broadcasted together
        
    Returns:
        Shape: The resultant shape, or `None` if the operation cannot be performed. """
    final_shape = []
    maxdim = max([len(s) for s in shapes])

    # Check backwards, and at each dimension check that any shapes with size greater than 1 at that dimension are all equal
    for i in range(-1,-maxdim-1,-1):
        cur_size=SizePlaceholder()
        one_found = False 
        for s in shapes:
            if not cur_size.is_set():
                val = s.get(i)
                if isinstance(val,SizePlaceholder):
                    val = val.value
                if val is None:
                    pass 
                elif val==1:
                    one_found = True
                else:
                    cur_size.value = val
            else:
                axis = s.get(i,1)
                if isinstance(axis,SizePlaceholder):
                    axis = axis.value
                # Found unequal sizes that are both not equal to 1, so the given shapes are unbroadcastable.
                if axis is not None and axis!=1 and cur_size.value!=axis:
                    return None
                
        # Build up the broadcasted shape in reverse
        if cur_size.is_set():
            final_shape.append(cur_size.value)
        elif one_found:
            final_shape.append(1)
        else:
            final_shape.append(cur_size)

    # Reverse `final_shape` to get the actual broadcasted shape
    return Shape(*(final_shape[::-1]))

def mmul(shape1: Shape, shape2: Shape):
    """Computes the resultant shape from matrix multiplying the two given shapes, following the same rules as `np.matmul`. 
    Returns `None` if the shapes are not compatible
    
    Parameters:
        shape1 (Shape): The first shape 
        shape2 (Shape): The second shape 
        
    Returns:
        Shape: The resultant shape, or `None` if the operation cannot be performed."""
    if len(shape1)==0 or len(shape2)==0:
        raise ValueError(f"Input operand does not have enough dimensions: shape1 {shape1} shape2 {shape2}")
    
    # Expand 1-d inputs
    remove_first = False 
    remove_last = False
    if len(shape1)==1:
        shape1 = Shape(1)+shape1
        remove_first = True 
    if len(shape2)==1:
        shape2 = shape2+Shape(1)
        remove_last = True

    # Check for compatible dimensions
    if SizePlaceholder.is_set_size(shape1[-1]) and SizePlaceholder.is_set_size(shape2[-2])  and shape1[-1]!=shape2[-2]:
        return None 
    
    # Attempt to broadcast previous dimensions
    bcast = broadcast(shape1[:-2],shape2[:-2])
    if bcast is None:
        return None 
    
    # Remove previously added dimensions
    res = bcast+Shape(shape1[-2],shape2[-1])
    if remove_first:
        res = res.squeeze(-2)
    if remove_last:
        res = res.squeeze(-1)

    # Return result
    return res 

def conv(shape1: Shape, 
         shape2: Shape, 
         stride: List[int] = None,
         padding: str | List[tuple[int,int]] = None, 
         lhs_dilation: List[int] = None, 
         rhs_dilation: List[int] = None) -> Shape:
    """Computes the resultant shape of a general convolution operation between matrices of the input shapes.
    
    Parameters:
        shape1 (Shape): The first input shape of the form (NCHW...)
        shape2 (Shape): The second input shape of the form (OIHW...)
        stride (list[int]): The stride length in each spatial dimension. 1 by default
        padding (list[tuple[int,int]]): The left/right padding along each spatial dimension. 0 by default
        lhs_dilation (list[int]): The dilation factor of `shape1` along each spatial dimension. 1 by default
        rhs_dilation (list[int]): The dilation factor of `shape2` along each spatial dimension. 1 by default.
        
    Returns:
        Shape: The shape of the resultant matrix (NCHW...), or `None` if a convolution with the given parameters can't be performed. 
        Convolutional operations which result in size-0 axes are considered invalid."""
    #Batch_Size In_Channels Height Width
    #Out_Channels In_Channels Channels Height Width
    #Batch_Size Out_Channels Height Width
    if len(shape1)!=len(shape2):
        return None 
    n = len(shape1)
    if n<2:
        return None 
    # Do the number of channels match?
    if SizePlaceholder.is_set_size(shape1[1]) and SizePlaceholder.is_set_size(shape2[1]) and shape1[1]!=shape2[1]:
        return None 
    
    allow_spatial_placeholders = (padding=="SAME" or (isinstance(padding,list) and all([p=="SAME" for p in padding]))) 

    # All spatial dimensions need to be specified, unless padding is "SAME". Otherwise we'd need to do some fancy arithmetic
    if not allow_spatial_placeholders:
        for i in range(2,n):
            if not (SizePlaceholder.is_set_size(shape1[i]) and SizePlaceholder.is_set_size(shape2[i])):
                return None 
    
    # Compute the final size along each spatial dimension
    def spatial_size(size: int, kernel: int, stride: int, padding: str | tuple[int,int], lhs: int, rhs: int) -> int:
        if padding=="SAME":
            # Pad with enough zeros so that all windows fit fully into padded area, same as jax, xla, or tensorflow (https://github.com/openxla/xla/blob/main/xla/client/padding.cc#L54)
            out_shape = -(size//-stride)
            pad_size = max(0,(out_shape-1)*stride+kernel-size)
            padding = (pad_size//2,pad_size-pad_size//2)
        return max(0,(size-kernel+(lhs-1)*(size-1)-(rhs-1)*(kernel-1)+padding[0]+padding[1])//stride+1)
    
    # Turn parameters into defaults / correct format
    def make_default(x, default, length):
        if isinstance(x,int):
            return [x]*length
        elif x is None or len(x)==0:
            return [default]*length
        else:
            return x 
    
    if padding=="SAME":
        padding = [padding]*(n-2)
    elif isinstance(padding,int):
        padding = [(padding,padding)]*(n-2)
    elif padding is None or len(padding)==0:
        padding = [(0,0)]*(n-2)
    elif isinstance(padding[0],int):
        padding = [padding for _ in range(n-2)]
    stride,lhs_dilation,rhs_dilation = [make_default(x,1,n-2) for x in [stride, lhs_dilation, rhs_dilation]]

    # Compute result (NCHW...)
    ret = Shape(shape1[0], shape2[0])+Shape(*map(spatial_size, shape1[2:], shape2[2:], stride, padding, lhs_dilation, rhs_dilation))

    # If any spatial dimension is 0, then this operation is impossible.
    for i in ret[2:]:
        if i==0:
            return None 
    return ret

def make_placeholder_arithmetic(func):
    """Make arithmetic operations return an unset `SizePlaceholder` if either argument is unset. 
    Also make operations safe for integer arguments"""
    def wrapper(self, other: int | SizePlaceholder):
        if self.is_set():
            other_val = other if isinstance(other, int) else other.value 
            if other_val is not None:
                return func(self.value, other_val)
        return SizePlaceholder()
    return wrapper

class SizePlaceholder():
    """An undetermined value with limited symbolic functionalities. Can be linked or set equal to other
    undetermined values, such that when one `SizePlaceholder`'s value is determined, all linked placeholders
    will be set to the same value."""

    _value: int 
    "The value of the placeholder, or `None` if unset. Do not modify."
    links: list[SizePlaceholder]
    "The linked placeholders, all of which have equal values"

    def __init__(self):
        self._value = None 
        self.links = []

    @property
    def value(self):
        "The value of the placeholder."
        return self._value 

    @value.setter
    def value(self, value: int):
        """Sets the value of this placeholder as well as those of all the linked placeholders"""
        if self._value is None:
            self._value = value 
            for other in self.links:
                other.value = value
        else:
            if value!=self._value:
                raise ValueError("Trying to set an already existing value")
        
    def is_set(self):
        "Has this placeholder's value been set?"
        return self.value is not None 
    
    def link(self, *others: int|SizePlaceholder):
        """Symbolically set equal to other placeholders. If one of the other placeholders already has a set value, 
        all of the placeholders will take on that value. 
        
        Raises:
            ValueError: If two placeholders have already been set to different values"""
        literals = [o for o in others if isinstance(o,int)]
        placeholders = [o for o in others if isinstance(o,SizePlaceholder)]
        
        self.links.extend(placeholders)
        for o in placeholders:
            o.links.append(self)

        if len(literals)>0:
            if len(set(literals))>1:
                raise ValueError("Attempting to link contradictory values")
            else:
                self.value = literals[0]

        if self.is_set():
            for o in placeholders:
                o.value = self.value 
        else:
            for o in placeholders:
                if o.is_set():
                    self.value = o.value  

    def copy(self) -> SizePlaceholder:
        """Return a new `SizePlaceholder` with the same value and linked placeholders. 
        Deep-ish copy in that linked placeholders are not recursively copied."""
        placeholder = SizePlaceholder()
        placeholder._value = self._value
        placeholder.links = [l for l in self.links]
        return placeholder

    @staticmethod
    def link_sizes(*sizes: Union[int,SizePlaceholder]):
        "Same as `SizePlaceholder.link()` but doesn't require a `SizePlaceholder` object"
        for i,s in enumerate(sizes):
            if isinstance(s,SizePlaceholder):
                s.link(*(sizes[:i]+sizes[i+1:]))
                return 
        if len(set(sizes))>1:
            raise ValueError("Attempting to link contradictory values")
        
    @staticmethod
    def is_set_size(other: Union[int,SizePlaceholder]):
        "Same as `SizePlaceholder.is_set()` but doesn't require a `SizePlaceholder` object"
        if isinstance(other,int):
            return True
        return other.is_set()
        
    def __neg__(self):
        if self.is_set():
            return -self.value 
        else:
            return SizePlaceholder()
        
    @make_placeholder_arithmetic
    def __add__(self, other: int | SizePlaceholder):
        return self+other
    @make_placeholder_arithmetic
    def __radd__(self, other: int):
        return other+self
    
    @make_placeholder_arithmetic
    def __mul__(self, other: int | SizePlaceholder):
        return self*other
    @make_placeholder_arithmetic
    def __rmul__(self, other: int | SizePlaceholder):
        return other*self 
        return self.__mul__(other)
    
    @make_placeholder_arithmetic
    def __sub__(self, other: int | SizePlaceholder):
        return self-other 
    @make_placeholder_arithmetic
    def __rsub__(self, other: int):
        return other-self 
    
    @make_placeholder_arithmetic
    def __truediv__(self, other: int | SizePlaceholder):
        return self/other 
    @make_placeholder_arithmetic
    def __rtruediv__(self, other: int):
        return other/self
    
    @make_placeholder_arithmetic
    def __floordiv__(self, other: int | SizePlaceholder):
        return self//other 
    @make_placeholder_arithmetic
    def __rfloordiv__(self, other: int):
        return other//self 
    
    @make_placeholder_arithmetic
    def __ge__(self, other: int | SizePlaceholder):
        return self>=other 
    @make_placeholder_arithmetic
    def __le__(self, other: int | SizePlaceholder):
        return self<=other 
    @make_placeholder_arithmetic
    def __gt__(self, other: int | SizePlaceholder):
        return self>other 
    @make_placeholder_arithmetic
    def __lt__(self, other: int | SizePlaceholder):
        return self<other 

    def __eq__(self, other: int | SizePlaceholder):
        if isinstance(other,SizePlaceholder):
            return self.value==other.value 
        return self.value==other

class Shape(tuple):
    """A tuple subclass that holds the shape of a matrix. Supports shape-specific operations such as 
    checking broadcastability and inferring dimensions. Supports some clojure-like syntax"""

    def __new__(cls, *args):
        return super().__new__(cls, args)
    
    def __deepcopy__(self, d):
        return Shape(*[deepcopy(s,d) for s in self])
    
    def __str__(self):
        return f"Shape({', '.join([str(i) for i in self])})"
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, key):
        """Ensures that slicing returns a Shape object"""
        if isinstance(key, slice):
            return Shape(*super().__getitem__(key))
        else:
            return super().__getitem__(key)
        
    def __add__(self, other: Shape) -> Shape:
        """Ensures that addition returns a Shape object"""
        return Shape(*self,*other)
    
    def __mul__(self, val: int) -> Shape:
        """Ensures that multiplication returns a Shape object"""
        return Shape(*(super().__mul__(val)))
    
    
    def get(self, idx: int, default=None):
        """Dict-like get with default. 
        \nReturns `self[idx]` when possible, i.e. `-len(self)<=idx<len(self)`, and `default` otherwise"""
        if -len(self)<=idx<len(self):
            return self[idx]
        return default 
    
    def assoc(self, idx: int, val) -> Shape:
        """Clojure-like assoc. 
        \nReturns a new Shape object with the entry at index `idx` changed to `val`"""
        if not -len(self)<=idx<len(self):
            raise IndexError(f"Shape index out of range: idx {idx} length {len(self)}")
        idx %= len(self)
        return self[:idx]+Shape(val)+self[idx+1:]

    def conj(self, *val) -> Shape:
        "Conjoins the given values onto the end of `self` and returns the new shape object"
        return self+Shape(*val)
    
    def cons(self, *val) -> Shape:
        "Prepends the given values in that order to the front of `self` and returns the new shape object"
        return Shape(*val)+self 

    def concat(self, other: Shape) -> Shape:
        "Concatenates the two shapes. Same as addition"
        return self+other
    
    def dissoc(self, idx: int)-> Shape:
        "Removes the dimension at index `idx` from the shape object"
        if not -len(self)<=idx<len(self):
            raise IndexError(f"Shape index out of range: idx {idx} length {len(self)}")
        idx %= len(self)
        return self[:idx]+self[idx+1:]
    
    def squeeze(self, idx: int) -> Shape:
        """Remove a dimension that has size 1.
        \nReturns a new Shape object with the requested dimension removed"""
        if self.get(idx,1)!=1:
            raise ValueError(f"Trying to squeeze dimension with size greater than 1: idx {idx} size {self.get(idx,1)}")
        return self.dissoc(idx)
    
    def broadcast(self, *shapes):
        "Tries to broadcast this shape together with the given shapes. Returns None if not possible"
        return broadcast(self, *shapes)

    def mmul(self, other: Shape):
        """Computes the resultant shape of matrix multiplying this shape with the given shape
        \nReturns `None` if the shapes are not compatible"""
        return mmul(self, other)
    
    def is_set(self):
        "Tests whether all size placeholders in this shape have been set"
        for s in self:
            if isinstance(s, SizePlaceholder) and s.value is None:
                return False 
        return True 
    
    def unbox(self, default: int = None):
        """Unboxes all the size placeholders in this shape after they have all been set. 
        If default is not None, replaces unset placeholders with the default"""
        vals = []
        for s in self:
            if isinstance(s,SizePlaceholder):
                if s.is_set():
                    vals.append(s.value)
                else:
                    if default is None:
                        raise RuntimeError("Cannot unbox shape when placeholders are None")
                    else:
                        vals.append(default)
            else:
                vals.append(s)
        return Shape(*vals)
    
    def set_placeholders(self, other: Shape):
        "Set placeholder values based on another shape"
        if other is None:
            return self 
        
        for s,o in zip(self[::-1], other[::-1]):
            if isinstance(s,SizePlaceholder):
                s.link(o)
        return self 
    
    def copy(self) -> Shape:
        "Deep-ish copy of the Shape, copying all SizePlaceholders but not recursively copying linked placeholders."
        return Shape(*[(s.copy() if isinstance(s,SizePlaceholder) else s) for s in self])


