from __future__ import annotations
import sympy 
from sympy import expand
import sympy.abc as abc 
from dataclasses import dataclass

def broadcast(*shapes: Shape):
    """Tests whether the given shapes can be broadcasted together following the numpy broadcasting rules.
    \nReturns the final broadcasted shape if possible, or None if impossible"""
    final_shape = []
    maxdim = max([len(s) for s in shapes])

    # Check backwards, and at each dimension check that any shapes with size greater than 1 at that dimension are all equal
    for i in range(-1,-maxdim-1,-1):
        cur_size=1 
        for s in shapes:
            if cur_size==1:
                cur_size=s.get(i,1)
            else:
                axis = s.get(i,1)
                # Found unequal sizes that are both not equal to 1, so the given shapes are unbroadcastable.
                if axis!=1 and cur_size!=axis:
                    return None 
                
        # Build up the broadcasted shape in reverse
        final_shape.append(cur_size)

    # Reverse `final_shape` to get the actual broadcasted shape
    return Shape(*(final_shape[::-1]))

def mmul(shape1: Shape, shape2: Shape):
    """Computes the resultant shape of matrix multiplying the two given shapes. 
    \nFollows same rules as `np.matmul`. 
    \nReturns `None` if the shapes are not compatible"""
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
    if shape1[-1]!=shape2[-2]:
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

@dataclass(frozen=True, init=False)
class Shape(tuple):
    """A tuple subclass that holds the shape of a matrix. Supports shape-specific operations such as 
    checking broadcastability and inferring dimensions. Supports some clojure-like syntax"""

    def __new__(cls, *args):
        return super().__new__(cls, args)
    
    def __str__(self):
        return f"Shape({', '.join([str(i) for i in self])})"
    
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


