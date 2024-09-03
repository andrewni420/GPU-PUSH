from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List 
from copy import deepcopy

class Individual(ABC):
    def __init__(self, genome: List, id: int = 0, parent: int = None ):
        self.genome = genome 
        self.params = None 
        self.fn = None 
        self.fitness = None 
        self.id = id
        self.parent = parent 

    def copy(self, id=0):
        ind = Individual(self.genome, id=id,parent=self.id)
        ind.params = self.params 
        return ind 
    
    def __call__(self, inputs):
        return self.fn(self.params, inputs)
    

    