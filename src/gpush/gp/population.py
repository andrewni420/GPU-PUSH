from abc import ABC, abstractmethod
from typing import List, Callable, Any
from .individual import Individual
import numpy as np

class Population(List[Individual], ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.updated = False 
        self._avg_fitness = None 
        self._min_fitness = None
        self._behavior_diversity = None 

    def apply(self, mapper: Callable[[Individual], Any] = None, reducer: Callable[[List[Individual]], Any] = None):
        pop = self if mapper is None else [mapper(i) for i in self]
        return pop if reducer is None else reducer(pop)
    
    def __getitem__(self, key):
        """Ensures that slicing returns a Shape object"""
        if isinstance(key, slice):
            return Population(*super().__getitem__(key))
        else:
            return super().__getitem__(key)
        
    @property
    def avg_fitness(self):
        if self.updated or (self._avg_fitness is None):
            self._avg_fitness = self.apply(mapper = lambda x:x.fitness, reducer = lambda x:np.mean(np.sum(x,axis=-1)))
            self.updated = False 
        return self._avg_fitness
    
    @property
    def behavior_diversity(self):
        if self.updated or (self._behavior_diversity is None):
            self._behavior_diversity = self.apply(mapper = lambda x:x.fitness, reducer = lambda x:len(np.unique(x, axis=0)))
            self.updated = False 
        return self._behavior_diversity
    
    @property 
    def min_fitness(self):
        if self.updated or (self._min_fitness is None):
            self._min_fitness = self.apply(mapper = lambda x:x.fitness, reducer = lambda x:np.min(np.sum(x,axis=-1)))
        return self._min_fitness
    
    
        
    


