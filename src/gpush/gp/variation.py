from __future__ import annotations
from abc import ABC, abstractmethod
from .individual import Individual
from .population import Population
from typing import List 
import numpy as np 
from scipy.special import softmax



class VariationOperator(ABC):
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, ind: Individual) -> Individual:
        return self.mutate(ind)

    @abstractmethod
    def mutate(self, ind: Individual) -> Individual:
        pass 

    def choose(self, attr, n=0):
        return self.rng.choice(attr[0], p=attr[1],size=n)

class Variation(VariationOperator):
    def __init__(self, operators: List[VariationOperator], logprobs: List[float] = None):
        self.operators = operators 
        self.logprobs = [1]*len(operators) if logprobs is None else logprobs
        self.probs = softmax(self.logprobs)
        self.rng = np.random.default_rng()


    def mutate(self, ind: Individual) -> Individual:
        operator = self.choose([self.operators, self.probs])
        return operator(ind)
    

