from __future__ import annotations
from abc import ABC, abstractmethod
from .individual import Individual
from .population import Population
from typing import List 
import numpy as np 
from scipy.special import softmax
from argmap import Argmap 

class VariationOperator(ABC):
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, ind: Individual) -> Individual:
        return self.mutate(ind)

    @abstractmethod
    def mutate(self, ind: Individual, argmap: Argmap) -> Individual:
        pass 

    def choose(self, attr, n=0):
        return self.rng.choice(attr[0], p=attr[1],size=n)

class UMAD(VariationOperator):
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob=prob
        self.del_prob = prob/(1+prob)

    def mutate(self, ind: Individual, argmap: Argmap, id=0) -> Individual:
        instr = argmap["instruction_set"]
        n = len(ind.genome)
        res = []
        for g,p,e in zip(ind.genome,self.rng.uniform(0,1,size=(n,)),instr.sample(n)):
            if (p<=self.prob):
                res.extend(self.rng.shuffle([g,e]))
            else:
                res.append(g)
        res = [g for g,p in zip(res,self.rng.uniform(0,1,size=(len(res),))) if p>self.del_prob]
        return Individual(res,id=id,parent=ind.id)

class Variation(VariationOperator):
    def __init__(self, operators: List[VariationOperator], logprobs: List[float] = None):
        self.operators = operators 
        self.logprobs = [1]*len(operators) if logprobs is None else logprobs
        self.probs = softmax(self.logprobs)
        self.rng = np.random.default_rng()

    def mutate(self, ind: Individual) -> Individual:
        operator = self.choose([self.operators, self.probs])
        return operator(ind)
    

