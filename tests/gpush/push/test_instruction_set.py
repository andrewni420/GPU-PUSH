from gpush.push.instruction_set import InstructionSet
from collections import Counter
import jax.numpy as jnp

def test_dict():
    "Test common dictionary operations"
    instr = InstructionSet(one=1,two=2)
    assert len(instr)==2 and instr["one"]==1 and instr["two"]==2 
    instr["three"]=3 
    assert len(instr)==3 and instr["three"]==3 

def test_sample():
    "Tests the sampling of instructions"
    instr = InstructionSet(one="one",two="two",three="three")
    sample = Counter([instr.sample() for _ in range(1000)])
    sample = [s/1000 for s in sample.values()]
    assert jnp.mean((jnp.array(sample)-1/3)**2)<0.005

    instr.logprobs["one"]=1
    instr.logprobs["two"]=2
    instr.logprobs["three"]=3
    instr.updated=True 
    sample = Counter([instr.sample() for _ in range(1000)])
    sample = [sample[k]/1000 for k in ["one", "two", "three"]]
    x = jnp.exp(jnp.arange(3))
    x = x/x.sum()
    assert jnp.mean((jnp.array(sample)-x)**2)<0.005




