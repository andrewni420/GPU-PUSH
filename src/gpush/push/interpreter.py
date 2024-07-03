from .state import PushState
from .instruction import Instruction
from copy import deepcopy
from .program import Program

class Interpreter():
    def __init__(self, initial_state: PushState = None, max_steps=None):
        self.initial_state = initial_state or PushState()
        self.state = deepcopy(self.initial_state)
        self.max_steps=max_steps or float('inf')

    def reset(self):
        self.state = deepcopy(self.initial_state)

    def reverse(self, program: Program):
        return [(self.reverse(p) if isinstance(p,list) else p) for p in program][::-1]

    def run(self, program: Program, state: PushState = None, max_steps=None, out_stacks=None):
        self.state = state or self.state 
        self.max_steps = max_steps or self.max_steps

        state["exec"].extend(self.reverse(program))
        while len(state["exec"])>0 and self.state.nsteps<self.max_steps:
            self.step(state)
        return self.state if out_stacks is None else self.state.observe(out_stacks)

    def step(self, state: PushState = None):
        state = state or self.state
        next_instr = state["exec"].pop()
        if isinstance(next_instr,list):
            state["exec"].extend(next_instr)
        elif isinstance(next_instr, Instruction):
            state = next_instr(state)
        else:
            raise NotImplementedError(f"Unrecognized item on the exec stack: {next_instr}")
    

        