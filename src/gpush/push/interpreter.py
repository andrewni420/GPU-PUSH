from .state import PushState
from .instruction import Instruction
from copy import deepcopy

class Interpreter():
    def __init__(self, initial_state: PushState = None, max_steps=None):
        self.initial_state = initial_state or PushState()
        self.state = deepcopy(self.initial_state)
        self.max_steps=max_steps or float('inf')

    def reset(self):
        self.state = deepcopy(self.initial_state)

    def run(self, program: list, state: PushState = None, max_steps=None, out_stacks=None):
        self.state = state or self.state 
        self.max_steps = max_steps or self.max_steps

        state["exec"].extend(program)
        while len(state["exec"])>0 and self.state.nsteps<self.max_steps:
            self.step(state)
        return self.state if out_stacks is None else self.state.observe(out_stacks)

    def step(self):
        next_instr = self.state["exec"].pop()
        if isinstance(next_instr,list):
            self.state["exec"].extend(next_instr)
        elif isinstance(next_instr, Instruction):
            self.state = next_instr(self.state)
        else:
            raise NotImplementedError(f"Unrecognized item on the exec stack: {next_instr}")
        