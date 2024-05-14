from abc import ABC, abstractmethod
from .instruction import CodeBlockClose, Instruction
from typing import Union

class Compiler():
    @abstractmethod
    def compile(self, program: list):
        pass 

    def __call__(self, program: list):
        return self.compile(program)
    
class PlushyCompiler(Compiler):
    def compile(self, program: list[Union[Instruction, CodeBlockClose]]):
        ret = []
        cur_depth=0
        for instr in program:
            if isinstance(instr,CodeBlockClose):
                cur_depth = max(0,cur_depth-1)
            else:
                r = ret 
                for _ in range(cur_depth):
                    r = r[-1]
                r.append(instr)
                opened_blocks = []
                o = opened_blocks
                for _ in range(instr.code_blocks):
                    o.append([])
                    o=o[-1]
                r.extend(opened_blocks)
                cur_depth+=instr.code_blocks
        return ret 
            