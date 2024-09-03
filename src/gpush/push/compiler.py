from abc import ABC, abstractmethod
from .instruction import CodeBlockClose, Instruction
from typing import Union
from .instruction_set import GLOBAL_INSTRUCTIONS

Genome = list[Union[Instruction, CodeBlockClose, "Genome", str]]
Program = list[Union[Instruction, "Program"]]

class Compiler():
    "A compiler that turns genomes into programs. Can directly call or use `Compiler.compile()`"

    @abstractmethod
    def compile(self, genome: list) -> list:
        pass 

    def __call__(self, genome: list) -> list:
        "Compiles a genome into a program"
        return self.compile(genome)
    
class PlushyCompiler(Compiler):
    "A compiler that turns plushy genomes into push programs"

    def compile(self, genome: Genome) -> Program:
        "Compiles a plushy genome into a push program"
        ret = []
        cur_depth=0
        for instr in genome:
            if isinstance(instr,CodeBlockClose) or (isinstance(instr,str) and instr.lower()=="close"):
                cur_depth = max(0,cur_depth-1)
            else:
                if isinstance(instr,str):
                    instr=GLOBAL_INSTRUCTIONS[instr]
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
            