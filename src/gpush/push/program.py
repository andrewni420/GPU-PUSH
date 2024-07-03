from __future__ import annotations
from .instruction import Instruction, CodeBlockClose
from typing import Union

CodeBlock = Union[Instruction,list["CodeBlock"]]
Program = list[CodeBlock]
Genome = list[Union[Instruction,CodeBlockClose]]