from gpush.push.compiler import PlushyCompiler
from gpush.push.instruction import LiteralInstruction,CodeBlockClose


def test_compiler():
    i1 = LiteralInstruction("i1",1,"int")
    i1.code_blocks=1
    i2 = LiteralInstruction("i2",2,"int")
    i2.code_blocks=2
    i3 = LiteralInstruction("i3",3,"int")
    i3.code_blocks=1
    i4 = LiteralInstruction("i4",4,"int")
    i5 = LiteralInstruction("i5",5,"int")
    i6 = LiteralInstruction("i6",6,"int")
    i7 = LiteralInstruction("i7",7,"int")
    i7.code_blocks=1
    i8 = LiteralInstruction("i8",8,"int")
    i9 = LiteralInstruction("i9",9,"int")
    i10 = LiteralInstruction("i10",10,"int")
    program = [i1,CodeBlockClose(),i2,i3,i4,CodeBlockClose(),i5,CodeBlockClose(),i6,CodeBlockClose(),i7,i8,i9,CodeBlockClose(),i10]

    compiler = PlushyCompiler()
    push = compiler(program)
    res = [i1, [], i2, [[i3, [i4], i5], i6], i7, [i8, i9], i10]
    assert push==res 




