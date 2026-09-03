"""
cpu_core.py
Models the major functional units of a processor:
    - Register File
    - Main Memory
    - ALU
    - Control Unit
    - Instruction Fetch / Decode / Execute cycle

This is the "single-cycle reference model" used to validate correctness of
instructions before they are run through the pipelined model in
pipeline_simulator.py.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Callable

WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1


def to_signed(value: int, bits: int = WORD_BITS) -> int:
    """Interpret an unsigned machine word as a signed two's-complement integer."""
    value &= (1 << bits) - 1
    if value & (1 << (bits - 1)):
        value -= (1 << bits)
    return value


def to_unsigned(value: int, bits: int = WORD_BITS) -> int:
    return value & ((1 << bits) - 1)


class RegisterFile:
    """32 general purpose registers, R0 is hard-wired to 0 (MIPS-style convention)."""

    def __init__(self, n_regs: int = 32):
        self.n_regs = n_regs
        self.regs = [0] * n_regs

    def read(self, idx: int) -> int:
        return 0 if idx == 0 else to_signed(self.regs[idx])

    def write(self, idx: int, value: int):
        if idx != 0:
            self.regs[idx] = to_unsigned(value)

    def dump(self) -> Dict[str, int]:
        return {f"R{i}": to_signed(v) for i, v in enumerate(self.regs)}


class Memory:
    """Byte-addressable main memory, word aligned access helpers."""

    def __init__(self, size_bytes: int = 4096):
        self.size = size_bytes
        self.mem = bytearray(size_bytes)

    def load_word(self, addr: int) -> int:
        word = int.from_bytes(self.mem[addr:addr + 4], byteorder="little")
        return to_signed(word)

    def store_word(self, addr: int, value: int):
        self.mem[addr:addr + 4] = to_unsigned(value).to_bytes(4, byteorder="little")

    def load_program(self, instructions: List[dict], base_addr: int = 0):
        """Store a list of decoded instruction dicts at successive word addresses
        (used purely for address bookkeeping; execution reads instr objects directly)."""
        self.instructions = instructions
        self.base_addr = base_addr


class ALU:
    """Arithmetic Logic Unit. Reports condition flags: Zero, Negative, Carry, Overflow."""

    def __init__(self):
        self.flags = {"Z": 0, "N": 0, "C": 0, "V": 0}

    def _set_flags(self, result_unsigned, result_signed):
        self.flags["Z"] = 1 if to_unsigned(result_unsigned) == 0 else 0
        self.flags["N"] = 1 if result_signed < 0 else 0

    def add(self, a: int, b: int) -> int:
        ua, ub = to_unsigned(a), to_unsigned(b)
        raw = ua + ub
        result = to_unsigned(raw)
        signed = to_signed(result)
        self.flags["C"] = 1 if raw > WORD_MASK else 0
        # signed overflow: operands share sign, result differs
        self.flags["V"] = 1 if (~(a ^ b) & (a ^ signed)) >> (WORD_BITS - 1) & 1 else 0
        self._set_flags(result, signed)
        return signed

    def sub(self, a: int, b: int) -> int:
        return self.add(a, -b)

    def bitwise(self, op: str, a: int, b: int) -> int:
        ua, ub = to_unsigned(a), to_unsigned(b)
        if op == "AND":
            r = ua & ub
        elif op == "OR":
            r = ua | ub
        elif op == "XOR":
            r = ua ^ ub
        else:
            raise ValueError(op)
        signed = to_signed(r)
        self._set_flags(r, signed)
        return signed

    def shift(self, op: str, a: int, amt: int) -> int:
        ua = to_unsigned(a)
        if op == "SLL":
            r = to_unsigned(ua << amt)
        elif op == "SRL":
            r = ua >> amt
        elif op == "SRA":
            r = to_unsigned(to_signed(a) >> amt)
        else:
            raise ValueError(op)
        signed = to_signed(r)
        self._set_flags(r, signed)
        return signed


@dataclass
class Instruction:
    op: str                     # mnemonic e.g. ADD, SUB, LW, SW, BEQ, JMP, NOP
    rd: int = None
    rs1: int = None
    rs2: int = None
    imm: int = None
    label: str = None           # for branch/jump target resolution
    raw: str = ""               # original assembly text, for tracing

    def __repr__(self):
        return self.raw or self.op


class ControlUnit:
    """Decodes an Instruction into control signals consumed by the datapath."""

    SIGNAL_TABLE = {
        "ADD":  dict(alu_op="ADD", reg_write=True,  mem_read=False, mem_write=False, branch=False, alu_src="reg"),
        "SUB":  dict(alu_op="SUB", reg_write=True,  mem_read=False, mem_write=False, branch=False, alu_src="reg"),
        "AND":  dict(alu_op="AND", reg_write=True,  mem_read=False, mem_write=False, branch=False, alu_src="reg"),
        "OR":   dict(alu_op="OR",  reg_write=True,  mem_read=False, mem_write=False, branch=False, alu_src="reg"),
        "ADDI": dict(alu_op="ADD", reg_write=True,  mem_read=False, mem_write=False, branch=False, alu_src="imm"),
        "LW":   dict(alu_op="ADD", reg_write=True,  mem_read=True,  mem_write=False, branch=False, alu_src="imm"),
        "SW":   dict(alu_op="ADD", reg_write=False, mem_read=False, mem_write=True,  branch=False, alu_src="imm"),
        "BEQ":  dict(alu_op="SUB", reg_write=False, mem_read=False, mem_write=False, branch=True,  alu_src="reg"),
        "BNE":  dict(alu_op="SUB", reg_write=False, mem_read=False, mem_write=False, branch=True,  alu_src="reg"),
        "JMP":  dict(alu_op="NOP", reg_write=False, mem_read=False, mem_write=False, branch=True,  alu_src="none"),
        "NOP":  dict(alu_op="NOP", reg_write=False, mem_read=False, mem_write=False, branch=False, alu_src="none"),
    }

    def decode(self, instr: Instruction) -> dict:
        if instr.op not in self.SIGNAL_TABLE:
            raise ValueError(f"Unknown opcode {instr.op}")
        return dict(self.SIGNAL_TABLE[instr.op])


class SingleCycleCPU:
    """Reference (non-pipelined) CPU used to validate correctness of a program
    before it is timed on the pipelined model. Executes strictly one
    instruction at a time through Fetch -> Decode -> Execute -> Memory -> Writeback."""

    def __init__(self, program: List[Instruction], mem_size=4096):
        self.program = program
        self.pc = 0
        self.rf = RegisterFile()
        self.mem = Memory(mem_size)
        self.alu = ALU()
        self.cu = ControlUnit()
        self.trace = []
        self.cycles = 0
        self.label_map = self._resolve_labels(program)

    @staticmethod
    def _resolve_labels(program):
        return {instr.label: i for i, instr in enumerate(program) if instr.label}

    def step(self) -> bool:
        """Execute one instruction. Returns False if PC ran off the end (halt)."""
        if self.pc >= len(self.program):
            return False
        instr = self.program[self.pc]
        signals = self.cu.decode(instr)

        # Fetch
        cycle_log = {"cycle": self.cycles, "pc": self.pc, "instr": repr(instr)}

        # Decode (read registers)
        rs1_val = self.rf.read(instr.rs1) if instr.rs1 is not None else 0
        rs2_val = self.rf.read(instr.rs2) if instr.rs2 is not None else 0
        alu_b = instr.imm if signals["alu_src"] == "imm" else rs2_val

        # Execute
        result = None
        branch_taken = False
        next_pc = self.pc + 1
        if signals["alu_op"] == "ADD":
            result = self.alu.add(rs1_val, alu_b)
        elif signals["alu_op"] == "SUB":
            result = self.alu.sub(rs1_val, alu_b)
        elif signals["alu_op"] == "AND":
            result = self.alu.bitwise("AND", rs1_val, alu_b)
        elif signals["alu_op"] == "OR":
            result = self.alu.bitwise("OR", rs1_val, alu_b)
        elif signals["alu_op"] == "NOP":
            result = 0

        if instr.op == "BEQ" and self.alu.flags["Z"] == 1:
            branch_taken = True
        elif instr.op == "BNE" and self.alu.flags["Z"] == 0:
            branch_taken = True
        elif instr.op == "JMP":
            branch_taken = True

        if branch_taken and instr.label in self.label_map:
            next_pc = self.label_map[instr.label]

        # Memory
        mem_result = None
        if signals["mem_read"]:
            mem_result = self.mem.load_word(result)
        if signals["mem_write"]:
            self.mem.store_word(result, rs2_val)

        # Writeback
        if signals["reg_write"] and instr.rd is not None:
            wb_value = mem_result if signals["mem_read"] else result
            self.rf.write(instr.rd, wb_value)

        cycle_log.update(alu_result=result, branch_taken=branch_taken, next_pc=next_pc)
        self.trace.append(cycle_log)
        self.cycles += 1
        self.pc = next_pc
        return True

    def run(self, max_cycles=10000):
        while self.step() and self.cycles < max_cycles:
            pass
        return self.trace


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Small self-test program: compute sum = 5 + 3, then store/load it back.
    program = [
        Instruction("ADDI", rd=1, rs1=0, imm=5, raw="ADDI R1, R0, 5"),
        Instruction("ADDI", rd=2, rs1=0, imm=3, raw="ADDI R2, R0, 3"),
        Instruction("ADD",  rd=3, rs1=1, rs2=2, raw="ADD  R3, R1, R2"),
        Instruction("SW",   rs2=3, rs1=0, imm=100, raw="SW   R3, 100(R0)"),
        Instruction("LW",   rd=4, rs1=0, imm=100, raw="LW   R4, 100(R0)"),
    ]
    cpu = SingleCycleCPU(program)
    cpu.run()
    print("Register file after run:", {k: v for k, v in cpu.rf.dump().items() if v != 0})
    assert cpu.rf.read(3) == 8
    assert cpu.rf.read(4) == 8
    print("SingleCycleCPU self-test PASSED")
