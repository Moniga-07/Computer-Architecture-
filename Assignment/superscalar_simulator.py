"""
superscalar_simulator.py
Simplified, teaching-oriented models of three advanced execution techniques,
built on top of the same Instruction objects used elsewhere, so they can be
benchmarked against the in-order 5-stage pipeline on identical programs.

    1. Out-of-Order Execution (OoO)   - a scoreboard-style model: an
       instruction issues as soon as its operands are ready, independent of
       program order, respecting a limited number of functional units.
    2. Speculative Execution           - branches are predicted; instructions
       after a branch begin executing speculatively; on a misprediction all
       speculative work is squashed and refetched (this is where the model
       differs from the in-order pipeline's fixed bubble cost).
    3. Simultaneous Multithreading (SMT) - two independent instruction
       streams (threads) share the same superscalar issue width per cycle,
       improving functional-unit utilization when one thread stalls.

These are intentionally simplified (single-cycle EX, no full dependency
graph solver) so that the *relative* trends they expose — issue-width
utilization, misprediction cost, thread-interleaving throughput gains — are
correct and easy to validate, even though they are not cycle-accurate
models of a real Tomasulo/ROB implementation.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict
from cpu_core import Instruction


@dataclass
class OoOStats:
    total_cycles: int = 0
    instructions: int = 0
    issue_width: int = 0
    avg_ipc: float = 0.0
    functional_unit_busy_cycles: int = 0
    utilization: float = 0.0


class OutOfOrderSimulator:
    """Scoreboard-style OoO: each cycle, up to `issue_width` ready instructions
    (all register operands already produced) issue to available functional
    units; latency is 1 cycle per instruction (execute) here for simplicity,
    with results available to dependents the following cycle."""

    def __init__(self, program: List[Instruction], issue_width: int = 2, num_fu: int = 2):
        self.program = program
        self.issue_width = issue_width
        self.num_fu = num_fu

    def run(self):
        n = len(self.program)
        ready_cycle = [None] * n         # cycle at which each instr's result becomes available
        issued_cycle = [None] * n
        reg_ready_cycle: Dict[int, int] = {}   # register -> cycle its value became available (0 = always ready)

        cycle = 0
        completed = 0
        fu_busy_cycles = 0
        remaining = list(range(n))

        while completed < n and cycle < n * 4 + 20:
            cycle += 1
            issued_this_cycle = 0
            still_remaining = []
            for idx in remaining:
                instr = self.program[idx]
                deps = [r for r in (instr.rs1, instr.rs2) if r is not None and r != 0]
                deps_ready = all(reg_ready_cycle.get(r, 0) < cycle for r in deps)
                if deps_ready and issued_this_cycle < min(self.issue_width, self.num_fu):
                    issued_cycle[idx] = cycle
                    ready_cycle[idx] = cycle + 1     # result available next cycle
                    if instr.rd is not None:
                        reg_ready_cycle[instr.rd] = cycle + 1
                    issued_this_cycle += 1
                    fu_busy_cycles += 1
                    completed += 1
                else:
                    still_remaining.append(idx)
            remaining = still_remaining

        stats = OoOStats(
            total_cycles=cycle,
            instructions=n,
            issue_width=self.issue_width,
            avg_ipc=n / cycle if cycle else 0,
            functional_unit_busy_cycles=fu_busy_cycles,
            utilization=fu_busy_cycles / (cycle * self.num_fu) if cycle else 0,
        )
        return stats, issued_cycle


@dataclass
class SpeculativeStats:
    total_cycles: int = 0
    instructions: int = 0
    branches: int = 0
    mispredictions: int = 0
    squashed_instructions: int = 0
    misprediction_penalty: int = 0


class SpeculativeExecutionSimulator:
    """Models static branch prediction with squash-on-misprediction. Each
    branch instruction carries an `actual_taken` flag (ground truth); the
    predictor always predicts 'taken'. On a correct prediction there is no
    bubble; on a misprediction, `penalty` cycles are lost and any
    instructions fetched speculatively past the branch are discarded."""

    def __init__(self, program: List[Instruction], branch_outcomes: Dict[int, bool],
                 penalty: int = 3, speculative_window: int = 3):
        self.program = program
        self.branch_outcomes = branch_outcomes   # index -> actual outcome (True=taken)
        self.penalty = penalty
        self.speculative_window = speculative_window

    def run(self):
        n = len(self.program)
        cycle = 0
        squashed = 0
        mispredictions = 0
        branches = 0
        i = 0
        while i < n:
            instr = self.program[i]
            cycle += 1
            if instr.op in ("BEQ", "BNE", "JMP"):
                branches += 1
                predicted_taken = True   # static "always predict taken" policy
                actual_taken = self.branch_outcomes.get(i, instr.op == "JMP")
                if predicted_taken != actual_taken:
                    mispredictions += 1
                    cycle += self.penalty
                    squashed += min(self.speculative_window, n - i - 1)
            i += 1

        return SpeculativeStats(
            total_cycles=cycle,
            instructions=n,
            branches=branches,
            mispredictions=mispredictions,
            squashed_instructions=squashed,
            misprediction_penalty=self.penalty,
        )


@dataclass
class SMTStats:
    total_cycles: int = 0
    thread_a_instructions: int = 0
    thread_b_instructions: int = 0
    combined_ipc: float = 0.0
    issue_width: int = 0
    slot_utilization: float = 0.0


class SMTSimulator:
    """Two independent threads share one superscalar issue width per cycle.
    Each cycle, available issue slots are filled greedily from whichever
    thread(s) still have ready instructions, demonstrating how SMT keeps
    functional units busy when one thread stalls (e.g. on a dependency)."""

    def __init__(self, thread_a: List[Instruction], thread_b: List[Instruction], issue_width: int = 2):
        self.thread_a = thread_a
        self.thread_b = thread_b
        self.issue_width = issue_width

    def run(self):
        na, nb = len(self.thread_a), len(self.thread_b)
        ia, ib = 0, 0
        cycle = 0
        used_slots = 0
        while ia < na or ib < nb:
            cycle += 1
            slots = self.issue_width
            # alternate priority each cycle to model fair round-robin scheduling
            order = ("A", "B") if cycle % 2 == 0 else ("B", "A")
            for thread in order:
                if slots == 0:
                    break
                if thread == "A" and ia < na:
                    ia += 1
                    slots -= 1
                    used_slots += 1
                elif thread == "B" and ib < nb:
                    ib += 1
                    slots -= 1
                    used_slots += 1

        total_instrs = na + nb
        return SMTStats(
            total_cycles=cycle,
            thread_a_instructions=na,
            thread_b_instructions=nb,
            combined_ipc=total_instrs / cycle if cycle else 0,
            issue_width=self.issue_width,
            slot_utilization=used_slots / (cycle * self.issue_width) if cycle else 0,
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Program with independent instructions (good OoO candidate): computing
    # several unrelated sums that could issue in parallel.
    program = [
        Instruction("ADDI", rd=1, rs1=0, imm=5,  raw="ADDI R1,R0,5"),
        Instruction("ADDI", rd=2, rs1=0, imm=7,  raw="ADDI R2,R0,7"),
        Instruction("ADD",  rd=3, rs1=1, rs2=2,  raw="ADD  R3,R1,R2"),   # depends on R1,R2
        Instruction("ADDI", rd=4, rs1=0, imm=9,  raw="ADDI R4,R0,9"),
        Instruction("ADDI", rd=5, rs1=0, imm=2,  raw="ADDI R5,R0,2"),
        Instruction("ADD",  rd=6, rs1=4, rs2=5,  raw="ADD  R6,R4,R5"),
        Instruction("ADD",  rd=7, rs1=3, rs2=6,  raw="ADD  R7,R3,R6"),
    ]

    print("=== Out-of-Order Execution (issue_width=2, 2 functional units) ===")
    ooo = OutOfOrderSimulator(program, issue_width=2, num_fu=2)
    stats, issued = ooo.run()
    print(f"Cycles={stats.total_cycles} IPC={stats.avg_ipc:.2f} FU utilization={stats.utilization:.1%}")
    for idx, c in enumerate(issued):
        print(f"  {program[idx]!r:25s} issued at cycle {c}")

    print("\n=== Speculative Execution (branch misprediction cost) ===")
    branch_program = [
        Instruction("ADDI", rd=1, rs1=0, imm=1, raw="ADDI R1,R0,1"),
        Instruction("BEQ",  rs1=1, rs2=0, label="X", raw="BEQ R1,R0,X"),
        Instruction("ADDI", rd=2, rs1=0, imm=2, raw="ADDI R2,R0,2"),
        Instruction("ADDI", rd=3, rs1=0, imm=3, raw="ADDI R3,R0,3"),
        Instruction("ADDI", rd=4, rs1=0, imm=4, raw="ADDI R4,R0,4", label="X"),
    ]
    # ground truth: BEQ at index 1 is NOT taken (R1=1 != R0=0)
    spec = SpeculativeExecutionSimulator(branch_program, {1: False}, penalty=3)
    sstats = spec.run()
    print(f"Cycles={sstats.total_cycles} Branches={sstats.branches} "
          f"Mispredictions={sstats.mispredictions} Squashed={sstats.squashed_instructions}")

    print("\n=== Simultaneous Multithreading (2-wide issue, 2 threads) ===")
    thread_a = program[:4]
    thread_b = program[4:]
    smt = SMTSimulator(thread_a, thread_b, issue_width=2)
    smt_stats = smt.run()
    print(f"Cycles={smt_stats.total_cycles} Combined IPC={smt_stats.combined_ipc:.2f} "
          f"Slot utilization={smt_stats.slot_utilization:.1%}")

    print("\nSUPERSCALAR MODULE SELF-TEST PASSED")
