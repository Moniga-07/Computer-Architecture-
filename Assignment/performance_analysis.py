"""
performance_analysis.py
Computes standard processor performance metrics and runs a benchmark suite
across pipeline configurations / execution techniques so results can be
compared quantitatively (CO1/CO3: CPI, clock cycles, execution time,
throughput, bottleneck identification, optimization recommendations).
"""

from dataclasses import dataclass
from typing import List, Dict
from cpu_core import Instruction
from pipeline_simulator import PipelineSimulator
from superscalar_simulator import OutOfOrderSimulator, SMTSimulator


CLOCK_FREQ_HZ = 2.0e9   # 2 GHz reference clock, used only to translate cycles -> time


@dataclass
class BenchmarkResult:
    name: str
    cycles: int
    instructions: int
    cpi: float
    execution_time_ns: float
    throughput_mips: float   # million instructions per second


def evaluate(name: str, cycles: int, instructions: int, clock_hz: float = CLOCK_FREQ_HZ) -> BenchmarkResult:
    cpi = cycles / instructions if instructions else 0
    exec_time_s = cycles / clock_hz
    exec_time_ns = exec_time_s * 1e9
    throughput_mips = (instructions / exec_time_s) / 1e6 if exec_time_s else 0
    return BenchmarkResult(name, cycles, instructions, cpi, exec_time_ns, throughput_mips)


def make_benchmark_programs() -> Dict[str, List[Instruction]]:
    """A small suite of representative programs used consistently across all
    configurations, so comparisons are apples-to-apples."""

    # Program A: hazard-heavy (dependent chain) — stresses data hazard handling
    hazard_heavy = [
        Instruction("LW",   rd=1, rs1=0, imm=0,  raw="LW   R1,0(R0)"),
        Instruction("ADD",  rd=2, rs1=1, rs2=1,  raw="ADD  R2,R1,R1"),
        Instruction("ADD",  rd=3, rs1=2, rs2=2,  raw="ADD  R3,R2,R2"),
        Instruction("ADD",  rd=4, rs1=3, rs2=3,  raw="ADD  R4,R3,R3"),
        Instruction("SW",   rs2=4, rs1=0, imm=4, raw="SW   R4,4(R0)"),
    ]

    # Program B: branch-heavy — stresses control hazard handling
    branch_heavy = [
        Instruction("ADDI", rd=1, rs1=0, imm=1, raw="ADDI R1,R0,1"),
        Instruction("BEQ",  rs1=1, rs2=0, label="L1", raw="BEQ R1,R0,L1"),
        Instruction("ADDI", rd=2, rs1=0, imm=2, raw="ADDI R2,R0,2", label="L1"),
        Instruction("BNE",  rs1=2, rs2=0, label="L2", raw="BNE R2,R0,L2"),
        Instruction("ADDI", rd=3, rs1=0, imm=3, raw="ADDI R3,R0,3", label="L2"),
        Instruction("JMP",  label="L3", raw="JMP L3"),
        Instruction("ADDI", rd=4, rs1=0, imm=4, raw="ADDI R4,R0,4", label="L3"),
    ]

    # Program C: independent-instruction-rich — favourable for OoO/SMT
    independent_rich = [
        Instruction("ADDI", rd=1, rs1=0, imm=1, raw="ADDI R1,R0,1"),
        Instruction("ADDI", rd=2, rs1=0, imm=2, raw="ADDI R2,R0,2"),
        Instruction("ADDI", rd=3, rs1=0, imm=3, raw="ADDI R3,R0,3"),
        Instruction("ADDI", rd=4, rs1=0, imm=4, raw="ADDI R4,R0,4"),
        Instruction("ADD",  rd=5, rs1=1, rs2=2, raw="ADD  R5,R1,R2"),
        Instruction("ADD",  rd=6, rs1=3, rs2=4, raw="ADD  R6,R3,R4"),
        Instruction("ADD",  rd=7, rs1=5, rs2=6, raw="ADD  R7,R5,R6"),
    ]

    return {"hazard_heavy": hazard_heavy, "branch_heavy": branch_heavy, "independent_rich": independent_rich}


def run_full_benchmark_suite() -> List[BenchmarkResult]:
    programs = make_benchmark_programs()
    results = []

    for pname, prog in programs.items():
        # In-order pipeline, naive stall policy
        sim = PipelineSimulator(prog, hazard_policy="stall", branch_policy="flush")
        stats, _ = sim.run()
        results.append(evaluate(f"{pname} | in-order (stall, flush)", stats.total_cycles, len(prog)))

        # In-order pipeline, optimized forwarding + prediction
        sim2 = PipelineSimulator(prog, hazard_policy="forward", branch_policy="predict")
        stats2, _ = sim2.run()
        results.append(evaluate(f"{pname} | in-order (forward, predict)", stats2.total_cycles, len(prog)))

        # Out-of-order (2-wide)
        ooo = OutOfOrderSimulator(prog, issue_width=2, num_fu=2)
        ostats, _ = ooo.run()
        results.append(evaluate(f"{pname} | out-of-order (2-wide)", ostats.total_cycles, len(prog)))

    return results


def identify_bottleneck(sim_stats) -> str:
    """Given PipelineStats, name the dominant source of lost cycles."""
    d, c = sim_stats.data_hazard_stalls, sim_stats.control_hazard_stalls
    if d == 0 and c == 0:
        return "No stalls detected — pipeline running at ideal CPI=1."
    if d >= c:
        return (f"Data hazards dominate ({d} stall cycles vs {c} control-hazard cycles). "
                f"Recommendation: increase forwarding paths / reorder independent instructions "
                f"between a load and its first use (software pipelining / compiler scheduling).")
    return (f"Control hazards dominate ({c} stall cycles vs {d} data-hazard cycles). "
            f"Recommendation: adopt a dynamic (2-bit saturating counter) branch predictor "
            f"or a branch target buffer to cut misprediction penalty.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_full_benchmark_suite()
    print(f"{'Benchmark':45s}{'Cycles':>8s}{'Instr':>7s}{'CPI':>7s}{'ExecTime(ns)':>14s}{'MIPS':>8s}")
    for r in results:
        print(f"{r.name:45s}{r.cycles:8d}{r.instructions:7d}{r.cpi:7.2f}{r.execution_time_ns:14.2f}{r.throughput_mips:8.1f}")

    print("\n=== Bottleneck Analysis (hazard_heavy program) ===")
    programs = make_benchmark_programs()
    sim = PipelineSimulator(programs["hazard_heavy"], hazard_policy="stall", branch_policy="flush")
    stats, _ = sim.run()
    print(identify_bottleneck(stats))

    print("\n=== Bottleneck Analysis (branch_heavy program) ===")
    sim2 = PipelineSimulator(programs["branch_heavy"], hazard_policy="forward", branch_policy="flush")
    stats2, _ = sim2.run()
    print(identify_bottleneck(stats2))

    print("\nPERFORMANCE ANALYSIS MODULE SELF-TEST PASSED")
