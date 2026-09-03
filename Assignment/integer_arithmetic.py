"""
integer_arithmetic.py
Implements and demonstrates:
    - Signed addition / subtraction with carry look-ahead adder (CLA)
    - Booth's Multiplication Algorithm (radix-2)
    - Restoring Division Algorithm
    - Non-Restoring Division Algorithm

Each function also returns a step-by-step trace so results can be validated
and presented (Requirement: Results & Validation / Use of Modern Tools).
"""

from typing import List, Tuple

BITS = 8            # small bit-width chosen so traces are human-readable
MASK = (1 << BITS) - 1


def to_signed(v, bits=BITS):
    v &= (1 << bits) - 1
    if v & (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def to_unsigned(v, bits=BITS):
    return v & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# 1. Carry Look-Ahead Adder (4-bit blocks, ripple across blocks for clarity)
# ---------------------------------------------------------------------------
def cla_add_4bit(a: int, b: int, cin: int = 0) -> Tuple[int, int, List[dict]]:
    """4-bit Carry Look-Ahead adder. Returns (sum, carry_out, trace)."""
    a &= 0xF
    b &= 0xF
    trace = []
    carry = cin
    generate = [(a >> i) & 1 & ((b >> i) & 1) for i in range(4)]
    propagate = [((a >> i) & 1) | ((b >> i) & 1) for i in range(4)]
    carries = [carry]
    for i in range(4):
        g, p = generate[i], propagate[i]
        c_next = g | (p & carries[i])
        carries.append(c_next)
        trace.append(dict(bit=i, a_bit=(a >> i) & 1, b_bit=(b >> i) & 1,
                           generate=g, propagate=p, carry_in=carries[i], carry_out=c_next))
    sum_bits = 0
    for i in range(4):
        s = ((a >> i) & 1) ^ ((b >> i) & 1) ^ carries[i]
        sum_bits |= (s << i)
    return sum_bits, carries[4], trace


def cla_add_8bit(a: int, b: int) -> Tuple[int, int, List[dict]]:
    """Compose two 4-bit CLA blocks to add 8-bit signed numbers."""
    ua, ub = to_unsigned(a), to_unsigned(b)
    lo_sum, lo_carry, lo_trace = cla_add_4bit(ua & 0xF, ub & 0xF, 0)
    hi_sum, hi_carry, hi_trace = cla_add_4bit((ua >> 4) & 0xF, (ub >> 4) & 0xF, lo_carry)
    result = (hi_sum << 4) | lo_sum
    return to_signed(result), hi_carry, {"low_block": lo_trace, "high_block": hi_trace}


def signed_add(a: int, b: int):
    result, carry, trace = cla_add_8bit(a, b)
    return result, trace


def signed_sub(a: int, b: int):
    """Two's complement subtraction: a - b = a + (~b + 1)."""
    b_comp = to_signed((~to_unsigned(b) + 1))
    return signed_add(a, b_comp)


# ---------------------------------------------------------------------------
# 2. Booth's Multiplication Algorithm (radix-2, signed)
# ---------------------------------------------------------------------------
def booth_multiply(multiplicand: int, multiplier: int, bits: int = BITS):
    """Radix-2 Booth's algorithm. Returns (product, trace_of_steps)."""
    m = to_unsigned(multiplicand, bits)
    q = to_unsigned(multiplier, bits)
    a = 0
    q_1 = 0
    trace = []
    n = bits
    for step in range(n):
        q0 = q & 1
        action = None
        if q0 == 1 and q_1 == 0:
            a = to_unsigned(a - m, bits)
            action = "A = A - M"
        elif q0 == 0 and q_1 == 1:
            a = to_unsigned(a + m, bits)
            action = "A = A + M"
        else:
            action = "No arithmetic op"

        # arithmetic right shift of {A, Q, Q-1}
        combined = (a << (bits + 1)) | (q << 1) | q_1
        sign_bit = (a >> (bits - 1)) & 1
        combined >>= 1
        combined |= (sign_bit << (2 * bits))
        a = (combined >> (bits + 1)) & MASK
        q = (combined >> 1) & MASK
        q_1 = combined & 1

        trace.append(dict(step=step + 1, action=action,
                           A=format(a, f'0{bits}b'), Q=format(q, f'0{bits}b'), Q_1=q_1))

    product_unsigned = (a << bits) | q
    product = to_signed(product_unsigned, 2 * bits)
    return product, trace


# ---------------------------------------------------------------------------
# 3. Restoring Division Algorithm (unsigned, magnitude based)
# ---------------------------------------------------------------------------
def restoring_division(dividend: int, divisor: int, bits: int = BITS):
    """Restoring division on unsigned magnitudes; caller re-applies sign."""
    n = bits
    Q = dividend & MASK
    M = divisor & MASK
    A = 0
    trace = []
    for step in range(n):
        # shift left A,Q
        combined = (A << (n + 1)) | (Q << 1)
        A = (combined >> n) & ((1 << (n + 1)) - 1)
        Q = combined & MASK
        A = to_unsigned(A - M, n + 1)
        if (A >> n) & 1:  # A negative -> restore
            A = to_unsigned(A + M, n + 1)
            Q = Q & ~1
            action = "Subtract -> negative -> restore, Q0=0"
        else:
            Q = Q | 1
            action = "Subtract -> non-negative -> keep, Q0=1"
        trace.append(dict(step=step + 1, action=action,
                           A=format(A & MASK, f'0{n}b'), Q=format(Q, f'0{n}b')))
    quotient = Q
    remainder = A & MASK
    return quotient, remainder, trace


# ---------------------------------------------------------------------------
# 4. Non-Restoring Division Algorithm
# ---------------------------------------------------------------------------
def non_restoring_division(dividend: int, divisor: int, bits: int = BITS):
    n = bits
    Q = dividend & MASK
    M = divisor & MASK
    A = 0
    trace = []
    for step in range(n):
        combined = (A << (n + 1)) | (Q << 1)
        A = (combined >> n) & ((1 << (n + 1)) - 1)
        Q = combined & MASK
        prev_negative = (A >> n) & 1
        if prev_negative == 0:
            A = to_unsigned(A - M, n + 1)
            action = "A >= 0: A = A - M"
        else:
            A = to_unsigned(A + M, n + 1)
            action = "A < 0: A = A + M"
        if (A >> n) & 1:
            Q = Q & ~1
        else:
            Q = Q | 1
        trace.append(dict(step=step + 1, action=action,
                           A=format(A & MASK, f'0{n}b'), Q=format(Q, f'0{n}b')))
    # final restoring correction if remainder negative
    if (A >> n) & 1:
        A = to_unsigned(A + M, n + 1)
        trace.append(dict(step="final-correction", action="Remainder negative -> A = A + M",
                           A=format(A & MASK, f'0{n}b'), Q=format(Q, f'0{n}b')))
    quotient = Q
    remainder = A & MASK
    return quotient, remainder, trace


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Carry Look-Ahead Adder ===")
    r, trace = signed_add(45, 30)
    print(f"45 + 30 = {r} (expected 75)")
    assert r == 75

    r, trace = signed_sub(50, 20)
    print(f"50 - 20 = {r} (expected 30)")
    assert r == 30

    print("\n=== Booth's Multiplication ===")
    prod, trace = booth_multiply(13, -6, bits=8)
    print(f"13 * -6 = {prod} (expected -78)")
    assert prod == -78
    prod2, _ = booth_multiply(-9, -7, bits=8)
    print(f"-9 * -7 = {prod2} (expected 63)")
    assert prod2 == 63

    print("\n=== Restoring Division ===")
    q, r_, trace = restoring_division(29, 4, bits=8)
    print(f"29 / 4 = quotient {q}, remainder {r_} (expected 7 r 1)")
    assert q == 7 and r_ == 1

    print("\n=== Non-Restoring Division ===")
    q, r_, trace = non_restoring_division(29, 4, bits=8)
    print(f"29 / 4 = quotient {q}, remainder {r_} (expected 7 r 1)")
    assert q == 7 and r_ == 1

    q, r_, trace = non_restoring_division(100, 7, bits=8)
    print(f"100 / 7 = quotient {q}, remainder {r_} (expected 14 r 2)")
    assert q == 14 and r_ == 2

    print("\nALL INTEGER ARITHMETIC SELF-TESTS PASSED")
