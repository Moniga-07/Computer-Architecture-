"""
ieee754.py
IEEE-754 single precision (32-bit) and double precision (64-bit) floating
point representation and arithmetic (addition, multiplication), implemented
from first principles (sign / exponent / mantissa manipulation) rather than
relying on native float hardware, so every step can be shown and validated.
"""

import struct
from dataclasses import dataclass


@dataclass
class FloatFields:
    sign: int
    exponent: int      # biased
    mantissa: int       # stored fraction bits (no implicit 1)
    bits: int
    exp_bits: int
    mant_bits: int
    bias: int

    def unbiased_exponent(self):
        return self.exponent - self.bias

    def significand(self):
        """Return the true significand as a float (1.mantissa) for normal numbers."""
        return 1.0 + self.mantissa / (2 ** self.mant_bits)


def decode(value: float, precision: str = "single") -> FloatFields:
    if precision == "single":
        packed = struct.pack(">f", value)
        bits = int.from_bytes(packed, "big")
        exp_bits, mant_bits, bias = 8, 23, 127
        total_bits = 32
    else:
        packed = struct.pack(">d", value)
        bits = int.from_bytes(packed, "big")
        exp_bits, mant_bits, bias = 11, 52, 1023
        total_bits = 64

    sign = (bits >> (total_bits - 1)) & 1
    exponent = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    mantissa = bits & ((1 << mant_bits) - 1)
    return FloatFields(sign, exponent, mantissa, bits, exp_bits, mant_bits, bias)


def encode(fields: FloatFields, precision: str = "single") -> float:
    total_bits = 32 if precision == "single" else 64
    bits = (fields.sign << (total_bits - 1)) | (fields.exponent << fields.mant_bits) | fields.mantissa
    nbytes = 4 if precision == "single" else 8
    raw = bits.to_bytes(nbytes, "big")
    fmt = ">f" if precision == "single" else ">d"
    return struct.unpack(fmt, raw)[0]


def to_binary_string(value: float, precision: str = "single") -> str:
    f = decode(value, precision)
    return f"{f.sign} | {format(f.exponent, f'0{f.exp_bits}b')} | {format(f.mantissa, f'0{f.mant_bits}b')}"


def explain(value: float, precision: str = "single") -> dict:
    f = decode(value, precision)
    return dict(
        value=value,
        sign=f.sign,
        biased_exponent=f.exponent,
        unbiased_exponent=f.unbiased_exponent(),
        mantissa_bits=format(f.mantissa, f'0{f.mant_bits}b'),
        significand=f.significand(),
        reconstructed=((-1) ** f.sign) * f.significand() * (2 ** f.unbiased_exponent()),
        layout=to_binary_string(value, precision),
    )


# ---------------------------------------------------------------------------
# IEEE-754 addition performed manually via align-shift-add-normalize, the
# classic floating point adder datapath, then cross-checked against the
# hardware double result.
# ---------------------------------------------------------------------------
def ieee754_add(a: float, b: float, precision: str = "single"):
    fa = decode(a, precision)
    fb = decode(b, precision)
    mant_bits = fa.mant_bits
    bias = fa.bias

    # implicit leading 1 (assume normalized, non-zero operands)
    ma = (1 << mant_bits) | fa.mantissa
    mb = (1 << mant_bits) | fb.mantissa
    ea = fa.unbiased_exponent()
    eb = fb.unbiased_exponent()

    trace = [f"Operand A: sign={fa.sign} exp={ea} mantissa(with hidden bit)={bin(ma)}",
             f"Operand B: sign={fb.sign} exp={eb} mantissa(with hidden bit)={bin(mb)}"]

    # Align exponents (shift smaller-magnitude-exponent mantissa right)
    if ea >= eb:
        shift = ea - eb
        mb >>= shift
        result_exp = ea
    else:
        shift = eb - ea
        ma >>= shift
        result_exp = eb
    trace.append(f"Aligned exponents to {result_exp} (shift amount = {shift})")

    # Apply sign via two's complement style magnitude add/sub
    sa = -ma if fa.sign else ma
    sb = -mb if fb.sign else mb
    total = sa + sb
    result_sign = 1 if total < 0 else 0
    total = abs(total)
    trace.append(f"Signed mantissa sum = {total} (sign={result_sign})")

    # Normalize
    if total == 0:
        return 0.0, trace + ["Result is exactly zero"]
    while total >= (1 << (mant_bits + 1)):
        total >>= 1
        result_exp += 1
    while total < (1 << mant_bits) and total != 0:
        total <<= 1
        result_exp -= 1
    trace.append(f"Normalized mantissa={bin(total)} exponent={result_exp}")

    result_mantissa = total & ((1 << mant_bits) - 1)
    biased_exp = result_exp + bias
    result_fields = FloatFields(result_sign, biased_exp, result_mantissa, fa.bits, fa.exp_bits, mant_bits, bias)
    result = encode(result_fields, precision)
    trace.append(f"Final IEEE-754 result = {result}")
    return result, trace


def ieee754_multiply(a: float, b: float, precision: str = "single"):
    fa = decode(a, precision)
    fb = decode(b, precision)
    mant_bits = fa.mant_bits
    bias = fa.bias

    ma = (1 << mant_bits) | fa.mantissa
    mb = (1 << mant_bits) | fb.mantissa
    ea = fa.unbiased_exponent()
    eb = fb.unbiased_exponent()

    result_sign = fa.sign ^ fb.sign
    product = ma * mb                       # (1.m)*(1.m) as fixed point ints
    result_exp = ea + eb
    trace = [f"Sign = {fa.sign} XOR {fb.sign} = {result_sign}",
             f"Exponents added: {ea} + {eb} = {result_exp}",
             f"Mantissa product (fixed point) = {product}"]

    # product has 2*(mant_bits+1) bits; normalize back to mant_bits+1
    total_bits = 2 * (mant_bits + 1)
    # the product's MSB position tells us if we need to shift right by mant_bits or mant_bits+1
    if product >> (total_bits - 1):
        product >>= (mant_bits + 1)
        result_exp += 1
    else:
        product >>= mant_bits
    trace.append(f"Normalized mantissa (with hidden bit) = {bin(product)}, exponent={result_exp}")

    result_mantissa = product & ((1 << mant_bits) - 1)
    biased_exp = result_exp + bias
    result_fields = FloatFields(result_sign, biased_exp, result_mantissa, fa.bits, fa.exp_bits, mant_bits, bias)
    result = encode(result_fields, precision)
    trace.append(f"Final IEEE-754 product = {result}")
    return result, trace


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== IEEE 754 Single Precision Decode ===")
    info = explain(13.625, "single")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n=== IEEE 754 Double Precision Decode ===")
    info = explain(-0.15625, "double")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n=== IEEE 754 Addition (single precision) ===")
    result, trace = ieee754_add(5.75, 2.5, "single")
    print(f"5.75 + 2.5 = {result} (expected {5.75 + 2.5})")
    assert abs(result - (5.75 + 2.5)) < 1e-6

    result, trace = ieee754_add(-3.25, 1.5, "single")
    print(f"-3.25 + 1.5 = {result} (expected {-3.25 + 1.5})")
    assert abs(result - (-3.25 + 1.5)) < 1e-6

    print("\n=== IEEE 754 Multiplication (single precision) ===")
    result, trace = ieee754_multiply(3.5, 2.0, "single")
    print(f"3.5 * 2.0 = {result} (expected {3.5 * 2.0})")
    assert abs(result - 7.0) < 1e-6

    result, trace = ieee754_multiply(1.25, -4.0, "single")
    print(f"1.25 * -4.0 = {result} (expected {1.25 * -4.0})")
    assert abs(result - (-5.0)) < 1e-6

    print("\n=== IEEE 754 Addition (double precision) ===")
    result, trace = ieee754_add(123.456, 78.9, "double")
    print(f"123.456 + 78.9 = {result} (expected {123.456 + 78.9})")
    assert abs(result - (123.456 + 78.9)) < 1e-9

    print("\nALL IEEE-754 SELF-TESTS PASSED")
