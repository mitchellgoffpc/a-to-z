import math
import struct

x = 8.5
print(f"X = {x}")
print(f"Answer: {1 / math.sqrt(x)}")

bits = struct.pack('>f', x)
long = struct.unpack('>l', bits)[0]

E = long >> 23
M = long & 0x7FFFFF
print(f"Exponent: {E - 127} | Mantissa: {1 + M / (2 ** 23)}")

# Reconstruct the original number from the mantissa and exponent
y = (1 + M / (2 ** 23)) * (2 ** (E - 127))
assert x == y

"""
  log2(x)
= log2((1 + M / (2^23)) * (2^(E - 127)))
= log2(1 + M / (2^23)) + log2(2^(E - 127))
= log2(1 + M / (2^23)) + E - 127

MAGIC: log2(1 + x) ≈ x

≈ M / (2^23) + E - 127 + mu
≈ (M + E * (2^23)) / (2^23) + mu - 127
≈ bits(x) / (2^23) + mu - 127
"""

mu = .0430  # TODO: Can we calculate this from scratch?
print(f"Approximation:")
print(f"   log2(x) ≈ bits(x) / 2^23 - 127 + mu")
print(f"=> {math.log2(x)} ≈ {long / (2**23) - 127 + mu}")

"""
Goal: 1 / sqrt(x)
  log2(1 / sqrt(x))
= log2(x^(-1/2))
= -1/2 * log2(x)
≈ -(bits(x) / (2^23) + mu - 127) / 2

Let Y be our solution, 1 / sqrt(x)
   log(Y) = log(1 / sqrt(x))
=> log(Y) = -1/2 * log(x)
=> bits(Y) / (2^23) + mu - 127 ≈ -1/2 * (bits(x) / (2^23) + mu - 127)
=> 2*bits(Y) / (2^23) + 2*mu - 2*127 ≈ -bits(x) / (2^23) - mu + 127
=> 2*bits(Y) / (2^23) ≈ -bits(x) / (2^23) - 3*mu + 3*127
=> 2*bits(Y) ≈ -bits(x) - (3*mu + 3*127) * 2^23
=> bits(Y) ≈ 3/2 * 2^23 * (127 - mu) - 1/2 * bits(x)
"""

magic = int(3/2 * 2**23 * (127 - mu))
magic = 0x5f3759df
sqrt_long = magic - (long >> 1)
sqrt_bits = struct.pack('>l', sqrt_long)
sqrt = struct.unpack('>f', sqrt_bits)[0]

print(f"Approximate answer: {sqrt}")

"""
Newton's method:
   f(y) = 1/y^2 - x
=> f'(y) = -2 * y^-3
=> f(y) / f'(y) = (y^-2 - x) / (-2y^-3)
=> f(y) / f'(y) = (y^-2 / -2y^-3) - (x / -2y^-3)
=> f(y) / f'(y) = -(y^-2 * y^3) / 2 + (x * y^3) / 2
=> f(y) / f'(y) = -y/2 + (xy^3)/2
=> y' = y - (-y/2 + (xy^3)/2)
=> y' = y + y/2 - (xy^3)/2
=> y' = 3/2*y - (xy^3)/2
=> y' = y * (3/2 - 1/2*x*y^2)
"""

y = sqrt
yprime = y * (1.5 - .5*x*y*y)
print(f"Better answer: {yprime}")
