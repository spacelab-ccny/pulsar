import hmac
import hashlib
import numpy as np
import torch
from functools import reduce


class Prg:
    def __init__(self, key, seed):
        self.key = key
        self.val = b"\x01" * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b""):
        self.key = self.hmac(self.key, self.val + b"\x00" + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b"\x01" + data)
            self.val = self.hmac(self.key, self.val)

    def get_bits(self, n):
        xs = np.zeros(n, dtype=bool)

        for i in range(0, n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs


class Uniform:
    def __init__(self, prg) -> None:
        self.prg = prg

    def generate_int32(self, shape):
        # Method from https://stackoverflow.com/a/41069967
        # Generate the required number of bits
        total = reduce(lambda x1, x2: x1 * x2, shape, 1)
        b = self.prg.get_bits(32 * total)

        # Convert the bits to ints using a dot product
        b = b.reshape(total, 32)
        b = b.dot(1 << np.arange(32)[::-1])

        # Reshape the generated ints to the required size
        b = b.reshape(shape)
        return b

    def generate_float32(self, shape):
        # Use the same method as numpy to generate 32-bit floats
        # https://github.com/numpy/numpy/blob/maintenance/1.24.x/numpy/random/src/distributions/distributions.c#L20

        f = np.full(shape, (1.0 / 16777216.0))
        return (self.generate_int32(shape) >> 8) * f


class Normal:
    def __init__(self, prg, mean=0, stddev=1, device="cpu") -> None:
        self.uniform = Uniform(prg)
        self.normal = torch.distributions.Normal(mean, stddev)
        self.device = device

    def generate(self, shape, dtype=torch.float32) -> torch.tensor:
        return self.normal.icdf(torch.tensor(self.uniform.generate_float32(shape), dtype=dtype, device=self.device))


class Permutation:
    def __init__(self, prg) -> None:
        self.uniform = Uniform(prg)

    def get(self, x: int):
        # Fisher-Yates Shuffle
        # https://en.wikipedia.org/wiki/Fisher–Yates_shuffle#The_modern_algorithm
        a = np.arange(x)
        j = np.mod(self.uniform.generate_int32((x,)), x)
        for i in reversed(range(x)):
            # Exchange two values in Numpy arrays
            # https://stackoverflow.com/a/47951813
            a[[i, j[i]]] = a[[j[i], i]]

        # permutation and inverse permutation
        return (a, np.argsort(a))
