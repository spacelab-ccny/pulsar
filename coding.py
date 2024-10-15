from reedmuller import reedmuller
from bitarray import bitarray
from bitarray.util import zeros
from reedsolo import RSCodec, ReedSolomonError
import numpy as np
import json
import subprocess
from typing import *
import sys
import os


class ReedSolomonCode:
    def __init__(self, k: int) -> None:
        self.message_len = k
        self.alphabet_size = 256
        self.block_len = self.alphabet_size - 1

        # The message size must be lower than the field size
        assert self.message_len < self.block_len

        self.rs_code = RSCodec(self.block_len - self.message_len)

    def encode(self, a: bytes) -> bitarray:
        block_encodings = bitarray()

        # Encode in blocks
        for i in range(0, len(a), self.message_len):
            encoded_bytes = self.rs_code.encode(a[i : i + self.message_len])
            encoded_bits = bitarray()
            encoded_bits.frombytes(encoded_bytes)
            block_encodings.extend(encoded_bits)

        return block_encodings

    def decode(self, b: bitarray) -> tuple[bytes, list[int]]:
        block_bit_length = self.block_len * 8
        decoded = bytearray()
        errors = []

        for i in range(0, len(b), block_bit_length):
            to_decode = b[i : i + block_bit_length].tobytes()
            try:
                decoded.extend(self.rs_code.decode(to_decode)[0])
            except ReedSolomonError:
                # Decoding error, so just add some zero bits
                decoded.extend(b"\x00" * (self.message_len * 8))
                errors.append(i)

        return (bytes(decoded), errors)


class ReedMullerCode:
    def __init__(self, r: int, m: int) -> None:
        self.r = r
        self.m = m

        self.rm_code = reedmuller.ReedMuller(r, m)
        self.message_len = self.rm_code.k
        self.block_len = self.rm_code.n

        # Parameters for Hadamard Codes (r=1, m)
        # self.block_len = 2**self.m
        # self.message_len = self.m+1
        # self.min_distance = 2**(self.m-1)

        # Parameters for repetition codes (r=0, m)
        # self.block_len = 2**self.m
        # self.message_len = 1
        # self.min_distance = 2**(self.m)

    def encode(self, a: bitarray) -> bitarray:
        ints = list(map(int, a.to01()))
        result = bitarray()
        for i in range(0, len(a), self.message_len):
            result.extend(self.rm_code.encode(ints[i : i + self.message_len]))
        return result

    def decode(self, b: bitarray) -> tuple[bitarray, list[int]]:
        ints = list(map(int, b.to01()))
        decoded = bitarray()
        errors = []
        for i in range(0, len(b), self.block_len):
            result = self.rm_code.decode(ints[i : i + self.block_len])
            if result is not None:
                decoded.extend(result)
            else:
                # Decoding error, so just add some zero bits
                decoded.extend(zeros(self.message_len))
                errors.append(i)
        return (decoded, errors)


def permuted(x, p):
    a = bitarray(len(x))
    assert len(x) == len(p)
    for i in range(0, len(p)):
        a[i] = x[p[i]]
    return a


class ConcatenatedCode:
    def __init__(
        self,
        inner_code: ReedSolomonCode,
        outer_code: ReedMullerCode,
        permute=False,
        redundancy=1,
        image_size=256 * 256,
    ) -> None:
        self.inner_code = inner_code
        self.outer_code = outer_code

        self.permute = permute

        # Maximum outer message length calculation
        max_outer_blocks_in_image = image_size // self.outer_code.block_len
        max_outer_bits_in_image = (
            max_outer_blocks_in_image * self.outer_code.message_len
        )
        max_outer_message_len = max_outer_bits_in_image // 8  # bytes

        # Calculate how many inner codes the outer code can hold
        inner_messages = max_outer_message_len // self.inner_code.block_len

        # Calculate the total number of bytes this concatenated code can hold
        self.capacity: int = inner_messages * self.inner_code.message_len

        # Calculate the size of a code output by this class
        # Note that this may differ from the image size due to rounding
        self.output_size = (
            inner_messages
            * self.inner_code.block_len
            * 8
            // self.outer_code.message_len
            * self.outer_code.block_len
        )

        # Note the amount of padding required to match the code output size to the image size
        self.padding = (
            image_size - self.output_size if image_size > self.output_size else 0
        )

        # Set up the ability for the inner code to be repeated for redundancy
        assert redundancy >= 1
        assert redundancy <= inner_messages
        self.redundancy = redundancy

    def encode(self, a: bytes) -> bitarray:
        to_encode = bytearray(a)

        # Duplicate the message to encode for redundancy
        assert len(to_encode) <= self.capacity // self.redundancy
        for _ in range(self.redundancy - 1):
            to_encode.extend(a)

        # Pad out the message
        if len(to_encode) < self.capacity:
            for _ in range(self.capacity - len(to_encode)):
                to_encode.append(b"\x00")

        to_encode = bytes(to_encode)
        assert len(to_encode) == self.capacity

        # Perform the encoding
        inner_encoded = self.inner_code.encode(to_encode)
        outer_encoded = self.outer_code.encode(inner_encoded)

        # Add any necessary padding
        result = outer_encoded
        result.extend(zeros(self.padding))

        if self.permute:
            # Apply a permutation/shuffle to the to_encode bitarray that is reversible
            self.p = np.random.permutation(len(result))
            self.p_inv = np.argsort(self.p)
            result = permuted(result, self.p)

        return result

    def decode(
        self, b: bitarray
    ) -> tuple[bytes, tuple[bytes, list[int], bitarray, list[int]]]:
        if self.permute:
            # Reverse the permutation/shuffle on the bitarray
            b = permuted(b, self.p_inv)

        # Remove any padding added
        b = b[: self.output_size]

        outer_decoded, outer_errors = self.outer_code.decode(b)
        inner_decoded, inner_errors = self.inner_code.decode(outer_decoded)

        # Remove the redundancy from the decoding, if any
        result = inner_decoded[: self.capacity // self.redundancy]
        meta = {
            "inner": {"decoded": inner_decoded, "errors": inner_errors},
            "outer": {"decoded": outer_decoded, "errors": outer_errors},
        }

        return (result, meta)


class SageCode(ConcatenatedCode):
    _SCRIPT_DIR = "sage/"
    _ENCODE_SCRIPT = "sage_encode.sage"
    _DECODE_SCRIPT = "sage_decode.sage"

    CODE_LIBRARY = {
        0.35: [
            {
                "input_size": 100,
                "output_size": 32640,
                "outer": {"type": "ReedSolomon", "parameters": (255, 100, 256)},
                "inner": {"type": "ReedMuller", "parameters": (1, 7)},
            },
        ],
        0.30: [
            {
                "input_size": 200,
                "output_size": 32640,
                "outer": {"type": "ReedSolomon", "parameters": (255, 200, 256)},
                "inner": {"type": "ReedMuller", "parameters": (1, 7)},
            },
        ],
        0.25: [
            {
                "input_size": 228,
                "output_size": 19783,
                "from_str": '[0.25, 0.975, 0.095, 183, 271, 10, 73, "BCH", 18]',
            },
        ],
        0.20: [
            {
                "input_size": 257,
                "output_size": 14739,
                "from_str": '[0.2, 0.95, 0.14254, 187, 289, 11, 51, "BCH", 17]',
            },
        ],
        0.15: [
            {
                "input_size": 243,
                "output_size": 9843,
                "from_str": '[0.15, 0.9, 0.20227572894442752, 177, 193, 11, 51, "BCH", 17]',
            },
        ],
        0.10: [
            {
                "input_size": 100,
                "output_size": 3570,
                "outer": {"type": "ReedSolomon", "parameters": (255, 100, 256)},
                # Hamming(7,4)
                "inner": {"type": "Hamming", "parameters": (3,)},
            },
        ],
        0.05: [
            {
                "input_size": 200,
                "output_size": 3570,
                "outer": {"type": "ReedSolomon", "parameters": (255, 200, 256)},
                # Hamming(7,4)
                "inner": {"type": "Hamming", "parameters": (3,)},
            },
        ],
    }

    def __init__(self, parameters, permute=False, redundancy=1) -> None:
        self.permute = permute

        # These are a result of the Sage script configuration
        self.params = parameters
        self.sage_input_size_bytes = parameters["input_size"]
        self.sage_code_size_bits = parameters["output_size"]

        # self.outer_code_name = parameters["outer"]["type"]
        # self.outer_code_params = parameters["outer"]["parameters"]
        # self.inner_code_name = parameters["inner"]["type"]
        # self.inner_code_params = parameters["inner"]["parameters"]

        self.redundancy = redundancy

    # def _code_params_to_str(code_params: tuple) -> str:
    #     return ",".join([str(x) for x in code_params])

    def _make_code_info(params: dict, message: list, start: int, length: int) -> dict:
        code_info = {}

        # Get code data
        if "from_str" in params:
            code_info["from_str"] = params["from_str"]
        else:
            code_info["inner"] = params["inner"]
            code_info["outer"] = params["outer"]

        # Get message data
        code_info["message"] = message[start : start + length]

        return code_info

    def call_sage(mode, to_sage):
        current_dir = os.getcwd()
        os.chdir(SageCode._SCRIPT_DIR)
        pr = subprocess.run(
            [
                "sage",
                (
                    SageCode._ENCODE_SCRIPT
                    if mode == "encode"
                    else SageCode._DECODE_SCRIPT
                ),
            ],
            capture_output=True,
            input=json.dumps(to_sage),
            encoding="utf-8",
        )
        os.chdir(current_dir)

        if len(pr.stderr) != 0:
            sys.stderr.write(pr.stderr)
            raise ValueError(to_sage)

        return json.loads(pr.stdout)

    def encode(self, a: bytes, region_size: int) -> bitarray:
        to_encode = bytearray(a)

        # Number of Sage encodings we can fit in a single image
        sage_calls = region_size // self.sage_code_size_bits
        capacity = sage_calls * self.sage_input_size_bytes
        output_size = sage_calls * self.sage_code_size_bits
        padding = region_size - output_size if region_size > output_size else 0

        # Duplicate the message to encode for redundancy
        assert len(to_encode) <= capacity // self.redundancy
        for _ in range(self.redundancy - 1):
            to_encode.extend(a)

        # Pad out the message
        if len(to_encode) < capacity:
            to_encode.extend(bytearray(b"\x00" * (capacity - len(to_encode))))

        to_encode = list(bytes(to_encode))
        assert len(to_encode) == capacity

        result = bitarray()

        to_sage = []

        for i in range(0, capacity, self.sage_input_size_bytes):
            to_sage.append(
                SageCode._make_code_info(
                    self.params, to_encode, i, self.sage_input_size_bytes
                )
            )

        for encoded in SageCode.call_sage("encode", to_sage):
            result.extend(encoded)

        result.extend(zeros(padding))

        if self.permute:
            # Apply a permutation/shuffle to the to_encode bitarray that is reversible
            self.p = np.random.permutation(len(result))
            self.p_inv = np.argsort(self.p)
            result = permuted(result, self.p)

        return result

    # TODO typing
    def batch_encode(message, regions):
        to_sage = []
        l = 0
        for region in regions:
            sage_input_size_bytes = region["code"]["input_size"]
            sage_code_size_bits = region["code"]["output_size"]
            region_size = region["size"]

            # Number of Sage encodings we can fit in a single image
            sage_calls = region_size // sage_code_size_bits
            capacity = sage_calls * sage_input_size_bytes
            output_size = sage_calls * sage_code_size_bits

            to_encode = bytearray(message[l : l + capacity])
            l += capacity

            # Pad out the message
            if len(to_encode) < capacity:
                to_encode.extend(bytearray(b"\x00" * (capacity - len(to_encode))))

            # Convert to_encode to a list
            to_encode = list(bytes(to_encode))
            assert len(to_encode) == capacity

            for i in range(0, capacity, sage_input_size_bytes):
                to_sage.append(
                    SageCode._make_code_info(
                        region["code"], to_encode, i, sage_input_size_bytes
                    )
                )

        results = SageCode.call_sage("encode", to_sage)
        result_idx = 0

        for region in regions:
            sage_code_size_bits = region["code"]["output_size"]
            region_size = region["size"]
            sage_calls = region_size // sage_code_size_bits
            output_size = sage_calls * sage_code_size_bits
            padding = region_size - output_size if region_size > output_size else 0

            result = bitarray()
            for i in range(sage_calls):
                result.extend(results[result_idx + i])
            result.extend(zeros(padding))

            # TODO Need better code here
            region["ecc"] = SageCode(region["code"], permute=True)
            region["ecc"].p = np.random.permutation(len(result))
            region["ecc"].p_inv = np.argsort(region["ecc"].p)
            result = permuted(result, region["ecc"].p)

            region["encoded_message"] = np.array(list(result))

            result_idx += sage_calls

    def decode(
        self, b: bitarray
    ) -> tuple[bytes, tuple[bytes, list[int], bitarray, list[int]]]:
        # Number of Sage encodings we can fit in a single image
        region_size = len(b)
        sage_calls = region_size // self.sage_code_size_bits
        output_size = sage_calls * self.sage_code_size_bits

        if self.permute:
            # Reverse the permutation/shuffle on the bitarray
            b = permuted(b, self.p_inv)

        # Remove any padding added
        b = b[:output_size]

        # Convert to a list
        b = list(b)

        result = bytearray()

        to_sage = []

        for i in range(0, output_size, self.sage_code_size_bits):
            to_sage.append(
                SageCode._make_code_info(self.params, b, i, self.sage_input_size_bits)
            )

        for decoded in SageCode.call_sage("decode", to_sage):
            result.extend(decoded)

        return bytes(result), None
