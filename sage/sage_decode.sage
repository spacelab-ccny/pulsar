import json
from cc_code import ConcatenatedCode
import sys

from_pulsar = json.loads(input())
# from_pulsar = json.loads(tst_json)
to_pulsar = []


def handle_task(task):
    # check if the task is coming directly from a str object
    to_send = None
    if "from_str" in task:
        as_str = json.loads(task["from_str"])
        as_obj = as_str[2:]  # beg. contains inform. on code
        cc = ConcatenatedCode.from_file(as_obj)
        msg = vector(GF(2), task["message"])
        # this could be more than decode call
        out = []
        bl_len = cc.length()
        for i in range(0, len(msg), bl_len):
            tmps = cc.decode(msg[i:i+bl_len]).list()
            out += tmps
        # change to bitstream
        bit_str = []
        f_size = int(round(math.log(cc._of.order(), 2)))
        for wo in out:
            icomp = wo.to_integer()
            for l in range(f_size):
                tmpz = (icomp >> l) & 1
                bit_str.append(tmpz)

        # re-combine
        to_send = []
        mc = 0
        up_b = len(bit_str) // 8
        for i in range(0, up_b*8, 8):
            to_int = 0
            for j in range(8):
                to_int += bit_str[i+j]*2**j
            to_send.append(int(to_int))
        # for i in range(len(out)):
        #    to_send.append(out[i].to_integer())
    else:
        # outer code
        n = task["outer"]["parameters"][0]
        k = task["outer"]["parameters"][1]
        q = task["outer"]["parameters"][2]

        # print field elements as integers (instead of polys)
        F = GF(q, repr="int")
        outer_C = codes.GeneralizedReedSolomonCode(F.list()[:n], k)

        # inner code -- Reed-Muller
        if task["inner"]["type"] == 'ReedMuller':
            r = task["inner"]["parameters"][0]  # order
            m = task["inner"]["parameters"][1]  # number of variables
            inner_n = 2**m
            inner_k = m + 1
            inner_C = codes.BinaryReedMullerCode(r, m)

        # inner code -- Hamming
        if task["inner"]["type"] == 'Hamming':
            inner_n = 2 ^ task["inner"]["parameters"][0] - 1
            inner_k = 2 ^ task["inner"]["parameters"][0] - \
                task["inner"]["parameters"][0] - 1
            inner_C = codes.HammingCode(GF(2), task["inner"]["parameters"][0])

        to_decode = vector(GF(2), task["message"])

        inner_D = inner_C.decoder("NearestNeighbor")

        inner_decoded = []
        for i in range(0, len(to_decode), inner_n):
            to_decode_partial = vector(GF(2), to_decode[i:i+(inner_n)])
            inner_dec = inner_D.decode_to_message(to_decode_partial)
            inner_decoded.extend(inner_dec)

        # Convert from the bit representation to the field representation
        # https://doc.sagemath.org/html/en/reference/finite_rings/sage/rings/finite_rings/element_givaro.html#sage.rings.finite_rings.element_givaro.Cache_givaro.element_from_data
        from_inner = [F(inner_decoded[i:i+8])
                      for i in range(0, len(inner_decoded), 8)]
        from_inner = vector(F, from_inner)

        # Unique decoding
        # https://doc.sagemath.org/html/en/reference/coding/sage/coding/decoder.html#sage.coding.decoder.Decoder.decode_to_code
        to_send = outer_C.decode_to_message(from_inner)

    # Sage is strange about type coercion, so I'll use the string representation instead
    return json.loads(list(to_send).__repr__())


for task in from_pulsar:
    to_send = handle_task(task)
    to_pulsar.append(to_send)

print(json.dumps(to_pulsar))
