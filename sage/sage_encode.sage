import json
from cc_code import ConcatenatedCode
import sys

from_pulsar = json.loads(input())
# from_pulsar = json.loads(tst_json)
to_pulsar = []


def handle_task(task):
    if "from_str" in task:
        as_str = json.loads(task["from_str"])
        # beginning just contains some information on the code
        as_obj = as_str[2:]
        cc = ConcatenatedCode.from_file(as_obj)
        F = cc._oc.base_field()
        k = cc._oc.dimension()
        # need to check to see how large the base field is... the input is a string of bytes
        size_bf = int(round(math.log(F.order(), 2)))
        # step 1, flatten to bitstream
        bit_str = []
        for i, byte in enumerate(task["message"]):
            for j in range(8):
                sb = (byte >> j) & 1
                bit_str.append(sb)
        # grab in chunks of the field size
        total = len(bit_str)
        bits_per_msg = size_bf*k
        num_msgs = (len(bit_str)+bits_per_msg-1) // bits_per_msg
        list_msgs = []
        list_msgs.append(vector(F, k))
        m_c = int(0)
        for i in range(0, total, size_bf):
            if m_c == k:
                m_c = 0
                list_msgs.append(vector(F, k))
            slice = bit_str[i:i+size_bf]
            # convert to int
            as_int = 0
            limit = len(slice)
            for j in range(limit):
                as_int += (slice[j]*2**j)
            list_msgs[-1][m_c] = F.from_integer(as_int)
            m_c = m_c+1

        # for i, byte in enumerate(task["message"]):

        #    msg[i] = F.from_integer(byte)
        # need to send multiple mesages
        to_send = []
        for msg in list_msgs:
            s_l = cc.encode(msg).list()
            to_send += s_l
        # import pdb; pdb.set_trace()
    else:
        # outer code
        n = task["outer"]["parameters"][0]
        k = task["outer"]["parameters"][1]
        q = task["outer"]["parameters"][2]

        # print field elements as integers (instead of polys)
        F = GF(q, repr="int")
        outer_C = codes.GeneralizedReedSolomonCode(F.list()[:n], k)

        msg = vector(F, k)
        for i, byte in enumerate(task["message"]):
            msg[i] = F.from_integer(byte)

        # Encode the message
        # https://doc.sagemath.org/html/en/reference/coding/sage/coding/encoder.html
        outer_enc = outer_C.encode(msg)

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

        to_encode = []
        for e in outer_enc:
            to_encode.extend(list(e))

        to_send = []
        for i in range(0, len(to_encode), inner_k):
            to_encode_partial = vector(GF(2), to_encode[i:i+inner_k])
            inner_enc = inner_C.encode(to_encode_partial)
            to_send.extend(inner_enc)

        to_send = vector(GF(2), to_send)

    return [int(x) for x in to_send]


for task in from_pulsar:
    to_send = handle_task(task)
    to_pulsar.append(to_send)

print(json.dumps(to_pulsar))
