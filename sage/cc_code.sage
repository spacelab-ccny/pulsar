# Need to preprocess to use in other code
# sage --preprocess cc_code.sage
# mv cc_code.sage.py cc_code.py

# point here is to explore what appears to be achievable
# theory results say that -> we can construct binary concatenated codes that achieve list decoding capacity, with high probability, using a folded RS code as the outer code and a random lin. code as the inner code
# however, even if this happens with high probability, it doesn't mean their is an efficient algorithm to decode and *we have no way of knowing if we actually picked a good code*
# anything provable has unacceptably low bounds

# Reference work for this code:
# [1] "Permuted Successive Cancellation Decoder for Polar Codes", Harish Vangala, Emanuele Viterbo, and Yi Hong, ISITA2014
# [2] "Channel polarization: A method for constructing capacity achieving codes for symmetric binary-input memoryless channels", Erdal Arikan <-- this is not an i, change it, IEEE Transactions on Information Theory 2009
# [3] "Information Theory Strikes Back: Polar Codes", Essential Coding Theory, Venkatesan Guruswami, Atri Rudra, and Madhu Sudan
# [4] "A Performance Comparison of Polar Codes and Reed-Muller Codes", Erdal Arikan, IEEE Communications Letters June 2008

# DISCLAIMER: I am not a software engineer, I know this is not good code
import math
from multiprocessing import get_context, Process, Pool


class PolarCode():
    def constructR(self, n):
        # this constructs a reverse shuffle operator
        # take a row vector, replace the entry at index i bin(i) = b_1 ... b_n
        # with the one at b_2 ... b_n b_1
        # this matrix has 2**n rows and columns (it is a permutation matrix)
        N = 2**n
        mats_to_c = [0]*(N)
        halfway_pt = int(N/2)
        for i in range(halfway_pt):
            first = zero_matrix(N, 1)
            first[2*i] = 1
            mats_to_c[i] = first
            sec = zero_matrix(N, 1)
            sec[2*i+1] = 1
            mats_to_c[i+halfway_pt] = sec
        ret = block_matrix(mats_to_c, subdivide=False, ncols=N)
        return ret

    def constructB(self):
        # this is a construction of the bit reversal matrix
        # as explained in [2], to make decoding a bit easier we apply it
        # at the end of the decoding procedure

        # construct through recursion: B_N = R_N (I_2 \tensor B_{N/2}), B_2 = I_2
        I_2 = identity_matrix(GF(2), 2)
        B = copy(I_2)
        for i in range(self._pow_of_two-1):
            B_part = I_2.tensor_product(B, subdivide=False)
            B = self.constructR(i+2) * B_part
        return B
    """
    def copy_constructor(self, orig):
        self._pow_of_two = orig._pow_of_two
        self._err_rate = orig._err_rate
        self._N = N 
        self._dimension = orig._dimension
        self._indices = orig._indices
        self.P = copy(orig.P)
        self.B = copy(orig.B)
    """

    def __init__(self, power_of_two, err_rate, dsd_r):
        if power_of_two == 0:
            raise ValueError("Block size is too small!!!")
        self._pow_of_two = power_of_two
        self._err_rate = err_rate
        # We'll use the (easy) code construction, from Arikan
        # That makes use of the heuristic that says for an arbitrary
        # binary channel with capacity C, we will assume we can get to
        # the correct set as if we assumed a BEC with the same capacity
        # the capacity of a binary symmetric channel is 1 - H(p)
        # for us, p is the error rate
        # the computation of the Bhattacharyya parameters is given in [1] for BEC, the heuristic conjecture is given in [4]
        sh_en = -1 * (err_rate * log(err_rate, 2) +
                      (1-err_rate)*log(1-err_rate, 2))
        N = int(2**power_of_two)
        self._N = N
        bin_cap = 1 - sh_en
        # check what is possible really quickly....
        if dsd_r > bin_cap:
            raise ValueError(
                "Trying to instantiate assuming you can beat capacity, which is impossible")
        else:
            print("Capacity with err {} is {}".format(err_rate, bin_cap))
        heur_er = 1 - bin_cap
        z_arr = self._N * [0]
        # fill the first slot
        z_arr[0] = heur_er
        for i in range(self._pow_of_two):
            real_ctr = int(2**i)
            for j in range(real_ctr):
                # you're going to be acting on two indices at once
                # real_ctr + j and j
                z_sq = z_arr[j]**2
                z_arr[j] = 2*z_arr[j] - z_sq
                z_arr[j+real_ctr] = z_sq
        pi = [(i, z_arr[i]) for i in range(self._N)]
        for i in range(self._N):
            pne = pi.index(
                max(pi[:self._N-i], key=lambda x: x[1]), 0, self._N-i)
            tmp = pi[self._N-i-1]
            pi[self._N-i-1] = pi[pne]
            pi[pne] = tmp
        # how many do I take?
        # use tau to decide this
        K = int(dsd_r*self._N)
        if K == 0:
            raise ValueError(
                "Polar Code input implies zero message dimension!")

        self._dimension = K
        pi = pi[:K]
        print("Have chosen {} indices based off of Bhatt params".format(K))
        # these row indices from the kroenecker product of P_2 will be used for encoding
        self._indices = [x[0] for x in pi]
        # this might take awhile, we're going to build the polarizing mtx first
        P_2 = matrix([[1, 0], [1, 1]])
        P = copy(P_2)
        print("Constructing the original polar matrix...")
        for i in range(self._pow_of_two-1):
            print("Constructioning P_{}...".format(2**(i+2)))
            P = P.tensor_product(P_2, subdivide=False)
        self.P = P
        print("Constructing bit reversal matrix...")
        self.B = self.constructB()
        return

    def dimension(self):
        return self._dimension

    def length(self):
        return self._N

    def encode(self, msg):
        if len(msg) != len(self._indices):
            raise ValueError(
                "Input msg dimension does not match dimension of the code")
        # the encoder as presented in [1] (modified so that x*G is the codeword instead of G*x) to match [2]
        # setting frozen bits to 0
        d = self._N * [0]
        ctr = 0
        for i in range(self._N):
            if i in self._indices:
                d[i] = msg[ctr]
                ctr += 1
        d_vec = vector(GF(2), d)
        out = d_vec * self.P
        return out  # * self.B

    # directly from [1], UpdateL function
    def updateL(self, l_m, d_m, row, col):
        s = 2**(self._pow_of_two - col)
        s_half = int(s / 2)
        l = int(row) % int(s)
        if l < s_half:
            if l_m[row][col+1] == NaN:
                self.updateL(l_m, d_m, row, col+1)
            if l_m[row + s_half][col+1] == NaN:
                self.updateL(l_m, d_m, row + s_half, col+1)
            a_p = l_m[row][col+1]
            b_p = l_m[row+s_half][col+1]
            # using the approximation mentioned in [5]
            # which apparently comes from BP decoding for LDPC codes
            # this is specifically an approx. of the "box" operator
            # OLD LIKELIHOOD: l_m[row][col] = (a_p*b_p + 1.0) / (a_p + b_p)
            res = a_p.sign() * b_p.sign() * min(a_p.abs(), b_p.abs())
            # print("Setting L[{}][{}] = {}".format(row, col, res))
            l_m[row][col] = res
        else:
            if d_m[row - s_half][col] == 0:
                # OLD LIKELIHOOD: l_m[row][col] = l_m[row][col+1]*l_m[row-s_half][col+1]
                l_m[row][col] = l_m[row][col+1] + l_m[row-s_half][col+1]
            else:
                # OLD LIKELIHOOD: l_m[row][col] = l_m[row][col+1] / l_m[row-s_half][col+1]
                l_m[row][col] = l_m[row][col+1] - l_m[row-s_half][col+1]
        return

    # directly from [1], UpdateB function
    def updateD(self, d_m, row, col):
        s = 2**(self._pow_of_two - col)
        s_half = int(s / 2)
        l = int(row) % int(s)
        if l < s_half or col >= self._pow_of_two:
            return
        else:
            # GF(2) add is XOR
            if d_m[row][col] == NaN or d_m[row-s_half][col] == NaN:
                raise ValueError(
                    "Attempting to use a decision value that hasn't been set yet!!")
            d_m[row-s_half][col+1] = d_m[row][col] + d_m[row-s_half][col]
            d_m[row][col+1] = d_m[row][col]
            self.updateD(d_m, row, col+1)
            self.updateD(d_m, row-s_half, col+1)
        return

    # NOTE: may need to change this to log scale domain
    def decode_to_message(self, word):
        # print("Received word: {}".format(word))
        # This is the SCD of [2] as presented in [1] (w. associated helper functions)
        L = []
        for i in range(self._N):
            new_r = [NaN] * (self._pow_of_two+1)
            L.append(new_r)
        D = []
        for i in range(self._N):
            new_r = [NaN] * (self._pow_of_two+1)
            D.append(new_r)
        # working from right to left, we have in the two case
        #    u_1  ------>  \oplus ------>    W_1    -------> y_1
        #                     ^
        #                     |
        #    u_2  ------>         ------>    W_2    -------> y_2
        #
        # we know y_1, y_2 (must guess u_1 and u_2)
        # first, we treat u_2 as random noise and calculate
        # W(y_1,y_2 | u_1) for u_1 \in {0,1} (prob. is over different values of u_2)
        # pick the u_1 that is more likely
        # then, assume your u_1 was correct and calculate \bar{W}(y_1,y_2, u_1 | u_2) for
        # u_2 \in {0,1}, then choose the best u_2. For N > 2, we just do this repeatedly many
        # times ( the network is such that once u_1 and u_2 are found, you just go back another
        # layer and u_1, u_2 become y'_1 and y'_2 ... etc.

        BSC_TPM = [[1-self._err_rate, self._err_rate],
                   [self._err_rate, 1-self._err_rate]]

        # first updating the last column of L
        for i in range(self._N):
            # L[i][-1] = BSC_TPM[0][word[i]] / BSC_TPM[1][word[i]]
            L[i][-1] = log(BSC_TPM[0][word[i]]) - log(BSC_TPM[1][word[i]])
        list_of_idxs = [x for x in range(self._N)]
        # might need to change the ring for this matrix for this to work....
        list_of_idxs = vector(list_of_idxs) * self.B.lift()
        bin_F = GF(2)
        for i in list_of_idxs:
            # print("Going through and updating index {}".format(i))
            # print_mtx(L)
            self.updateL(L, D, i, 0)
            # print("After update, likelihood matrix is ")
            # print_mtx(L)
            if i not in self._indices:
                # print("Setting (frozen) index {} to 0".format(i))
                D[i][0] = bin_F(0)
            else:
                # OLD LIKELIHOOD: if L[i][0] < 1:
                if L[i][0] < 0:
                    D[i][0] = bin_F(1)
                    # print("Setting (non-frozen) index {} to 1".format(i))
                else:
                    D[i][0] = bin_F(0)
                    # print("Setting (non-frozen) index {} to 0".format(i))
            self.updateD(D, i, 0)
            # print("Decision mtx after update")
            # print_mtx(D)
        # take all the bit decisions from the first column of the decision matrix
        msg = []
        for i in range(self._N):
            if i in self._indices:
                msg.append(D[i][0])
        return vector(GF(2), msg)


def decode_individ(idx, code, individ_w):
    msg = code.decode_to_message(individ_w)
    return (idx, msg)


class ConcatenatedCode():
    def __init__(self, outer_code, inner_code, ic_name):
        self._N = outer_code.length()*inner_code.length()

        # checking dimensions
        if inner_code.dimension() != int(math.log(outer_code.base_field().order(), 2)):
            raise ValueError("IC dimension {} should be equal to log of OC field size {}".format(
                inner_code.dimension(), outer_code.base_field().order()))

        self._K = outer_code.dimension()*inner_code.dimension()
        self._oc = outer_code
        self._of = outer_code.base_field()
        self._ic = inner_code
        self._ic_name = ic_name

    @classmethod
    def from_file(cls, line_f):
        inner_c_name = line_f[-2]
        inner_c_details = line_f[-1]
        ic_len = int(line_f[-3])
        ic_dim = int(line_f[-4])
        inner_code = None
        if inner_c_name == "ReedMuller":
            num_vars = int(math.log(ic_len, 2))
            inner_code = codes.ReedMullerCode(
                GF(2), int(inner_c_details), num_vars)
        elif inner_c_name == "BCH":
            inner_code = codes.BCHCode(GF(2), ic_len, int(inner_c_details))
        elif inner_c_name == "RandomLinear":
            # make a matrix from the list
            code_mtx = Matrix(GF(2), inner_c_details)
            inner_code = codes.LinearCode(code_mtx)

        if inner_code == None:
            raise ValueError(
                "Don't support this inner code for constructing from file. Quitting...")

        oc_len = int(line_f[2])
        oc_dim = int(line_f[1])
        F = GF(2**ic_dim)
        list_elts = []
        if len(F) > 4096:
            for z in range(i):
                new_elt = F.random_element()
                while new_elt in list_elts:
                    new_elt = F.random_element()
                list_elts.append(new_elt)
        else:
            list_elts = F.list()[:oc_len]
        outer_code = codes.GeneralizedReedSolomonCode(list_elts, oc_dim)

        return cls(outer_code, inner_code, inner_c_name)

    def serialize(self):
        rate = float(self._K) / self._N
        cstr = [rate, int(self._oc.dimension()), int(self._oc.length()), int(
            self._ic.dimension()), int(self._ic.length()), self._ic_name]
        if self._ic_name == "ReedMuller":
            cstr.append(int(self._ic.order()))
        elif self._ic_name == "BCH":
            cstr.append(int(self._ic.designed_distance()))
        elif self._ic_name == "RandomLinear":
            to_recov = self.get_lin_mtx()
            cstr.append(to_recov)
        else:
            raise ValueError(
                "Cannot serialize yet type {}. This is not supported :(".format(self._ic_name))
        return cstr

    def get_lin_mtx(self):
        to_recov = self._ic.generator_matrix().rows()
        resp = []
        for row in to_recov:
            as_list = row.list()
            as_list_ints = [int(x) for x in as_list]
            resp.append(as_list_ints)
        return resp

    def length(self):
        return self._N

    def dimension(self):
        return self._K

    def encode_random_msg(self):
        rmv = random_vector(self._of, self._of.dimension())
        cw = self.encode(rmv)
        return cw

    def encode(self, msg):
        # apply the outer code first
        outer_cw = self._oc.encode(msg)
        # encode each symbol from the outer codeword
        full_out = []
        for symbol in outer_cw:
            as_bv = vector(GF(2), list(symbol))
            full_out.extend(self._ic.encode(as_bv))
        return vector(GF(2), full_out)

    def decode(self, codeword):
        # decode the inner code first
        bin_decoder = None
        if self._ic_name == "ReedMuller" or self._ic_name == "BCH" or self._ic_name == "RandomLinear" or "Hamming":
            bin_decoder = self._ic.decoder("NearestNeighbor")
        elif self._ic_name == "PolarCode":
            bin_decoder = self._ic
        inner_len = self._ic.length()
        decoded_fl = []
        # we're going to try to be more efficient
        list_args = []
        for i in range(0, len(codeword), inner_len):
            word_as_vec = vector(GF(2), codeword[i:i+inner_len])
            idx = int(i / inner_len)
            list_args.append((idx, bin_decoder, word_as_vec))
        # macOS: https://stackoverflow.com/a/67999590
        with get_context("fork").Pool() as pool:
            out = pool.starmap(decode_individ, list_args)
            # re-shuffle to correct location
            out.sort(key=lambda elem: elem[0])
            for x in out:
                decoded_fl.extend(x[1])
            # decode_in_cw = bin_decoder.decode_to_message(word_as_vec)
            # decoded_fl.extend(decode_in_cw)
        bit_l = int(math.log(self._of.cardinality(), 2))
        from_inner = [self._of(decoded_fl[i:i+bit_l])
                      for i in range(0, len(decoded_fl), bit_l)]
        from_inner = vector(self._of, from_inner)
        if len(from_inner) != self._oc.length():
            raise ValueError("Size discrepancy!!")
        try:
            col_m = self._oc.decode_to_message(from_inner)
            return col_m
        except sage.coding.decoder.DecodingError:
            # GSD = codes.decoders.GRSGuruswamiSudanDecoder
            # tst_dist = int(self._oc.minimum_distance()*(6.0/10))
            # if tst_dist > int(self._oc.minimum_distance()/2):
            #    dec_gs = GSD(self._oc, tst_dist)
            #    if dec_gs.parameters()[0] <= 2:
            #        try:
            #            col_m = dec_gs.decode_to_message(from_inner)
            #            return col_m
            #        except:
            #            return None
            return None
        except Exception as e:
            print("Caught a separate decoding error. I think this is a sage math bug.")
            return None

    def print_code(self):
        print("Outer code is ")
        print(self._oc)
        print("Inner code is ")
        print(self._ic)
