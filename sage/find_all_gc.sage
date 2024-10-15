
import math
import json
import time
import os.path
import argparse
import sys
from operator import itemgetter
from multiprocessing import Process, Pool
from cc_code import PolarCode, ConcatenatedCode

NUM_COPIES = 400

def enc_with_to(cc, rmv):
    cw = cc.encode(rmv)
    return

def dec_with_to(cc, cw):
    msg = cc.decode(cw)
    return

def test_polar_codes():
    power_of_two = 9
    err = 0.05
    pc = PolarCode(power_of_two, err, 0.55)
    # choose a random message  
    msg_len = pc.dimension()
    # no errs test
    import pdb; pdb.set_trace() 
    passed_all = True
    TEST_AMT = 10
    """
    for i in range(TEST_AMT):
        print("Test {}".format(i))
        msg = random_vector(GF(2), msg_len) 
        cw = pc.encode(msg)
        msg_decoded = pc.decode_to_message(cw)
        if msg != msg_decoded:
            passed_all = False
            print("Failed to decode when decoding should have passed!!!")
    if passed_all:
        print("Passed {} no-error tests with no errors!".format(TEST_AMT))
    """
    # attempting a channel with errors...
    block_len = 2**power_of_two
    chan = channels.QarySymmetricChannel(GF(2)^block_len,err)
    failures = 0
    for i in range(TEST_AMT):
        msg = random_vector(GF(2), msg_len) 
        cw = pc.encode(msg)
        word = chan.transmit(cw)
        msg_decoded = pc.decode_to_message(word)
        if msg != msg_decoded:
            failures += 1
    print("Failed {}/{}".format(failures,TEST_AMT)) 
    return    

def test_code(cc, omk, err_rate):
    scs = 0
    block_len = cc.length()
    chan = channels.QarySymmetricChannel(GF(2)^block_len,err_rate)
    # we start here by just seeing if encoding 
    # with these parameters is efficient
    rmv = random_vector(cc._of, omk) 
    pr = Process(target=enc_with_to, args=(cc, rmv))
    pr.start()
    pr.join(timeout=60)
    if pr.exitcode == None:
        print("Timed out while encoding... not using this code")
        return False, 0
    # attemtping decoding, 
    # if this is too slow also give up 
    cw = cc.encode(rmv)
    pr = Process(target=dec_with_to, args=(cc, cw))
    pr.start()
    pr.join(timeout=60)
    if pr.exitcode == None:
        print("Decoding takes >1 minute... not using this code")
        return False, 0
    NUM_RUNS = 40
    for idc in range(NUM_RUNS):
        rmv = random_vector(cc._of, omk)
        cw = cc.encode(rmv)
        #actually injecting errors here 
        rcv = chan.transmit(cw)
        #start_de = time.time()
        msg = cc.decode(rcv) 
        #total = time.time() - start_de
        #print("Took {} seconds to decode".format(total))
        if msg == rmv:
            #print("Was a success")
            scs += 1 
        if idc == 1 and scs == 0:
            print("Failing out after two attempts")
            return False, 0
        if (idc+1) % 10 == 0:
            prob_scs = float(scs)/(idc+1)
            if prob_scs < .90:
                print("Failed out with prob. of success {}".format(prob_scs))
                return False,prob_scs
            else:
                print("Survived {} attempts so far".format(idc+1))
    return True,float(scs)/NUM_RUNS

def first_finding_step(err_rate, N):                          
    rates_to_try = set()
    best_possible_rate = 1-(-err_rate*math.log(err_rate,2)-(1-err_rate)*math.log(1-err_rate,2))
    for ocl in range(N,1, -1):
        print("Going through run with outer code length: {}".format(ocl))  
        inner_code_len = int(N / ocl)
        # using list decoding probability as a bound on the rate/error trade-off that is achievable, look for
        # use the nearest field that is larger than actual_N and a power of 2
        RS_field_size = ocl.bit_length()
        if 2**(ocl.bit_length()-1) == ocl:
            RS_field_size -= 1
        if RS_field_size == 0:
            continue
        F = GF(2**RS_field_size)
        inner_msg_len = RS_field_size
        upp_bound_K = int(best_possible_rate*N)
        cc = None
        for poss_K in range(upp_bound_K, int(upp_bound_K/4), -1):
            outer_msg_len = int(poss_K / inner_msg_len)
            # now we build both codes
            if ocl < outer_msg_len+2 or inner_code_len < inner_msg_len+2:
                continue
            # trying to prioritize those codes where the inner code could potentially have high relative distance
            if float(inner_msg_len)/inner_code_len <= 0.50:
                #print("Appending a code to rates_to_try")
                rates_to_try.add((int(ocl), int(outer_msg_len), int(inner_code_len), int(inner_msg_len)))
    rates_to_try = list(rates_to_try) 
    rates_to_try.sort(key=lambda elem: float(elem[1]*elem[3])/(elem[0]*elem[2]))
    return rates_to_try 

def decode_time(decoder, msg, w):
    out = decoder.decode_to_message(w)
    return msg != out

def test_channel(ic, decoder, copies, err):
    TEST = 1
    print("Number of copies is {}".format(copies))
    chan = channels.QarySymmetricChannel(GF(2)^ic.length(),err) 
    simple_tst = random_vector(GF(2),ic.dimension())
    word = ic.encode(simple_tst)
    pr = Process(target=decode_time,args=(decoder,simple_tst,word))
    pr.start()
    pr.join(timeout=60)
    if pr.exitcode == None:
        print("Timed out while decoding inner code... not using this code")
        return 1
    errs = []
    for i in range(TEST):
        failed = 0
        list_args = []
        for _ in range(copies):
            msg = random_vector(GF(2),ic.dimension())
            cw = ic.encode(msg)
            w = chan.transmit(cw)
            list_args.append((decoder, msg, w))
        with Pool() as pool:
            out = pool.starmap(decode_time, list_args)
            failed = sum(out)
        err = float(failed) / copies 
        print("Appending to errors: {}".format(err))
        errs.append(err)
    return float(sum(errs)) / len(errs)


class BinSearch():
    def __init__(self,low,high):
        self._ch = high
        self._cl = low
        self._mid = int((self._cl + self._ch)/2)
        self._fr = True
    
    def next(self):
        if self._mid == self._cl and not self._fr:
            return None
        if self._fr:
            self._fr = False
        return self._mid

    def update_choice(self, choice):
        if choice == "L":
            self._ch = self._mid
        elif (self._ch - self._cl) == 1:
            self._cl = self._ch
            self._fr = True
        else: 
            self._cl = self._mid
        self._mid =  int((self._cl + self._ch)/2)
        
        return

# return TRUE/FALSE depending on if you succeeded or failed 
def find_outer_code(ic, ic_err, oc_len, F, ic_name, already_found_rates):
    double_for_dist = 2 * ic_err
    dist = int(math.ceil(double_for_dist*oc_len)) + 1
    # need to make sure that the RS code 
    # can handle *some* error, even if supposedly there was none
    if dist == 1:
        dist = 2
    if dist > oc_len:
        return (False, None)  

    list_elts = []
    if len(F) > 4096:
        for z in range(oc_len):
            new_elt = F.random_element()
            while new_elt in list_elts:
                new_elt = F.random_element()
            list_elts.append(new_elt) 
    else:
        list_elts = F.list()[:oc_len] 

    max_k = oc_len - dist + 1
    max_rate = float((ic.dimension()*max_k)/(ic.length()*oc_len))
    max_rate = float(int(100*max_rate)) / float(100)

    for j in range(dist, oc_len):     
        k = oc_len - j + 1

        rate = float((ic.dimension()*k)/(ic.length()*oc_len))
        rate = float(int(100*rate)) / float(100)
        rrate = int(100*rate)        
        for f_rate in already_found_rates:
            fr_rate = int(f_rate*100)
            if fr_rate >= rrate:
                return (False, None)
           
        print("Attempting a code with rate: {}".format(rate))
    
        outer_code = codes.GeneralizedReedSolomonCode(list_elts, k)
        cc = ConcatenatedCode(outer_code, ic, ic_name)  
        suc, prob_suc = test_code(cc, k, err_rate)
        print("Testing Conc. Code: {}, probability of success: {}".format(suc, prob_suc))
        if suc:
            return (True, (cc, err_rate, prob_suc))

    return (False, None)


def find_good_heuristic_code(err_rate, N, write_f):
    if not os.path.isdir('./cache'):
        os.mkdir('./cache')
    cache = './cache/{}-{}'.format(N, err_rate)
    res_to_try = []
    if os.path.isfile(cache):
        with open(cache, "r") as f_obj:
            res_to_try = json.loads(f_obj.read())
    else:
        res_to_try = first_finding_step(err_rate,N)
        with open(cache, 'w') as f_obj:
            f_obj.write(json.dumps(res_to_try))

    res_to_try.reverse()

    CAP = 1 + (err_rate*log(err_rate,2) + (1-err_rate)*log(1-err_rate,2))
    print("Expected capacity is {}".format(float(CAP)))
    #res = input("pick a different cut-off?")
    #if res.startswith("Y") or res.startswith("y"):
    #    res2 = input("what index?")
    #    try:
    #        idx = int(res2)
    #        res_to_try = res_to_try[idx:]
    #    except:
    #        print("Not a real index, starting from beginning...")

    #res_to_try = res_to_try[4196:]
    #ctr = 4196
    mao = 2
    low_bound_test_len = int(N^((mao-1)/mao))
     
    """
    print("Testing Polar")
    succeeded_with = []
    
    for i in range(5, 11):
        h_rate_fail = 1
        inner_code_len = 2**i
        outer_code_len = int(N/inner_code_len)
        # use capacity to drive this, you likely can't get up to capacity but perhaps you could get close?
        pot_rate = CAP - 0.20 
        if pot_rate < 0.05:
            pot_rate = 0.05

        while pot_rate <= CAP:
            pc = PolarCode(i, err_rate, pot_rate)
            perct_failed_recov = test_channel(pc, 20, err_rate)
            F = GF(2**pc.dimension())
            if outer_code_len > len(F):
                print("Can't use this code because the field size is too small for the code length")
                continue 
            double_for_dist = 2 * perct_failed_recov
            dist = int(double_for_dist*outer_code_len)
            k = outer_code_len - dist + 1
            print("RS code dimension is {}, outer length {}".format(k, outer_code_len))
            if k < 1:
                print("Msg. Dimension must be positive")
                continue
            rate = float((pc.dimension()*k)/(inner_code_len*outer_code_len))
            rate = float(int(100*rate)) / float(100)
            if rate >= h_rate_fail:
                print("We've already had a code fail at a higher rate, skipping")
                continue
            skip = False
            for r in succeeded_with:
                if r >= rate:
                    # just quit now
                    print("This code rate is worse or the same as a previously passing rate, continuing on...")
                    skip = True
            if skip:
                continue
            print("Attempting a code with rate: {}".format(rate)) 
            list_elts = []
            if len(F) > 4096:
                for z in range(outer_code_len):
                    new_elt = F.random_element()
                    while new_elt in list_elts:
                        new_elt = F.random_element()
                    list_elts.append(new_elt) 
            else:
                list_elts = F.list()[:outer_code_len] 
            if k <= 0 or rate > CAP or k > outer_code_len:
                print("Polar Code, failed out")
                continue 
            print("RS actual code_len {}, msg_len {}".format(outer_code_len, k))
            outer_code = codes.GeneralizedReedSolomonCode(list_elts, k)
            cc = ConcatenatedCode(outer_code, pc, "PolarCode")  
            suc, prob_suc = test_code(cc, k, err_rate)
            if suc:
                #write down the code 
                succeeded_with.append(rate)
                with open("potential_codes_with_good_rates.txt","a") as f_obj:
                    print("Succeeded with polar code")
                    save_this = (float(err_rate), prob_suc, float(rate), int(k), int(outer_code_len), int(inner_code.dimension()), int(inner_code.length()), "Polar Code:", int(i), float(pot_rate))
                    f_obj.write(json.dumps(save_this))
                    f_obj.write("\n")
            else:
                #h_rate_fail = rate 
                # you won't be able to get higher
                pot_rate = CAP + 1
            pot_rate += 0.05
    """   
    
    already_tried = set()
    succeeded_with = []
    print("Testing BCH ")
    for i in range(low_bound_test_len, N):
        print("Outer code length {}".format(i))
        inner_code_len = int(N / i)
        if inner_code_len in already_tried:
            continue
        else: 
            already_tried.add(inner_code_len)
        # try and binary search for this? I guess...
        bs = BinSearch(2, inner_code_len)
        cond_halt = False
        print("Binary search on distance...")
        while not cond_halt:
            j = bs.next()
            if j == None:
                cond_halt = True
                continue
            print("Trying dist {}".format(j))
            d_r = int(inner_code_len/j)
            try:   
                inner_code = codes.BCHCode(GF(2), inner_code_len, d_r)
            except:
                print("Could not construct a code via constructor")
                bs.update_choice("R")
                continue
            # must be true that inner code 2^k'.
            if i > 2^(inner_code.dimension()):
                print("BCH dimension too low, decrease dist")
                # increase dim, decrease dist, increase j
                bs.update_choice("R")
                continue # you cannot use this
            if inner_code.dimension() > 11:
                print("BCH code dim too high, increase dist")
                # decrease dim, increase dist, decrease j
                bs.update_choice("L")
                continue 
            # you should be able to use this, so you can attempt to 
            F = GF(2**inner_code.dimension())
            # how high should you go on the outer msg rate? unclear. 
            # k \leq n-d+1 and d/2. If I want to correct 10% error... 
            # dist. 20% symbol corruption? 
            #dist = int(0.20*i)
            dec = inner_code.decoder("NearestNeighbor")
            perct_failed_recov = test_channel(inner_code, dec, NUM_COPIES, err_rate)
            if perct_failed_recov == 1:
                bs.update_choice("L") 
                continue

            found, obj_v = find_outer_code(inner_code, perct_failed_recov, i, F, "BCH", succeeded_with)
            if found: 
                cc, err_rate, prob_s = obj_v 
                rate = float(cc.dimension()/cc.length())
                print("Found a code with rate {}".format(rate))
                succeeded_with.append(rate)
                bs.update_choice("L")
                with open(write_f,"a") as f_obj:
                    list_v = [float(err_rate), prob_s] + cc.serialize()
                    save_this = tuple(list_v)
                    f_obj.write(json.dumps(save_this))
                    f_obj.write("\n")
                    break     
            else: 
                bs.update_choice("R")

    # test some bin reed muller 
    print("Testing Binary Reed Muller") 
    succeeded_with = []
    # find power of two closest to low_bound_test_len and just go from like
    # 4 up to that? 
    
    for outer_code_len in range(low_bound_test_len, N):
        success = False
        ic_code_len = int(N/outer_code_len)
        m = ic_code_len.bit_length() 
        if 2**m > ic_code_len:
            m -= 1
            ic_code_len = 2**m        
        if m == 0:
            continue
        for order in [1,2,3]:
            # r <= m 
            if order > m: 
                print("Can't use rm code because order is greater than number of variables") 
                continue
            inner_code = codes.ReedMullerCode(GF(2), order, m) 
            if inner_code.dimension() > 12:
                # skipping this 
                print("Inner code dimension is too large") 
                continue
            # go as far as you can, tolerating error
            F = GF(2**inner_code.dimension())
            if outer_code_len > len(F):
                print("Can't use this code because the field size is too small for the code length")
                continue 
            # how high should you go on the outer msg rate? unclear. 
            # k \leq n-d+1 and d/2. If I want to correct 10% error... 
            # dist. 20% symbol corruption? 
            dec = inner_code.decoder("NearestNeighbor")
            perct_failed_recov = test_channel(inner_code, dec, NUM_COPIES, err_rate)
            if perct_failed_recov == 1: 
                continue

            found, obj_v = find_outer_code(inner_code, perct_failed_recov, outer_code_len, F, "ReedMuller", succeeded_with)
            if found: 
                cc, err_rate, prob_s = obj_v 
                rate = float(cc.dimension()/cc.length())
                print("Found a code with rate {}".format(rate))
                succeeded_with.append(rate)
                bs.update_choice("L")
                with open(write_f,"a") as f_obj:
                    list_v = [float(err_rate), prob_s] + cc.serialize()
                    save_this = tuple(list_v)
                    f_obj.write(json.dumps(save_this))
                    f_obj.write("\n")
                    break     
            else: 
                bs.update_choice("R")

    
    print("Testing Random Lin. Code")
    
    start_interv = 0
    end_interv = len(res_to_try)-1
    succeeded_with = []
    bs = BinSearch(0, len(res_to_try)-1)
    itr = bs.next()
    while itr != None:
        # the front of this list has the highest rate
        rate_info = res_to_try[itr]
        # from this point, do a binary search 
        #for rate_info in res_to_try:
        #print("This is attempt {}".format(ctr))
        #ctr += 1
        outer_code_len, outer_msg_len, inner_code_len, inner_msg_len = rate_info
        print("Lower: {}, Upper: {}, Current Idx:{}".format(start_interv, end_interv, itr))
        print("msg. rate {}".format((inner_msg_len*outer_msg_len)/(outer_code_len*inner_code_len)))
        F = GF(2**inner_msg_len)
        block_K = inner_msg_len*outer_msg_len
        block_N = outer_code_len*inner_code_len
        rate = float(block_K)/float(block_N) 
        rate = float(int(100*rate)) / float(100)
        if block_K <= 0 or rate > CAP:
            print("Skipping everything b.c of error in block_K or rate is higher than capacity!")
            itr = None
            continue  
        skip = False
        for r in succeeded_with:
            if r >= rate:
                # just quit now
                print("This code rate is worse or the same as a previously passing rate, continuing on...")
                skip = True
                break
        if skip:
            itr = None 
            continue
        list_elts = []
        if len(F) > 4096:
            for z in range(outer_code_len):
                new_elt = F.random_element()
                while new_elt in list_elts:
                    new_elt = F.random_element()
                list_elts.append(new_elt) 
        else:
            list_elts = F.list()[:outer_code_len] 
        outer_code = codes.GeneralizedReedSolomonCode(list_elts, outer_msg_len)
        success = False
        for _ in range(2):
            inner_code = codes.random_linear_code(GF(2),inner_code_len, inner_msg_len)
            cc = ConcatenatedCode(outer_code, inner_code, "RandomLinear")
            print("Testing Code")
            cc.print_code()
            suc, prob_suc = test_code(cc, outer_msg_len,err_rate)
            if suc:
                success = True
                print("This code achieves rate: {}, Probability of success: {}".format(rate, prob_suc))
                #write down the code 
                with open(write_f,"a") as f_obj:
                    succeeded_with.append(rate)
                    list_v = [float(err_rate), prob_suc] + cc.serialize()
                    save_this = tuple(list_v)
                    f_obj.write(json.dumps(save_this))
                    f_obj.write("\n")
                break
        
        if success:
            bs.update_choice("L")
            
        else: 
            bs.update_choice("R")
        itr = bs.next()


# to use tool, need to supply: desired error rate, block size n, a file to write best codes
if __name__ == "__main__":
    #N = 32640 
    #for i in range(6):
    #    err_rate = 0.05*(i+1)
    #    find_good_heuristic_code(err_rate, N)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "n",
        type=int,
        help="Desired codeword length for the concatenated code"
    )

    parser.add_argument(
        "errRate",
        type=float,
        help="Desired (heuristic) error rate code should handle"
    )

    parser.add_argument(
        "filename",
        type=str,
        help="File to write codes to"
    )

    args = parser.parse_args()
    fname = args.filename
    n = args.n
    err_rate = args.errRate
    if err_rate < 0 or err_rate >= 1:
        raise ValueError("Error rate not in the correct range!")

    find_good_heuristic_code(err_rate, n, fname)

#if os.path.isfile(fname):
#    with open(fname, "r") as f_obj:
#        all_lines = f_obj.readlines()
#        all_lines = all_lines[1:]
#        for line in all_lines:
#            strip_line = json.loads(line)
#            try_loading_code = strip_line[2:]
#            cc = ConcatenatedCode.from_file(try_loading_code)


#find_good_heuristic_code(0.31,32640)    
