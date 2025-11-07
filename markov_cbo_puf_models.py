from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
import random
import os
from copy import deepcopy,copy

def gen_challenge(num = 10_000, chal_len = 64, seed = None):
    if seed != None:
        np.random.seed(int(seed))

    chal = np.random.randint(0, 2, [num, chal_len])
    return chal

class puf_basic:
    @classmethod
    def ml_dat_gen(cls, chal, resp_folder):
        chal_numb = 1000_000
        stage_n = [32, 64, 128]
        path_k = [2,3,4,5,6]

        if not os.path.isdir(resp_folder):
            os.makedirs(resp_folder)

        for n in stage_n:
            for k in path_k:
                puf = cls.gen_new_puf(n, k)

                resp = cls.gen_resp(puf, chal[0:chal_numb, 0:n])

                print('ML data gen: chal_num = %d, chal_bits= %d, xor_num = %d, uniformity = %.4f'\
                    %(chal_numb, n, k, np.sum(resp)/chal_numb))

                puf_name = 'puf_'+str(n)+'_'+str(k)
                np.save(resp_folder + puf_name, puf)
                resp_name = 'puf_'+str(n)+'_'+str(k)+'_res_1M'
                np.save(resp_folder + resp_name, resp)

    @classmethod
    def gen_CRPs_PUF(cls, n, k, chal_num, seed=None, m=8):
        """
        Generate Challenge-Response Pairs (CRPs).

        Parameters:
        - n: number of stages in the PUF.
        - k: number of paths (XOR degree).
        - chal_num: number of challenges to generate.
        - seed: random seed (optional).
        - m: group length used by CBO mechanism (default 8).

        Returns:
        - crps_dat: challenge-response pairs with shape (chal_num, n + 1).
        - puf1: the generated PUF instance.
        """
        chal_numb = chal_num
        chal_bits = n
        xor_num = k
        print('stage=%d, m=%d, k=%d'\
            %(n,m,k))
        if seed is not None:
            np.random.seed(int(seed))

        # Generate all challenges at once
        chal = np.random.randint(0, 2, size=[chal_numb, chal_bits], dtype=np.int8)

        # Create a PUF instance
        puf1 = cls.gen_new_puf(chal_bits, xor_num)

        # Initialize response array
        resp = np.zeros(chal_numb, dtype=np.int8)
        curr_delayDiff = None  # No previous delayDiff at start

        # Process each challenge sequentially (to support CBO state)
        for i in range(chal_numb):
            C = chal[i]  # current challenge
            resp[i], curr_delayDiff = cls.gen_resp_single(puf1, C, curr_delayDiff, m=m)

        # Concatenate challenges and responses
        crps_dat = np.concatenate((chal, resp.reshape(chal_numb, 1)), axis=1)

        return crps_dat, puf1

    def gen_CRPs_multiPUFs(self, resp_folder):
        """
        Generate CRPs of (32/64/128, 2/4/6/8)-OIPUF.
        For each setting, 100_000 CRPs will be generated.
        All the CRPs are stored into 'resp_folder'
        """
        test_numb = 1
        chal_numb = 100000
        n_set = [32, 64, 128]
        k_set = [2,4,6,8]

        if not os.path.isdir(resp_folder):
            os.makedirs(resp_folder)

        rand = [0.0] * test_numb
        for chal_bits in n_set:
            for xor_num in k_set:
                print('OI PUF Test: chal_number = %d, chal_bits= %d, xor_num = %d'\
                    %(chal_numb, chal_bits, xor_num))
                chal = np.random.randint(0,2,size=[chal_numb, chal_bits],dtype=np.int8)
                puf = self.gen_new_puf(chal_bits, xor_num)
                time1 = time()
                resp = self.gen_resp(puf, chal)
                rand = np.sum(resp) / chal_numb
                print(rand, time() - time1)
                np.save(resp_folder+'resp'+str(chal_bits)+'_'+ str(xor_num)+'.npy', resp)
                np.save(resp_folder+'chal'+str(chal_bits)+'_'+ str(xor_num)+'.npy', chal)

class xor_puf(puf_basic):
    var = 1
    type = 'XORPUF'
    def __init__(self, processe_var = 1):
        pass
        #self.var = processe_var # standard deviation of normal MUX

    def read_puf(self, file):
        puf = np.load(file, allow_pickle=True).items()
        return puf

    @classmethod
    def gen_new_puf(cls, chal_bits, path_k, average_delay = 4):
        stage_n = chal_bits
        puf = {}
        puf['par'] = np.random.normal(0, cls.var, [2, stage_n, path_k])
        puf['type'] = cls.type
        return puf

    @staticmethod
    def get_linear_vector(C, prev_delayDiff=None, m=8):
        """
        Transform the challenge C according to the CBO mechanism and produce a linear vector.

        Parameters:
        - C: single challenge vector of shape (stage_n,).
        - prev_delayDiff: previous internal outputs of K sub-APUFs (delayDiff), shape (path_k,).
        - m: group length, default 8.

        Returns:
        - chal: transformed linear vector of shape (stage_n,).
        """
        stage_n = C.shape[0]  # length of the challenge
        num_groups = stage_n // m  # number of groups

        # If prev_delayDiff is None, initialize confusion vector with zeros
        if prev_delayDiff is None:
            initial_confusion = np.zeros(m, dtype=np.int8)
        else:
            # Use elements of prev_delayDiff as the high bits of the initial confusion vector
            initial_confusion = np.zeros(m, dtype=np.int8)
            prev_len = min(len(prev_delayDiff), m)
            initial_confusion[:prev_len] = prev_delayDiff[:prev_len]

            # If prev_delayDiff is shorter than m, fill the remaining bits with the high bits of C
            if prev_len < m:
                initial_confusion[prev_len:] = C[-(m - prev_len):]

        # Split C into groups
        C_groups = C.reshape(num_groups, m)  # shape (num_groups, m)

        # Apply confusion to each group
        confusion_groups = np.zeros_like(C_groups, dtype=np.int8)
        confusion_groups[0] = np.bitwise_xor(initial_confusion, C_groups[0])  # first group

        for i in range(1, num_groups):
            confusion_groups[i] = np.bitwise_xor(confusion_groups[i - 1], C_groups[i])  # subsequent groups

        # Flatten the confused groups
        C_new = confusion_groups.flatten()  # shape (stage_n,)

        # Generate linear vector: map 0 -> 1, 1 -> -1
        chal = 1 - 2 * C_new

        return chal

    @classmethod
    def gen_resp_single(cls, puf, C, prev_delayDiff=None, noise=0, offset=0, m=8):
        """
        Generate a single XOR-PUF response.

        Parameters:
        - puf: PUF instance containing delay parameters.
        - C: single challenge vector of shape (stage_n,).
        - prev_delayDiff: previous internal K sub-APUF outputs (delayDiff), shape (path_k,).
        - noise: noise level.
        - offset: offset value (unused).
        - m: group length for CBO (default 8).

        Returns:
        - resp_xor: the XOR response for this challenge (scalar).
        - curr_delayDiff: the delayDiff vector for this round (shape (path_k,)), to be used next time.
        """
        puf_par = puf['par']
        stage_n = puf_par.shape[-2]
        path_k = puf_par.shape[-1]

        # Transform challenge C using CBO mechanism
        chal = cls.get_linear_vector(C, prev_delayDiff, m)

        # Initialize delay arrays for this round
        delay_par = np.zeros((stage_n,))
        curr_delayDiff = np.zeros((path_k,))

        # Compute delayDiff for each internal APUF (k paths)
        for j in range(path_k):
            for i in range(stage_n):
                delay_par[i] = puf_par[C[i], i, j]  # select delay parameter according to challenge bit

            curr_delayDiff[j] = np.dot(delay_par, chal)  # compute delayDiff as dot product

        # Add noise to delayDiff
        delayNoise = np.random.normal(0, cls.var * noise * (stage_n ** (0.5)), (path_k,))
        curr_delayDiff = curr_delayDiff + delayNoise

        # Binarize curr_delayDiff
        curr_delayDiff[curr_delayDiff >= 0] = 1
        curr_delayDiff[curr_delayDiff < 0] = 0

        # Determine resp_xor according to grouping logic based on prev_delayDiff
        if prev_delayDiff is not None:
            # Convert prev_delayDiff to integer type
            prev_delayDiff = prev_delayDiff.astype(np.int8)

            # Initial grouping: pair adjacent bits of prev_delayDiff
            intermediate_results = []
            for i in range(0, len(prev_delayDiff), 2):
                idx1, idx2 = i, i + 1
                if prev_delayDiff[idx1] == prev_delayDiff[idx2]:
                    intermediate_results.append(curr_delayDiff[min(idx1, idx2)])  # choose the smaller index output
                else:
                    intermediate_results.append(curr_delayDiff[max(idx1, idx2)])  # choose the larger index output

            # Perform multiple rounds of selection until one result remains
            while len(intermediate_results) > 1:
                new_intermediate_results = []
                for i in range(0, len(intermediate_results), 2):
                    if i + 1 < len(intermediate_results):  # avoid out-of-range
                        if intermediate_results[i] == intermediate_results[i + 1]:
                            new_intermediate_results.append(intermediate_results[i])  # choose the smaller
                        else:
                            new_intermediate_results.append(intermediate_results[i + 1])  # choose the larger
                    else:
                        new_intermediate_results.append(intermediate_results[i])  # odd element, keep it
                intermediate_results = new_intermediate_results

            # Final result
            resp_xor = intermediate_results[0]
        else:
            # If no previous delayDiff, output the first k-th delayDiff
            resp_xor = curr_delayDiff[0]  # choose output of first path

        # Return XOR response and current delayDiff
        return int(resp_xor), curr_delayDiff

    @classmethod
    def test(cls, xor_num = 1, chal_bits = 64, chal_numb=100_000):
        print('XOR PUF Test: chal_number = %d, chal_bits= %d, xor_num = %d'\
            %(chal_numb, chal_bits, xor_num))

        chal = np.random.randint(0,2,size=[chal_numb, chal_bits])
        puf = {}
        resp = {}
        rand = np.zeros(20)
        for i in range(20):
            puf[i] = cls.gen_new_puf(chal_bits, xor_num)
            resp[i] = cls.gen_resp(puf[i], chal)
            rand[i] = np.sum(resp[i]) / chal_numb * 100
            print(rand[i])

        print('Rand=(%.2f, %.2f), chal_bits=%d, xor_num=%d'\
            %(np.mean(rand), np.std(rand), chal_bits, xor_num))



if __name__ == "__main__":
    chal_numb = 2_000_000
    crps, puf = xor_puf.gen_CRPs_PUF(64, 2 , chal_numb, m=32 )
    resp = crps[:, -1]  # responses are in the last column
    print("Uniformity is %.2f" %  (np.sum(resp) / chal_numb * 100))

    challenges = crps[:, :-1]  # all columns except last are challenges
    responses = crps[:, -1]    # last column is responses
    
    challenge_file = "challenges.txt"
    with open(challenge_file, "w") as f:
        for challenge in challenges:
            # Convert each challenge to a string, e.g. "1010..."
            challenge_str = "".join(map(str, challenge))
            f.write(f"{challenge_str}\n")  # one challenge per line

    print(f"Challenges saved to {challenge_file}")

    response_file = "responses.txt"
    with open(response_file, "w") as f:
        for response in responses:
            f.write(f"{response}\n")

    print(f"Responses saved to {response_file}")
