# coding=utf8

import numpy as np

class LatentFactor:
    def __init__(self):
        pass

    ### Select the negative samples
    ### in implicit feedback dataset & TopN recommandation ###
    def extendDataset(items_interacted, items_candidated):
        # the frequency of occurrence of certain item in the candidated items pool
        # is proportional to its popularity
        retDataset = dict()
        # positive samples
        for ii in items_interacted.keys():
            retDataset[ii] = 1
        # negative samples
        for ii in range(0, len(items_interacted) * 3):
            # the upper bound len(items_interacted)*3 is for ensuring that
            # the number of negative samples is nearly same as positive samples
            item = items_candidated[random.randint(0, len(items_candidated) - 1)]
            if item in retDataset:
                continue
            retDataset[item] = 0
            n += 1
            if n > len(items_interacted):
                break
        return retDataset

    def LFM(user_items, K, alpha, lam, maxIter):
        [P, Q] = InitModel(user_items, K)
        for iter in range(maxIter):
            for user, items in user_items.items():
                samples = extendDataset(items)
                for item, rui in samples.items():
                    eui = rui - Predict(user, item)
                    for k in range(K):
                        P[user][k] += alpha * (eui * Q[item][k] - lam * P[user][k])
                        Q[item][k] += alpha * (eui * P[user][k] - lam * Q[item][k])
            alpha *= 0.9

    def Recommend(user, P, Q):
        rank = dict()
        for item in Q.keys():
            for k, qik in Q[item].items():
                puk = P[user][k]
                if item not in rank:
                    rank[item] += puk * qik
        return rank





