class MethodParams:
    def __init__(self, M, K):
        self.M = M # number of codebooks
        self.K = K # size of each codebook
        self.prefix = str(M) + '_' + str(K) + '_'
