import cPickle as pickle
from yael import yael, ynumpy
from xvecReadWrite import *
from multiprocessing import Pool
from functools import partial
import math

def learnCodebooksPQ(learnFilename, dim, M, K, pointsCount, codebooksFilename, \
                     threadsCount=32, iterCount=30):
    if dim % M != 0:
        raise Exception('Dim is not a multiple of M!')
    else:
        vocabDim = dim / M
    codebooks = np.zeros((M, K, vocabDim), dtype='float32')
    points = readXvecs(learnFilename, dim, pointsCount)
    for m in xrange(M):
        subpoints = points[:,m*vocabDim:(m+1)*vocabDim].copy()
        (codebook, qerr, dis, assign, nassign) = ynumpy.kmeans(subpoints, K, nt=threadsCount, \
                                                               niter=iterCount, output='all')
        codebooks[m,:,:] = codebook
    codebooksFile = open(codebooksFilename, 'wb')
    pickle.dump((codebooks), codebooksFile)
    codebooksFile.close()

def encodeDatasetPQ(baseFilename, pointsCount, vocabFilename, codeFilename, threadsCount=30):
    codebooks = pickle.load(open(vocabFilename, 'rb'))
    M = codebooks.shape[0]
    dim = codebooks.shape[2] * M
    vocabDim = codebooks.shape[2]
    codes = np.zeros((pointsCount, M), dtype='int32')
    basePoints = readXvecs(baseFilename, dim, pointsCount)
    for m in xrange(M):
        codebook = codebooks[m,:,:]
        subpoints = basePoints[:,m*vocabDim:(m+1)*vocabDim].copy()
        (idx, dis) = ynumpy.knn(subpoints, codebook, nt=threadsCount)
        codes[:,m] = idx.flatten()
    codeFile = open(codeFilename, 'w')
    pickle.dump(codes, codeFile)
    codeFile.close()

def findNearestForRangePQ(rangeId, rangeSize, codebookDistances, pointsCodes, listLength):
    startQid = rangeId * rangeSize
    finishQid = min((rangeId + 1) * rangeSize, codebookDistances.shape[1])
    if startQid >= finishQid:
        return None
    nearest = np.zeros((finishQid - startQid, listLength), dtype='int32')
    for qid in xrange(startQid, finishQid):
        print 'Handling qid ' + str(qid)
        distances = np.zeros((pointsCodes.shape[0]), dtype='float32')
        for pid in xrange(pointsCodes.shape[0]):
            for m in xrange(pointsCodes.shape[1]):
                distances[pid] += codebookDistances[m, qid, pointsCodes[pid, m]]
        nearest[qid - startQid,:] = distances.argsort()[0:listLength]
    return nearest

def searchNearestNeighborsPQ(codeFilename, codebookFilename, queriesFilename, \
                             queriesCount, k, threadsCount=30):
    codebooks = pickle.load(open(codebookFilename, 'r'))
    M = codebooks.shape[0]
    dim = codebooks.shape[2] * M
    codebookDim = codebooks.shape[2]
    codebookSize = codebooks.shape[1]
    codes = pickle.load(open(codeFilename, 'rb'))
    queries = readXvecs(queriesFilename, dim, queriesCount)
    nearest = np.zeros((queriesCount, k), dtype='int32')
    codeDistances = np.zeros((M, queriesCount, codebookSize),dtype='float32')
    for m in xrange(M):
        subqueries = queries[:,codebookDim*m:codebookDim*(m+1)].copy()
        codeDistances[m, :, :] = ynumpy.cross_distances(codebooks[m,:,:], subqueries)
    nearest = np.zeros((queriesCount, k), dtype='int32')
    qidRangeSize = 1
    rangesCount = int(math.ceil(float(queriesCount) / qidRangeSize))
    pool = Pool(threadsCount)
    ans = pool.map(partial(findNearestForRangePQ, \
                           rangeSize=qidRangeSize, codebookDistances=codeDistances, pointsCodes=codes, listLength=k), \
                           range(0, rangesCount))
    pool.close()
    pool.join()
    for i in xrange(len(ans)):
        if ans[i] == None:
            pass
        else:
            qidsCount = ans[i].shape[0]
            nearest[i*qidRangeSize:i*qidRangeSize+qidsCount,:] = ans[i]
    return nearest

def getQuantizationErrorPQ(testFilename, dim, pointsCount, codebooksFilename):
    codebooks = pickle.load(open(codebooksFilename, 'rb'))
    points = readXvecs(testFilename, dim, pointsCount)
    M = codebooks.shape[0]
    if dim % M != 0:
        raise Exception('Dim is not a multiple of M!')
    else:
        codebooksDim = dim / M
    errors = np.zeros(pointsCount)
    for m in xrange(M):
        codebook = codebooks[m,:,:]
        subpoints = points[:,m*codebooksDim:(m+1)*codebooksDim].copy()
        (idx, dis) = ynumpy.knn(subpoints, codebook, nt=3)
        errors += np.reshape(dis, pointsCount)
    return np.mean(errors)