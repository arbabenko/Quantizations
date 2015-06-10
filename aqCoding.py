import cPickle as pickle
from scipy import sparse
from xvecReadWrite import *
from scipy.sparse.linalg import lsmr
from multiprocessing import Pool
from functools import partial


def solveDimensionLeastSquares(startDim, dimCount, data, indices, indptr, trainPoints, codebookSize, M):
    A = sparse.csr_matrix((data, indices, indptr), shape=(trainPoints.shape[0], M*codebookSize), copy=False)
    discrepancy = 0
    dimCount = min(dimCount, trainPoints.shape[1] - startDim)
    codebooksComponents = np.zeros((M, codebookSize, dimCount), dtype='float32')
    for dim in xrange(startDim, startDim+dimCount):
        b = trainPoints[:, dim].flatten()
        solution = lsmr(A, b, show=False, maxiter=250)
        codebooksComponents[:, :, dim-startDim] = np.reshape(solution[0], (M, codebookSize))
        discrepancy += solution[3] ** 2
    return (codebooksComponents, discrepancy)


def getMeanQuantizationError(points, assigns, codebooks):
    errors = getQuantizationErrors(points, assigns, codebooks)
    return np.mean(errors)


def getQuantizationErrors(points, assigns, codebooks):
    pointsCopy = points.copy()
    for m in xrange(codebooks.shape[0]):
        pointsCopy = pointsCopy - codebooks[m,assigns[:,m],:]
    errors = np.zeros((points.shape[0]), dtype='float32')
    for pid in xrange(points.shape[0]):
        errors[pid] = np.dot(pointsCopy[pid,:], pointsCopy[pid,:].T)
    return errors


def encodePointsBeamSearch(startPid, pointsCount, pointCodebookProducts, codebooksProducts, codebooksNorms, branch):
    M = codebooksProducts.shape[0]
    K = codebooksProducts.shape[1]
    hashArray = np.array([13 ** i for i in xrange(M)])
    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)
    assigns = np.zeros((pointsCount, M), dtype='int32')
    errors = np.zeros((pointsCount), dtype='float32')
    for pid in xrange(startPid, startPid+pointsCount):
        distances = - pointCodebookProducts[pid,:] + codebooksNorms
        bestIdx = distances.argsort()[0:branch]
        vocIds = bestIdx / K
        wordIds = bestIdx % K
        bestSums = -1 * np.ones((branch, M), dtype='int32')
        for candidateIdx in xrange(branch):
            bestSums[candidateIdx,vocIds[candidateIdx]] = wordIds[candidateIdx]
        bestSumScores = distances[bestIdx]
        for m in xrange(1, M):
            candidatesScores = np.array([bestSumScores[i].repeat(M * K) for i in xrange(branch)]).flatten()
            candidatesScores += np.tile(distances, branch)
            globalHashTable = np.zeros(115249, dtype='int8')
            for candidateIdx in xrange(branch):
                for m in xrange(M):
                      if bestSums[candidateIdx,m] < 0:
                          continue
                      candidatesScores[candidateIdx*M*K:(candidateIdx+1)*M*K] += \
                          codebooksProducts[m, bestSums[candidateIdx,m], :]
                      candidatesScores[candidateIdx*M*K + m*K:candidateIdx*M*K+(m+1)*K] += 999999
            bestIndices = candidatesScores.argsort()
            found = 0
            currentBestIndex = 0
            newBestSums = -1 * np.ones((branch, M), dtype='int32')
            newBestSumsScores = -1 * np.ones((branch), dtype='float32')
            while found < branch:
                bestIndex = bestIndices[currentBestIndex]
                candidateId = bestIndex / (M * K)
                codebookId = (bestIndex % (M * K)) / K
                wordId = (bestIndex % (M * K)) % K
                bestSums[candidateId,codebookId] = wordId
                hashIdx = np.dot(bestSums[candidateId,:], hashArray) % 115249
                if globalHashTable[hashIdx] == 1:
                    bestSums[candidateId,codebookId] = -1
                    currentBestIndex += 1
                    continue
                else:
                    bestSums[candidateId,codebookId] = -1
                    globalHashTable[hashIdx] = 1
                    newBestSums[found,:] = bestSums[candidateId,:]
                    newBestSums[found,codebookId] = wordId
                    newBestSumsScores[found] = candidatesScores[bestIndex]
                    found += 1
                    currentBestIndex += 1
            bestSums = newBestSums.copy()
            bestSumScores = newBestSumsScores.copy()
        assigns[pid-startPid,:] = bestSums[0,:]
        errors[pid-startPid] = bestSumScores[0]
    return (assigns, errors)


def encodePointsAQ(points, codebooks, branch):
    pointsCount = points.shape[0]
    M = codebooks.shape[0]
    K = codebooks.shape[1]
    codebooksProducts = np.zeros((M,K,M*K), dtype='float32')
    fullProducts = np.zeros((M,K,M,K), dtype='float32')
    codebooksNorms = np.zeros((M*K), dtype='float32')
    for m1 in xrange(M):
        for m2 in xrange(M):
            fullProducts[m1,:,m2,:] = 2 * np.dot(codebooks[m1,:,:], codebooks[m2,:,:].T)
        codebooksNorms[m1*K:(m1+1)*K] = fullProducts[m1,:,m1,:].diagonal() / 2
        codebooksProducts[m1,:,:] = np.reshape(fullProducts[m1,:,:,:], (K,M*K))
    assigns = np.zeros((pointsCount, M), dtype='int32')
    pidChunkSize = min(pointsCount, 5030)
    errors = np.zeros(pointsCount, dtype='float32')
    for startPid in xrange(0, pointsCount, pidChunkSize):
        realChunkSize = min(pidChunkSize, pointsCount - startPid)
        chunkPoints = points[startPid:startPid+realChunkSize,:]
        queryProducts = np.zeros((realChunkSize, M * K), dtype=np.float32)
        for pid in xrange(realChunkSize):
            errors[pid+startPid] += np.dot(chunkPoints[pid,:], chunkPoints[pid,:].T)
        for m in xrange(M):
            queryProducts[:,m*K:(m+1)*K] = 2 * np.dot(chunkPoints, codebooks[m,:,:].T)
        poolSize = 8
        chunkSize = realChunkSize / poolSize
        pool = Pool(processes=poolSize+1)
        ans = pool.map_async(partial(encodePointsBeamSearch, \
                               pointsCount=chunkSize, \
                               pointCodebookProducts=queryProducts, \
                               codebooksProducts=codebooksProducts, \
                               codebooksNorms=codebooksNorms, \
                               branch=branch), xrange(0, realChunkSize, chunkSize)).get()
        pool.close()
        pool.join()
        for startChunkPid in xrange(0, realChunkSize, chunkSize):
            pidsCount = min(chunkSize, realChunkSize - startChunkPid)
            assigns[startPid+startChunkPid:startPid+startChunkPid+pidsCount,:] = ans[startChunkPid/chunkSize][0]
            errors[startPid+startChunkPid:startPid+startChunkPid+pidsCount] += ans[startChunkPid/chunkSize][1]
    return (assigns, errors)


def learnCodebooksAQ(learnFilename, dim, M, K, pointsCount, codebooksFilename, branch, threadsCount=30, itsCount=10):
    if M < 1:
        raise Exception('M is not positive!')
    points = readXvecs(learnFilename, dim, pointsCount)
    assigns = np.zeros((pointsCount, M), dtype='int32')
    codebooks = np.zeros((M, K, dim), dtype='float32')
    # random initialization of assignment variables
    # (initializations from (O)PQ should be used for better results)
    for m in xrange(M):
        assigns[:,m] = np.random.randint(0, K, pointsCount)
    errors = getQuantizationErrors(points, assigns, codebooks)
    print "Error before learning iterations: %f" % (np.mean(errors))
    data = np.ones(M * pointsCount, dtype='float32')
    indices = np.zeros(M * pointsCount, dtype='int32')
    indptr = np.array(range(0, pointsCount + 1)) * M
    for it in xrange(itsCount):
        for i in xrange(pointsCount * M):
            indices[i] = 0
        for pid in xrange(pointsCount):
            for m in xrange(M):
                indices[pid * M + m] = m * K + assigns[pid,m]
        dimChunkSize = dim / threadsCount
        pool = Pool(threadsCount)
        ans = pool.map(partial(solveDimensionLeastSquares, \
                               dimCount=dimChunkSize, \
                               data=data, \
                               indices=indices, \
                               indptr=indptr, \
                               trainPoints=points, \
                               codebookSize=K, M=M), range(0, dim, dimChunkSize))
        pool.close()
        pool.join()
        for d in range(0, dim, dimChunkSize):
            dimCount = min(dimChunkSize, dim - d)
            codebooks[:, :, d:d+dimCount] = ans[d / dimChunkSize][0]
        errors = getQuantizationErrors(points, assigns, codebooks)
        print "Error after LSMR step: %f" % (np.mean(errors))
        (assigns, errors) = encodePointsAQ(points, codebooks, branch)
        errors = getQuantizationErrors(points, assigns, codebooks)
        print "Error after encoding step: %f" % (np.mean(errors))
    resultFile = open(codebooksFilename, 'wb')
    pickle.dump(codebooks, resultFile)
    resultFile.close()


def getQuantizationErrorAQ(testFilename, dim, pointsCount, codebooksFilename, branch):
    codebooks = pickle.load(open(codebooksFilename, 'rb'))
    points = readXvecs(testFilename, dim, pointsCount)
    (assigns, errors) = encodePointsAQ(points, codebooks, branch)
    return np.mean(errors)


def encodeDatasetAQ(baseFilename, pointsCount, codebooksFilename, codeFilename, branch):
    codebooks = pickle.load(open(codebooksFilename, 'rb'))
    dim = codebooks.shape[2]
    points = readXvecs(baseFilename, dim, pointsCount)
    (codes, errors) = encodePointsAQ(points, codebooks, branch)
    print "Mean AQ quantization error: %f" %(np.mean(errors))
    resultFile = open(codeFilename, 'wb')
    pickle.dump(codes, resultFile)
    resultFile.close()


def calculateDistancesAQ(qid, codebooksProducts, queryCodebookProducts, pointCodes, listLength):
    print 'Handling qid ' + str(qid)
    distances = np.zeros(pointCodes.shape[0], dtype='float32')
    for pid in xrange(pointCodes.shape[0]):
        for i in xrange(codebooksProducts.shape[0]):
            distances[pid] -= 2 * queryCodebookProducts[i, qid, pointCodes[pid, i]]
            distances[pid] += codebooksProducts[i, i, pointCodes[pid, i], pointCodes[pid, i]]
            for j in xrange(i+1, codebooksProducts.shape[1]):
               distances[pid] += 2 * codebooksProducts[i, j, pointCodes[pid, i], pointCodes[pid, j]]
    return distances.argsort()[0:listLength]


def searchNearestNeighborsAQ(codeFilename, codebookFilename, queriesFilename, queriesCount, k=10000, threadsCount=30):
    codebooks = pickle.load(open(codebookFilename, 'rb'))
    M = codebooks.shape[0]
    dim = codebooks.shape[2]
    codebookSize = codebooks.shape[1]
    codes = pickle.load(open(codeFilename, 'rb'))
    queries = readXvecs(queriesFilename, dim, queriesCount)
    codebooksProducts = np.zeros((M, M, codebookSize, codebookSize),dtype='float')
    for i in xrange(M):
        for j in xrange(M):
            codebooksProducts[i, j, :, :] = np.dot(codebooks[i,:,:], codebooks[j,:,:].T)
    queryCodebookProducts = np.zeros((M, queriesCount, codebookSize),dtype='float')
    for i in xrange(M):
        queryCodebookProducts[i,:,:] = np.dot(queries, codebooks[i,:,:].T)
    k = min(k, codes.shape[0])
    nearest = np.zeros((queries.shape[0], k), dtype='int32')
    pool = Pool(processes=threadsCount)
    ans = pool.map(partial(calculateDistancesAQ, \
                           codebooksProducts=codebooksProducts, \
                           queryCodebookProducts=queryCodebookProducts, \
                           pointCodes=codes, \
                           listLength=k), range(0,queries.shape[0]))
    pool.close()
    pool.join()
    for i in xrange(queries.shape[0]):
        nearest[i,:] = ans[i].flatten()
    return nearest