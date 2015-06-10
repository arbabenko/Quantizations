from xvecReadWrite import *
import numpy as np
from yael import yael, ynumpy
import cPickle as pickle
from pqCoding import *

def reconstructPointsOPQ(codes, codebooks):
    M = codes.shape[1]
    codebookDim = codebooks.shape[2]
    dim = M * codebookDim
    pointsCount = codes.shape[0]
    points = np.zeros((pointsCount, dim), dtype='float32')
    for i in xrange(M):
        points[:,codebookDim*i:codebookDim*(i+1)] = codebooks[i,codes[:,i],:]
    return points

def learnCodebooksOPQ(pointsFilename, pointsCount, dim, M, K, vocFilename, ninit=20):
    points = readXvecs(pointsFilename, dim, pointsCount)
    R = np.identity(dim)
    rotatedPoints = np.dot(points, R.T).astype('float32')
    codebookDim = dim / M
    codebooks = np.zeros((M, K, codebookDim), dtype='float32')
    # init vocabs
    for i in xrange(M):
        perm = np.random.permutation(pointsCount)
        codebooks[i, :, :] = rotatedPoints[perm[:K], codebookDim*i:codebookDim*(i+1)].copy()
    # init assignments
    assigns = np.zeros((pointsCount, M), dtype='int32')
    for i in xrange(M):
        (idx, dis) = ynumpy.knn(rotatedPoints[:,codebookDim*i:codebookDim*(i+1)].astype('float32'), codebooks[i,:,:], nt=30)
        assigns[:,i] = idx.flatten()
    for it in xrange(ninit):
        approximations = reconstructPointsOPQ(assigns, codebooks)
        errors = rotatedPoints - approximations
        error = 0
        for pid in xrange(pointsCount):
            error += np.dot(errors[pid,:], errors[pid,:].T)
        print 'Quantization error: ' + str(error / pointsCount)
        U, s, V = np.linalg.svd(np.dot(approximations.T, points), full_matrices=False)
        R = np.dot(U, V)
        rotatedPoints = np.dot(points, R.T).astype('float32')
        for m in xrange(M):
            counts = np.bincount(assigns[:,m])
            for k in xrange(K):
                codebooks[m,k,:] = np.sum(rotatedPoints[assigns[:,m]==k,codebookDim*m:codebookDim*(m+1)], axis=0) / counts[k]
        for m in xrange(M):
            subpoints = rotatedPoints[:,codebookDim*m:codebookDim*(m+1)].copy()
            (idx, dis) = ynumpy.knn(subpoints, codebooks[m,:,:], nt=30)
            assigns[:,m] = idx.flatten()
    error = 0
    for m in xrange(M):
        subpoints = rotatedPoints[:,m*codebookDim:(m+1)*codebookDim].copy()
        (idx, dis) = ynumpy.knn(subpoints, codebooks[m,:,:], nt=2)
        error += np.sum(dis.flatten())
    print 'Quantization error: ' + str(error / pointsCount)
    model = (codebooks, R)
    vocFile = open(vocFilename, 'wb')
    pickle.dump(model, vocFile)

def getQuantizationErrorOPQ(codebooksFilename, pointsFilename, pointsCount):
    model = pickle.load(open(codebooksFilename, 'rb'))
    R = model[1]
    codebooks = model[0]
    codebookDim = codebooks.shape[2]
    M = codebooks.shape[0]
    dim = codebookDim * M
    points = readXvecs(pointsFilename, dim, pointsCount)
    rotatedPoints = np.dot(points, R.T).astype('float32')
    errors = 0.0
    for m in xrange(M):
        subpoints = rotatedPoints[:,m*dim/M:(m+1)*dim/M].copy()
        (idx, dis) = ynumpy.knn(subpoints, codebooks[m,:,:], nt=2)
        errors += np.sum(dis.flatten())
    print errors / pointsCount

def encodeDatasetOPQ(baseFilename, pointsCount, vocabFilename, codeFilename, threadsCount=30):
    model = pickle.load(open(vocabFilename, 'rb'))
    codebooks = model[0]
    R = model[1]
    M = codebooks.shape[0]
    dim = codebooks.shape[2] * M
    codes = np.zeros((pointsCount, M), dtype='int32')
    basePoints = readXvecs(baseFilename, dim, pointsCount)
    basePoints = np.dot(basePoints, R.T).astype('float32')
    error = 0
    for m in xrange(M):
        subpoints = basePoints[:,m*dim/M:(m+1)*dim/M].copy()
        (idx, dis) = ynumpy.knn(subpoints, codebooks[m,:,:], nt=threadsCount)
        codes[:,m] = idx.flatten()
        error += np.sum(dis.flatten())
    codeFile = open(codeFilename, 'wb')
    pickle.dump(codes, codeFile)
    codeFile.close()

def searchNearestNeighborsOPQ(codeFilename, codebooksFilename, queriesFilename, \
                                     queriesCount, k=10000, threadsCount=30):
    model = pickle.load(open(codebooksFilename, 'r'))
    codebooks = model[0]
    R = model[1]
    M = codebooks.shape[0]
    codebookDim = codebooks.shape[2]
    dim = codebookDim * M
    codebookSize = codebooks.shape[1]
    codes = pickle.load(open(codeFilename, 'r'))
    queries = readXvecs(queriesFilename, dim, queriesCount)
    queries = np.dot(queries, R.T).astype('float32')
    result = np.zeros((queriesCount, k), dtype='int32')
    codeDistances = np.zeros((M, queriesCount, codebookSize),dtype='float32')
    for m in xrange(M):
        subqueries = queries[:,m*codebookDim:(m+1)*codebookDim].copy()
        codeDistances[m,:,:] = ynumpy.cross_distances(codebooks[m], subqueries)
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
