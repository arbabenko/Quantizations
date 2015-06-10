from quantizers import *
from methodParams import *
from dataParams import *
from getRecall import *

# parameters
paramSets = []
paramSets.append(MethodParams(M=4, K=256))

# datasets
datasets = []
#datasets.append(DataParams('sift1M'))
datasets.append(DataParams('gist1M'))

# methods
quantizers = []
quantizers.append(PqQuantizer(threadsCount=30, itCount=30))
quantizers.append(OpqQuantizer(threadsCount=30, itCount=30))
quantizers.append(AqQuantizer(threadsCount=30, itCount=20))

trainCodebooks = True
encodeDatasets = True

if trainCodebooks:
    for params in paramSets:
        for data in datasets:
            for method in quantizers:
                method.trainCodebooks(data, params)
                print 'Codebooks for settings ' + data.prefix + method.prefix + params.prefix  + ' are learned'

if encodeDatasets:
    for params in paramSets:
        for data in datasets:
            for method in quantizers:
                method.encodeDataset(data, params)
                print 'Dataset for settings ' + data.prefix + method.prefix + params.prefix  + ' is encoded'

for params in paramSets:
    for data in datasets:
        for method in quantizers:
            print 'Settings: ' + data.prefix + method.prefix + params.prefix
            print 'Quantization error: ' + str(method.getQuantizationError(data, params))


for params in paramSets:
    for data in datasets:
        for method in quantizers:
            neighborsCount = min(1024, data.basePointsCount)
            result = method.searchNearestNeighbors(data, params, k=neighborsCount)
            groundtruth = readXvecs(data.groundtruthFilename, \
                                    data.groundSize, data.queriesCount)
            print 'Results: ' + data.prefix + method.prefix + params.prefix
            for i in xrange(10):
                length = 2 ** i
                print 'Recall@' + str(length) + ' ' + str(getRecallAt(length, result, groundtruth))
