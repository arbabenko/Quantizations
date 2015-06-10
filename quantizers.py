from pqCoding import *
from opqCoding import *
from aqCoding import *

dataFolder = './' # set folder for your quantization models structure

class Quantizer():
    def __init__(self):
        self.prefix = 'base_'
    def trainCodebooks(self, data, params):
        pass
    def getQuantizationError(self, data, params):
        pass
    def encodeDataset(self, data, params):
        pass
    def searchNearestNeighbors(self, data, params, k=10000):
        pass
    def getCodebooksFilename(self, data, params):
        return dataFolder + data.prefix + self.prefix + params.prefix + 'codebooks.dat'
    def getCodesFilename(self, data, params):
        return dataFolder + data.prefix + self.prefix + params.prefix + 'code.dat'

class PqQuantizer(Quantizer):
    def __init__(self, threadsCount, itCount):
        self.prefix = 'pq_'
        self.threadsCount = threadsCount
        self.itCount = itCount
    def trainCodebooks(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        learnCodebooksPQ(data.learnFilename, \
                         data.dim, params.M, params.K, \
                         data.learnPointsCount, \
                         codebooksFilename, self.threadsCount, \
                         self.itCount)
    def getQuantizationError(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        return getQuantizationErrorPQ(data.testFilename, data.dim, data.testPointsCount, codebooksFilename)
    def encodeDataset(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        encodeDatasetPQ(data.baseFilename, data.basePointsCount, \
                       codebooksFilename, codeFilename, self.threadsCount)
    def searchNearestNeighbors(self, data, params, k):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        return searchNearestNeighborsPQ(codeFilename, codebooksFilename, \
                                        data.queriesFilename,
                                        data.queriesCount, k, self.threadsCount)

class OpqQuantizer(Quantizer):
    def __init__(self, threadsCount, itCount):
        self.prefix = 'opq_'
        self.threadsCount = threadsCount
        self.itCount = itCount
    def trainCodebooks(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        learnCodebooksOPQ(data.learnFilename, \
                          data.learnPointsCount, \
                          data.dim, params.M, params.K, \
                          codebooksFilename, self.itCount)
    def getQuantizationError(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        return getQuantizationErrorOPQ(codebooksFilename, \
                                      data.testFilename, \
                                      data.testPointsCount)
    def encodeDataset(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        encodeDatasetOPQ(data.baseFilename, \
                         data.basePointsCount, \
                         codebooksFilename, codeFilename, self.threadsCount)
    def searchNearestNeighbors(self, data, params, k=10000):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        return searchNearestNeighborsOPQ(codeFilename, codebooksFilename, \
                                         data.queriesFilename, \
                                         data.queriesCount, k, self.threadsCount)


class AqQuantizer(Quantizer):
    def __init__(self, threadsCount, itCount):
        self.prefix = 'aq_'
        self.threadsCount = threadsCount
        self.itCount = itCount
    def trainCodebooks(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        learnCodebooksAQ(data.learnFilename, data.dim, \
                         params.M, params.K, \
                         data.learnPointsCount, codebooksFilename, \
                         self.threadsCount, self.itCount)
    def getQuantizationError(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        return getQuantizationErrorAQ(data.testFilename, data.dim, \
                                      data.testPointsCount, codebooksFilename, self.threadsCount)
    def encodeDataset(self, data, params):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        encodeDatasetAQ(data.baseFilename, data.basePointsCount, \
                        codebooksFilename, codeFilename, self.threadsCount)
    def searchNearestNeighbors(self, data, params, k=10000):
        codebooksFilename = self.getCodebooksFilename(data, params)
        codeFilename = self.getCodesFilename(data, params)
        return searchNearestNeighborsAQ(codeFilename, codebooksFilename, \
                                        data.queriesFilename, data.queriesCount, \
                                        k, self.threadsCount)
