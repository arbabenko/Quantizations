class DataParams:
    def __init__(self, dataName):
        if dataName == 'sift1M':
            self.prefix = 'sift1M_'
            self.learnFilename = '/sata/ResearchData/BigAnn/sift/sift_learn.fvecs'
            self.testFilename = '/sata/ResearchData/BigAnn/sift/sift_query.fvecs'
            self.baseFilename = '/sata/ResearchData/BigAnn/sift/sift_base.fvecs'
            self.groundtruthFilename = '/sata/ResearchData/BigAnn/sift/sift_groundtruth.ivecs'
            self.queriesFilename = '/sata/ResearchData/BigAnn/sift/sift_query.fvecs'
            self.dim = 128
            self.learnPointsCount = 100000
            self.testPointsCount = 1000
            self.basePointsCount = 1000000
            self.queriesCount = 10000
            self.groundSize = 100
        elif dataName == 'gist1M':
            self.prefix = 'gist1M_'
            self.learnFilename = '/sata/ResearchData/BigAnn/gist/gist_learn.fvecs'
            self.testFilename = '/sata/ResearchData/BigAnn/gist/gist_query.fvecs'
            self.baseFilename = '/sata/ResearchData/BigAnn/gist/gist_base.fvecs'
            self.groundtruthFilename = '/sata/ResearchData/BigAnn/gist/gist_groundtruth.ivecs'
            self.queriesFilename = '/sata/ResearchData/BigAnn/gist/gist_query.fvecs'
            self.dim = 960
            self.learnPointsCount = 200000
            self.testPointsCount = 1000
            self.basePointsCount = 1000000
            self.queriesCount = 1000
            self.groundSize = 100
        else:
            raise Exception("Unknown data!")

