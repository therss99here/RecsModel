from JobData import JobData
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator
import pickle

import random
import numpy as np

def LoadJobData():
    jb = JobData()
    print("Loading Job ratings...")
    data = jb.LoadJobData()
    print("\nComputing Job popularity ranks  we can measure novelty later...")
    rankings = jb.getPopularityRanks()
    return (jb, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommending algorithms
(jb, evaluationData, rankings) = LoadJobData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# SVD
SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")


# SVD++
SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

err = evaluator.SampleTopNRecs(jb, testSubject=1)

