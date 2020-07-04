from JobData import JobData
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
import pickle
import random
import numpy as np

def LoadJobData():
    jb = JobData()
    print("Loading job ratings...")
    data = jb.LoadJobData()
    print("\nComputing job popularity ranks so we can measure novelty later...")
    rankings = jb.getPopularityRanks()
    return (jb, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(jb, evaluationData, rankings) = LoadJobData()

print("Searching for best parameters...")
param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
              'n_factors': [50, 100]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

grid_result = gs.fit(evaluationData)
print(grid_result)
#Using pickle to dump the model





# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Construct an Evaluator to evaluate data and rankings
evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Endgame!
evaluator.Evaluate(False)

rdata = evaluator.SampleTopNRecs(jb,testSubject=17)
