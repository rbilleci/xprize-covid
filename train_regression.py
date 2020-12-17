# from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# first neural network with keras tutorial
from sklearn import tree
from sklearn.metrics import r2_score

import covid_constants
from pipeline import df_pipeline

train, validation, test = df_pipeline.process_for_training(covid_constants.path_data_baseline, 24, 10)
train = train.sample(frac=1).reset_index(drop=True)
train_x = train.iloc[:, 1:]
train_y = train.iloc[:, :1]
validation = validation.sample(frac=1).reset_index(drop=True)
validation_x = validation.iloc[:, 1:]
validation_y = validation.iloc[:, :1]
test = test.sample(frac=1).reset_index(drop=True)
test_x = test.iloc[:, 1:]
test_y = test.iloc[:, :1]

columns = train_x.shape[1]

# TREE BASED
regr = tree.DecisionTreeRegressor()  # .70
# regr = tree.ExtraTreeRegressor() # .697

# regr = svm.SVR(kernel='sigmoid', gamma='auto', epsilon=0.001) # -0.129
# regr = svm.SVR(kernel='sigmoid', gamma='auto', epsilon=0.001) # -3.227781243643915


# LINEAR
# regr = linear_model.Ridge() # 0.38
# regr = linear_model.ARDRegression() # .16
# regr = linear_model.LinearRegression() # -9.206457621136167e+21
# regr = linear_model.BayesianRidge() # 0.37
# regr = linear_model.ElasticNet() # -0.027548553698355738
# regr = linear_model.RidgeCV() # 0.39
# regr = linear_model.RidgeCV() # 0.39
# regr = linear_model.ElasticNetCV()
# regr = linear_model.GammaRegressor() #.38
# regr = linear_model.PoissonRegressor() # values out of range
# regr = linear_model.TweedieRegressor() # 0.0139
# regr = linear_model.HuberRegressor(max_iter=1000) # did not complete
# regr = linear_model.Lars() # weird
# regr = linear_model.LarsCV() # weird, 0.32
# regr = linear_model.LassoLars() # - 0.027
# regr = linear_model.LassoCV() # ... # did not converge
# regr = linear_model.LassoLarsCV() #weird, .33
# regr = linear_model.LassoLarsIC() # .33
# regr = linear_model.OrthogonalMatchingPursuit() # .37
# regr = linear_model.OrthogonalMatchingPursuitCV() # .37
# regr = linear_model.PassiveAggressiveRegressor() # -0.06
# regr = linear_model.RANSACRegressor() # -3.2848475088270746e+17
# regr = linear_model.SGDRegressor() #.305
# regr = linear_model.TheilSenRegressor() # -22

print("fitting")
for i in range(0, 10):
    regr.fit(train_x, train_y.values.ravel())
    score = r2_score(validation_y, regr.predict(validation_x))
    print(score)
