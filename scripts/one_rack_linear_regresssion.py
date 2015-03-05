import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from foo import prep
from foo import evaluation

train = pd.read_pickle('data/rack_temps_train.pickle')
test = pd.read_pickle('data/rack_temps_test.pickle')

train = prep.transform_data(train, 'EO6')
test = prep.transform_data(test, 'EO6')

regr = linear_model.LinearRegression()
regr.fit(train[['l1', 'l2']].values, train['t'].values)

predicted = regr.predict(test[['l1', 'l2']].values)

print("MSE: " + str(mean_squared_error(test['t'], predicted)))

e = evaluation.eval_n_iterate(regr, test[['l1', 'l2']], test['t'], 12)

plt.figure(1)
plt.plot(predicted, 'r')
plt.plot(test['t'].values, 'b')

plt.figure(2)
plt.plot(e.T)

plt.show()

print("Coef: " + str(regr.coef_))
