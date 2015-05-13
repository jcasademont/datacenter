import utils
import layouts
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold

def main():
    K = list(layouts.ahu_layout.keys())
    df = utils.prep_dataframe(keep=K)

    df_air_on = df[['ahu_1_air_on',
                    'ahu_2_air_on',
                    'ahu_3_air_on',
                    'ahu_4_air_on']]
    X_air_on = df_air_on.values

    df_outlet = df[['ahu_1_outlet',
                    'ahu_2_outlet',
                    'ahu_3_outlet',
                    'ahu_4_outlet']]
    X_outlet = df_outlet.values

    df_inlet = df[['ahu_1_inlet',
                   'ahu_2_inlet',
                   'ahu_3_inlet',
                   'ahu_4_inlet']]
    X_inlet = df_inlet.values

    df_inlet_rh = df[['ahu_1_inlet_rh',
                      'ahu_2_inlet_rh',
                      'ahu_3_inlet_rh',
                      'ahu_4_inlet_rh']]
    X_inlet_rh = df_inlet_rh.values

    df_power = df[['ahu_1_power',
                   'ahu_2_power',
                   'ahu_3_power',
                   'ahu_4_power']]
    X_power = df_power.values

    cooling_shifted = df['room_cooling_power_(kw)'].shift(1)
    cooling_shifted = cooling_shifted.dropna()

    linreg = LinearRegression(normalize=True)

    coefs = np.empty(5)
    intercepts = np.empty(5)
    p = 0

    powers = []
    mean_power = np.empty(5)

    for i in range(4):
        if i != 2:
            X = X_air_on[:, i].reshape(X_air_on.shape[0], 1)
            Y = X_outlet[:, i].reshape(X_outlet.shape[0], 1)
            linreg.fit(X, Y)
            print("{} Linear regression: intercept = {}, coef = {}"
                    .format(i, linreg.intercept_, linreg.coef_))
            coefs[p] = linreg.coef_[0][0]
            intercepts[p] = linreg.intercept_[0]
            powers.append(X_power[:, i])
            mean_power[p] = np.mean(X_power[:, i])
            p += 1
        else:
            indices = np.array([np.where(X_outlet[:, i] < 21.5)[0],
                                np.where(X_outlet[:, i] > 23.3)[0]])

            for j in range(indices.shape[0]):
                X = X_air_on[:, i][indices[j]].reshape(np.size(indices[j]), 1)
                Y = X_outlet[:, i][indices[j]].reshape(np.size(indices[j]), 1)
                linreg.fit(X, Y)
                print("{} Linear regression: intercept = {}, coef = {}"
                        .format(i, linreg.intercept_, linreg.coef_))
                coefs[p] = linreg.coef_[0][0]
                intercepts[p] = linreg.intercept_[0]
                powers.append(X_power[:, i][indices[j]])
                mean_power[p] = np.mean(X_power[:, i][indices[j]])
                p += 1

    p = 0
    c = ['b', 'g', 'r', 'm']
    scatters = []

    plt.figure()
    s0 = plt.scatter(X_air_on[:, 0], X_outlet[:, 0], c=c[0])
    s1 = plt.scatter(X_air_on[:, 1], X_outlet[:, 1], c=c[1])
    s2 = plt.scatter(X_air_on[:, 2], X_outlet[:, 2], c=c[2])
    s3 = plt.scatter(X_air_on[:, 3], X_outlet[:, 3], c=c[3])
    plt.xlabel('Air on')
    plt.ylabel('Outlet')

    plt.legend((s0, s1, s2, s3), ("Ahu 1", "Ahu 2", "Ahu 3", "Ahu 4"))
        # if i != 2:
        #     x = np.sort(X_air_on[:, i])
        #     x = np.arange(10, 30)
        #     y = coefs[p] * x + intercepts[p]
        #     plt.plot(x, y, 'r-')
        #     p += 1
        # else:
        #     indices = np.array([np.where(X_outlet[:, i] < 21)[0],
        #                         np.where(X_outlet[:, i] > 23.5)[0]])

        #     for j in range(indices.shape[0]):
        #         x = np.sort(X_air_on[:, i][indices[j]])
        #         x = np.arange(10, 30)
        #         y = coefs[p] * x + intercepts[p]
        #         plt.plot(x, y, 'g-')
        #         p += 1

    plt.figure()
    for i in range(5):
        X = powers[i]
        plt.scatter(np.ones(X.shape[0]) * i, X, alpha=0.01)

    plt.figure()
    plt.scatter(intercepts, coefs, s=mean_power * 100 + 10, c=['r', 'b', 'g', 'c', 'm'])

    # for i in range(4):
    #     plt.figure()
    #     plt.scatter(df['acu_supply_temperature_(c)'] - X_air_on[:, i], X_power[:, i])

    # plt.figure()
    # plt.scatter(np.sum(X_air_on, axis=1), np.sum(X_outlet, axis=1), s=df['room_cooling_power_(kw)'])

    plt.figure()
    plt.scatter((np.sum(X_air_on, axis=1) - np.sum(X_outlet, axis=1))[:-1], cooling_shifted)
    plt.xlabel("Air on - Outlet")
    plt.ylabel("Cooling power")

    # X = np.hstack((X_air_on[:, :3], X_outlet[:, :3]))
    #X = np.array([X_air_on[:, 3], X_outlet[:, 3]]).T
    X = X_air_on - X_outlet
    print(X.shape)
    Y = df['room_cooling_power_(kw)'].reshape(X.shape[0], 1)
    Y = X_power

    for degree in [0, 1, 2, 3]:
        kf = KFold(n=X.shape[0], n_folds=4)
        scores = []
        for train, test in kf:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(X[train], Y[train])
            s = model.score(X[test], Y[test])
            scores.append(s)
            print("Degree {}: score = {}".format(degree, s))
        print("Mean score = {}".format(np.mean(scores)))

    linreg.fit(X, Y)
    print("Coef = {}, intercept = {}".format(linreg.coef_, linreg.intercept_))

    plt.show()

if __name__ == "__main__":
    main()
