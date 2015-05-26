import utils
import layouts
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def main():
    K = list(layouts.ahu_layout.keys())
    df = utils.prep_dataframe(keep=K)

    # df[['ahu_1_inlet', 'ahu_1_outlet', 'ahu_1_power']].plot()
    # plt.figure()
    # plt.scatter(df['ahu_1_inlet'] - df['ahu_1_outlet'], df['ahu_1_power'])
    # plt.figure()
    # plt.scatter(df['ahu_2_inlet'] - df['ahu_2_outlet'], df['ahu_2_power'])
    # plt.figure()
    # plt.scatter(df['ahu_3_inlet'] - df['ahu_3_outlet'], df['ahu_3_power'])
    # plt.figure()
    # plt.scatter(df['ahu_4_inlet'] - df['ahu_4_outlet'], df['ahu_4_power'])

    df_air_on = df[['ahu_1_air_on', 'ahu_2_air_on', 'ahu_3_air_on', 'ahu_4_air_on']]
    X_air_on = df_air_on.values

    df_outlet = df[['ahu_1_outlet', 'ahu_2_outlet', 'ahu_3_outlet', 'ahu_4_outlet']]
    X_outlet = df_outlet.values

    df_inlet = df[['ahu_1_inlet', 'ahu_2_inlet', 'ahu_3_inlet', 'ahu_4_inlet']]
    X_inlet = df_inlet.values

    df_inlet_rh = df[['ahu_1_inlet_rh', 'ahu_2_inlet_rh', 'ahu_3_inlet_rh', 'ahu_4_inlet_rh']]
    X_inlet_rh = df_inlet_rh.values

    df_power = df[['ahu_1_power', 'ahu_2_power', 'ahu_3_power', 'ahu_4_power']]
    X_power = df_power.values

    # plt.scatter(X_air_on.ravel() - X_outlet.ravel(), X_power.ravel())
    # plt.scatter(X_air_on.ravel() - X_outlet.ravel(), X_power.ravel())
    # for i in range(4):
        # plt.figure()
        # plt.plot(X_air_on[:,i])
        # plt.plot(X_outlet[:,i])
        # plt.plot(X_power[:,i])
        # plt.figure()
        # plt.scatter(X_power[:,i], (X_inlet[:,i] + X_air_on[:,i]) / 2 - X_outlet[:,i])
        # plt.ylabel('Mean air on / inlet - outlet')
        # plt.xlabel('Power')
        # plt.figure()
        # plt.scatter(X_power[:,i], X_air_on[:,i] - X_outlet[:,i])
        # plt.ylabel('Air on - outlet')
        # plt.xlabel('Power')
        # plt.title('AHU {}'.format(i + 1))
        # plt.figure()
        # plt.scatter(X_power[:,i], X_inlet[:,i] - X_outlet[:,i])
        # plt.ylabel('Inlet - outlet')
        # plt.xlabel('Power')
        # plt.figure()
        # plt.scatter(X_air_on[:, i], X_outlet[:, i])
        # plt.ylabel('Outlet')
        # plt.xlabel('Air on')
        # plt.figure()
        # plt.scatter(X_inlet[:, i], X_outlet[:, i])
        # plt.ylabel('Outlet')
        # plt.xlabel('Inlet')
        # plt.figure()
        # plt.scatter(X_outlet[:,i], (X_inlet[:,i] + X_air_on[:,i]) / 2 - X_outlet[:,i])
        # plt.ylabel('Mean air on / inlet - outlet')
        # plt.xlabel('Outlet')

    # plt.figure()
    # plt.scatter(np.sum(df_air_on, axis=1) - np.sum(df_outlet, axis=1),
    #             df['room_cooling_power_(kw)'])
    # plt.figure()
    # plt.scatter((np.sum(df_outlet, axis=1)),
    #             df['room_cooling_power_(kw)'])
    plt.figure()
    plt.scatter(X_power[:,2], X_air_on[:,2] - X_outlet[:,2])
    plt.ylabel('Air on - outlet')
    plt.xlabel('Power')
    plt.title('AHU 3')

    low = np.where(X_air_on[:,2] - X_outlet[:,2] < 4)[0]
    high = np.where(X_air_on[:,2] - X_outlet[:,2] >= 4)[0]

    linreg = LinearRegression()

    linreg.fit(X_power[low, 2].reshape(len(low), 1), (X_air_on[low, 2] - X_outlet[low, 2]).reshape(len(low), 1))
    plt.plot(X_power[low, 2], linreg.predict((X_power[low, 2]).reshape(len(low), 1)))
    print("{} Linear regression: intercept = {}, coef = {}".format(0, linreg.intercept_, linreg.coef_))
    linreg.fit(X_power[high, 2].reshape(len(high), 1), (X_air_on[high, 2] - X_outlet[high, 2]).reshape(len(high), 1))
    plt.plot(X_power[high, 2], linreg.predict((X_power[high, 2]).reshape(len(high), 1)))
    print("{} Linear regression: intercept = {}, coef = {}".format(0, linreg.intercept_, linreg.coef_))
    # for i in range(4):
        # linreg.fit(X_air_on[:, i], X_outlet[:, i])
        # print("{} Linear regression: intercept = {}, coef = {}".format(i, linreg.intercept_, linreg.coef_))

    plt.figure()
    powersLines = plt.plot(X_power)
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend(powersLines, ("AHU 1", "AHU 2", "AHU 3", "AHU 4"))

    plt.figure()
    powersLines = plt.plot(X_outlet)
    plt.xlabel("Time")
    plt.ylabel("Outlet")

    plt.legend(powersLines, ("AHU 1", "AHU 2", "AHU 3", "AHU 4"))

    plt.figure()
    plt.scatter(X_air_on[:, 0] - X_outlet[:, 0], X_air_on[:, 1] - X_outlet[:, 1])
    plt.scatter(X_air_on[:, 3] - X_outlet[:, 3], X_air_on[:, 1] - X_outlet[:, 1])


    plt.show()

if __name__ == "__main__":
    main()
