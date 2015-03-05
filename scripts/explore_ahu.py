import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_pickle('data/ahu_train.pickle')
train2 = pd.read_pickle('data/room-level_train.pickle')

for i in range(1, 5):
    plt.figure(i)
    plt.clf()
    plt.subplot(2, 1, 1)
    train['ahu_' + str(i) + '_outlet'].plot()
    train['ahu_' + str(i) + '_inlet'].plot()
    train['ahu_' + str(i) + '_air_on'].plot()
    plt.legend()

    # plt.subplot(2, 2, 2)
    # train['ahu_' + str(i) + '_inlet'].plot()
    # train['ahu_' + str(i) + '_inlet_rh'].plot()
    # plt.plot(np.diff(np.array([train['ahu_' + str(i) + '_outlet'].values,
    #         train['ahu_' + str(i) + '_air_on'].values]), axis=0)[0])
    # train2['acu_supply_temperature_(c)'].plot()
    # train2['acu_return_temperature_(c)'].plot()
    # plt.legend()

    plt.subplot(2, 1, 2)
    train['ahu_' + str(i) + '_power'].plot()
    plt.legend()

    # plt.subplot(2, 2, 4)
    # train['ahu_' + str(i) + '_air_on'].plot()
    # plt.legend()

plt.show()
