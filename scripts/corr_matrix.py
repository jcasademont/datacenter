import matplotlib.pyplot as plt
import matplotlib.colors as co
import numpy as np
import scipy.stats as stats
import pandas as pd

data = pd.read_pickle('data/rack_temps.pickle')
ahu = pd.read_pickle('data/ahu.pickle')
rl = pd.read_pickle('data/room-level.pickle')

#data['ahu_1_out'] = ahu['ahu_1_outlet']
#data['ahu_2_out'] = ahu['ahu_2_outlet']
#data['ahu_3_out'] = ahu['ahu_3_outlet']
#data['ahu_4_out'] = ahu['ahu_4_outlet']
#
#data['ahu_1_power'] = ahu['ahu_1_power']
#data['ahu_2_power'] = ahu['ahu_2_power']
#data['ahu_3_power'] = ahu['ahu_3_power']
#data['ahu_4_power'] = ahu['ahu_4_power']

# data['acu_temp_suply'] = rl['acu_supply_temperature_(c)']
# data['acu_temp_return'] = rl['acu_return_temperature_(c)']
# data['acu_humidity'] = rl['acu_return_humidity_(%)']
# data['room_temp'] = rl['room_temperature_(c)']

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                                       (0.5, 0.0, 0.1),
                                       (1.0, 1.0, 1.0)),

                   'green': ((0.0, 0.0, 0.0),
                                                (1.0, 0.0, 0.0)),

                   'blue':  ((0.0, 0.0, 1.0),
                                                (0.5, 0.1, 0.0),
                                                (1.0, 0.0, 0.0))
                  }

blue_red1 = co.LinearSegmentedColormap('BlueRed1', cdict1)

data = data.dropna()

cols = list(data.columns.values)

corr_mat = np.zeros((len(cols), len(cols)))

for i in range(len(cols)):
    for j in range(len(cols)):
        corr_mat[i][j] = stats.pearsonr(data[cols[i]], data[cols[j]])[0]

fig, ax = plt.subplots()
#plt.pcolor(corr_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0)
plt.pcolor(corr_mat, cmap=blue_red1, vmin=-1.0, vmax=1.0)
plt.colorbar()

ax.set_xticks(np.arange(corr_mat.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(corr_mat.shape[1])+0.5, minor=False)

# ax.invert_yaxis()
# ax.xaxis.tick_top()

ax.set_xticklabels(cols, rotation=-90, minor=False)
ax.set_yticklabels(cols, minor=False)
ax.axis('tight')
plt.show()
