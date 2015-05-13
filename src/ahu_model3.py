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

    room_cooling = df['room_cooling_power_(kw)']

    plt.scatter(X_air_on[:, 0] - X_outlet[:, 0], X_power[:, 0])

    plt.show()

if __name__ == "__main__":
    main()

