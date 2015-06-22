import os
import sys
import utils
import layouts
import numpy as np
import pandas as pd

K = list(layouts.datacenter_layout.keys())

df = utils.prep_dataframe(keep=K)
df_shifted = utils.create_shifted_features(df)
df = df.join(df_shifted, how="outer")
df = df.dropna()

kf_scores = np.load("../results/gmrfNone_kf_scores.npy")
r2 = np.load("../results/gmrfNone_r2.npy")
kf_scores_hybrid = np.load("../results/hybridNone_kf_scores.npy")
r2_hybrid = np.load("../results/hybridNone_r2.npy")

def tables(data, r2, name):
    print(name.upper())
    columns = ["\textbf{Variables}", "\textbf{MAD t = 1}", "\textbf{MAD t = 4}", "\textbf{MAD t = 8}", "\textbf{$R^2$ = 0}"]
    table = pd.DataFrame(columns=columns)

    j = 0
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    for i, n in enumerate(df.columns.values):
        if 'l1_' not in n:
            v = ((n.replace("_", " ")).upper())[:5]
            mt1 = "{:.2f} +- {:.2f}".format(mean[0, i], std[0, i])
            mt4 = "{:.2f} +- {:.2f}".format(mean[3, i], std[3, i])
            mt8 = "{:.2f} +- {:.2f}".format(mean[7, i], std[7, i])
            nb_steps = np.where(r2[:, i] <= 0)[0][0]
            table.loc[j] =  [v, mt1, mt4, mt8, nb_steps]
            j += 1

    with open('../doc/tables/{}.tex'.format(name), 'w') as f:
        f.write(table.to_latex(longtable=True, escape=False, index=False))

if __name__ == "__main__":

    if len(sys.argv) == 1:
        pass
    else:
        basename = os.path.basename(sys.argv[1])
        name = os.path.splitext(basename)[0]

        if name == 'gmrf':
            tables(kf_scores, r2, name)
        else:
            tables(kf_scores_hybrid, r2_hybrid, name)
