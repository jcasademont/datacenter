import argparse
import utils
import layouts
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfinv

# def main(layout, file_mad, file_r2, file_var):
def main(layout, score, r2, names):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    mad = []
    r2_data = []

    for i, s in enumerate(score):
        mad.append(np.load(s))

    for i, r in enumerate(r2):
        r2_data.append(np.load(r))

    mad = np.array(mad)
    r2 = np.array(r2_data)

    # var = np.load(file_var)

    # conf_int = np.sqrt(2) * erfinv(0.80) * np.sqrt(var)

    for i in range(mad.shape[2]):
        fig, (ax, ax2) = plt.subplots(2, 1)

        for j in range(mad.shape[0]):
            ax.plot(mad[j, :, i], label=names[j])

        # ax.plot(mad[:, i] + abs(conf_int[i]), 'b--', label="Confidence Int.")
        # ax.plot(mad[:, i] - abs(conf_int[i]), 'b--')
        ax.set_title("MAD: {}".format(df.columns.values[i]))

        for j in range(r2.shape[0]):
            ax2.plot(r2[j, :, i], label=names[j])

        ax2.set_ylim((-1, 1.5))
        ax2.set_title("R2: {}".format(df.columns.values[i]))
        ax.legend(loc=0)
        ax2.legend(loc=0)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot MAD and R2.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout.")
    parser.add_argument('-s', '--score', nargs='+',
                        help="File(s) containing the MAD.")
    parser.add_argument('-r', '--r2', nargs='+',
                        help="File(s) containing R2.")
    parser.add_argument('-n', '--names', nargs='+',
                        help="File(s) containing R2.")
    # parser.add_argument('file_var', metavar='FILE_VAR',
    #                     help="File containing confidence interval.")
    args = parser.parse_args()
    main(**vars(args))
