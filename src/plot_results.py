import argparse
import utils
import layouts
import numpy as np
import matplotlib.pyplot as plt

def main(layout, file_mad, file_r2, file_conf):
    variables = getattr(layouts, layout)

    K = list(variables.keys())

    df = utils.prep_dataframe(keep=K)

    mad = np.load(file_mad)
    r2 = np.load(file_r2)
    conf_int = np.load(file_conf)

    for i in range(mad.shape[1]):
        fig, (ax, ax2) = plt.subplots(2, 1)
        ax.plot(mad[:, i], 'r-', label="Mean Absolute Deviation")
        ax.plot(mad[:, i] + abs(conf_int[i]), 'b--', label="Confidence Int.")
        ax.plot(mad[:, i] - abs(conf_int[i]), 'b--')
        ax.set_title("MAD: {}".format(df.columns.values[i]))
        ax2.plot(r2[:, i], 'g-')
        ax2.set_title("R2: {}".format(df.columns.values[i]))
        plt.legend(loc=0)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot MAD and R2.')
    parser.add_argument('layout', metavar='LAYOUT',
                        help="Layout.")
    parser.add_argument('file_mad', metavar='FILE_MAD',
                        help="File containing the MAD.")
    parser.add_argument('file_r2', metavar='FILE_R2',
                        help="File containing R2.")
    parser.add_argument('file_conf', metavar='FILE_CONF',
                        help="File containing confidence interval.")
    args = parser.parse_args()
    main(**vars(args))
