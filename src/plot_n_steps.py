import numpy as np
import matplotlib.pyplot as plt

def main():

    NG = np.load("n_step_non_gauss_cv_mae_score.npy")
    G = np.load("n_step_gauss_cv_mae_score.npy")

    print(NG[0,:])
    print(NG[1,:])

    plt.figure()
    plt.plot(NG)
    # plt.figure()
    # plt.plot(G)

    plt.show()


if __name__ == "__main__":
    main()
