import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def affichage(P, aretes=None):
    plt.axes(projection="3d")
    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]

    if aretes is None:
        aretes = []
        for i in range(len(P)):
            for j in range(i):
                aretes.append([i, j])

        aretes = np.array(aretes)

    for a in aretes:
        i = a[0]
        j = a[1]
        plt.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], '-', LineWidth=3)
    plt.show()


def affichage_transformations(P, aretes=None):
    #plt.axes(projection="3d")
    xg = []
    yg = []
    xd = []
    yd = []
    e = 7  # distance entre les 2 yeux (en cm)
    f = 40  # distance entre les yeux et le plan de projection (en cm)

    for i in range(len(P)):
        x = P[i][0]
        y = P[i][1]
        z = P[i][2]
        xg.append((f * x) / z)
        yg.append((f * y) / z)
        xd.append((f * (x - e)) / (z + e))
        yd.append((f * y) / z)

    xg = np.array(xg)
    yg = np.array(yg)
    xd = np.array(xd)
    yd = np.array(yd)

    if aretes is None:
        aretes = []
        for i in range(len(P)):
            for j in range(i):
                aretes.append([i, j])

        aretes = np.array(aretes)

    for a in aretes:
        i = a[0]
        j = a[1]
        plt.plot([xg[i], xg[j]], [yg[i], yg[j]], '-', LineWidth=3)
        plt.plot([xd[i], xd[j]], [yd[i], yd[j]], '-', LineWidth=3)
    plt.show()


if __name__ == '__main__':
    # Pour rendre intéractif le plot
    mpl.use('macosx')

    # tetraedre = [P1,P2,P3,P4]
    tetraedre = np.array([[1.2, -0.5, 65], [3, 1.5, 63], [4, -1, 62], [1.8, -1.5, 62]])
    affichage(tetraedre)
    affichage_transformations(tetraedre)

    # prisme = [P1,P2,P3,P4,P5,P6]
    prisme = np.array([[1, -1, 65], [1.8, -2, 62], [4.2, -1.5, 65], [1, 2, 65], [1.8, 1, 62], [4.2, 1.5, 65]])
    prisme_aretes = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
    # affichage(prisme, prisme_aretes)
    # affichage_transformations(prisme, prisme_aretes)
