import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from random import randint
import math


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
    # plt.axes(projection="3d")
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


def calcul_aretes_polyedre_regulier(P, nbPointParFace):
    aretes = []

    # Premiere face
    for i in range(0, nbPointParFace):
        if i + 1 >= nbPointParFace:
            aretes.append([i, 0])
        else:
            aretes.append([i, i + 1])

    # Deuxieme face
    for j in range(nbPointParFace, len(P)):
        if j + 1 >= len(P):
            aretes.append([j, nbPointParFace])
        else:
            aretes.append([j, j + 1])

    # Liaison premiere et deuxieme face
    for k in range(0, nbPointParFace):
        aretes.append([k, k + nbPointParFace])

    aretes = np.array(aretes)

    return aretes


def ptsAleaSurface(nbPoints, f, e):
    transforme = []
    for i in range(0, nbPoints):
        x = randint(-6, 12)
        y = randint(-8, 8)
        z = calculZ(x, y)
        transforme.append([(f * x) / z, (f * y) / z])
        transforme.append([((f * (x - e)) / z) + e, (f * y) / z])
    plt.plot(np.array(transforme)[:, 0], np.array(transforme)[:, 1], '.', markersize=5, color='black')
    plt.show()


def pointFixe(a, b, c, d, nbY, nbPts, f, e):
    for k in range(nbY):
        xyP = []
        # Generation des pts
        for j in range(nbPts):
            print(j)
            xyP.append([randint(a, b), c + ((d - c) / nbY) * k])
            xyI = [calculTd(xyP[j][0], xyP[j][1], calculZ2(xyP[j][0], xyP[j][1]), f, e)]
            nbI = 1
            while xyI[nbI - 1][0] < b:
                delta = calculDelta(xyP[j], f, e)
                sAppro = calculSApprox(xyI[nbI - 1], f, e, delta)
                xp = calculXp(xyI[nbI - 1][0], sAppro, e)
                xyI.append(calculTd(xp, c + ((d - c) / nbY) * k, calculZ(xp, c + ((d - c) / nbY) * k), f, e))
                nbI = nbI + 1
                print(xyI)
            plt.plot(xyI, '.', markersize=20)
    plt.show()


def approxDirect(a, b, c, d, nbY, nbPts, f, e):
    plt.figure(dpi=300)
    for k in range(nbY):
        xyP = []
        for j in range(nbPts):
            xyP.append([randint(a, b), c + ((d - c) / nbY) * k])
            xyI = [calculTd(xyP[j][0], xyP[j][1], calculZ(xyP[j][0], xyP[j][1]), f, e)]
            nbI = 1
            while xyI[nbI - 1][0] < b:
                delta = calculDelta(xyP[j], f, e)
                sAppro = calculSApprox(xyI[nbI - 1], f, e, delta)
                xyI.append([xyI[nbI - 1][0] + sAppro, c + ((d - c) / nbY) * k])
                nbI = nbI + 1
            plt.plot(np.array(xyI)[:, 0], np.array(xyI)[:, 1], '.', markersize=1,color='black')
    plt.show()


def test(nbPoints, f, e):
    transforme = []
    for i in range(0, nbPoints):
        x = randint(0, 6)
        y = randint(-10, 10)
        z = calculZ2(x, y)
        transforme.append([(f * x) / z, (f * y) / z])
        transforme.append([((f * (x - e)) / z) + e, (f * y) / z])
    plt.plot(np.array(transforme)[:,0],np.array(transforme)[:,1], '.', markersize=5, color='black')
    plt.show()


def calculTd(x, y, z, f, e):
    return [((f * (x - e)) / z) + e, y]


def calculZ(x, y):
    if abs(y) > 8 or x > 12 or -6 > x:
        z = 80
    else:
        z = ((x * x) / 4) - ((3 * x) / 2) + 62
    return z


def calculZ2(x, y):
    if pow((x - 3), 2) + pow(y, 2) < 12:
        z = 58 + math.sqrt(16 - pow(x - 3, 2) - pow(y, 2))
    else:
        z = 60
    return z


def calculDelta(xyP, f, e):
    return (1 - (f / calculZ(xyP[0], xyP[1]))) * e


def calculSApprox(xyI, f, e, delta):
    return (1 - (f / calculZ(xyI[0] + (delta / 2), xyI[1]))) * e


def calculXp(xI, s, e):
    return xI / (1 - (s / e))


if __name__ == '__main__':
    # Pour rendre interactif le plot
    # mpl.use('macosx')

    # Tetraedre = [P1,P2,P3,P4]
    # tetraedre = np.array([[1.2, -0.5, 65], [3, 1.5, 63], [4, -1, 62], [1.8, -1.5, 62]])
    # affichage(tetraedre)
    # affichage_transformations(tetraedre)

    # # Prisme = [P1,P2,P3,P4,P5,P6]
    # prisme = np.array([[1, -1, 65], [1.8, -2, 62], [4.2, -1.5, 65], [1, 2, 65], [1.8, 1, 62], [4.2, 1.5, 65]])
    # prisme_aretes = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
    # affichage(prisme, prisme_aretes)
    # affichage_transformations(prisme, prisme_aretes)

    # # Polyedre
    # z = 64
    # polyedre = np.array([[0, 2, z], [0, 3, z], [3, 3, z], [3, 2, z], [2, 2, z], [2, 0, z], [1, 0, z], [1, 2, z],
    #                      [0, 2, z + 1], [0, 3, z + 1], [3, 3, z + 1], [3, 2, z + 1], [2, 2, z + 1], [2, 0, z + 1],
    #                      [1, 0, z + 1], [1, 2, z + 1]])
    # affichage(polyedre, calcul_aretes_polyedre_regulier(polyedre, 8))
    # affichage_transformations(polyedre, calcul_aretes_polyedre_regulier(polyedre, 8))
    test(800, 40, 7)
    approxDirect(-20, 25, -11, 20, 150, 50, 12, 7)
