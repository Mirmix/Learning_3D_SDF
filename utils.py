import numpy as np
from skimage import measure
import matplotlib.pyplot as plt


def calculateSurf(SDF, iso_val=0.0, space=0.01):
    return measure.marching_cubes(SDF, iso_val, spacing=(space, space, space))


def visualize(SDF, iso_val=0.0, space=0.01):
    v, f, n, vals = calculateSurf(SDF, iso_val, space)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], lw=1)
    plt.show()
    return v, f, n, vals


def writeOBJ(objFilename, v, t, x_displacement = 0):
    objFile = open(objFilename, "w");
    # write obj vertices
    for i in range(len(v)):
        objFile.write("v %f %f %f\n" % (v[i][0], v[i][1], v[i][2]+x_displacement))
    # write obj triangles
    for i in range(len(t)):
        objFile.write("f %d %d %d\n" % (t[i][0] + 1, t[i][1] + 1, t[i][2] + 1))
    print("\t Exporting %s (%d verts , %d tris)" % (objFilename, len(v), len(t)))
    objFile.close()


def fillSDF(y_pred, D, H, W):
    SDF = np.zeros((D, H, W))
    count = 0
    for k in range(D):
        for j in range(H):
            for i in range(W):
                SDF[int(k)][int(j)][int(i)] = y_pred[int(count)]
                count = count + 1
    return SDF

