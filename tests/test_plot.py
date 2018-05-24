from os import mkdir
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from vtk_vacreader import VacDataSorter

out = Path(__file__).parent / 'out/'
if not out.exists():
    mkdir(out)

datafile = Path(__file__).absolute().parent.parent / 'data' / 'hd142527_dusty0029.vtu'
assert Path(datafile).exists()
datafile = str(datafile)

myshape = (512, 128)

def test_2dplot_polar():
    dh = VacDataSorter(datafile, data_shape=myshape)

    fig, ax = plt.subplots()
    phigrid, rgrid = dh.get_meshgrid()

    im = ax.pcolormesh(rgrid, phigrid, dh['rho'])
    fig.colorbar(im)

    # density contour
    ax.contour(rgrid, phigrid, dh['rho'], colors='k')

    #local maxima detection
    max0 = argrelextrema(dh['rho'], np.greater, axis=0)
    max1 = argrelextrema(dh['rho'], np.greater, axis=1)
    maxs = np.array(list(set(zip(max0[0], max0[1])).intersection(zip(max1[0], max1[1]))))

    ax.scatter(
        dh.get_axis(0)[maxs[:,0]],
        dh.get_axis(1)[maxs[:,1]],
        s=20, marker='x'
    )
    fig.savefig(str(out/'2dplot_polar.png'))


def test_disk_shape():
    dh = VacDataSorter(datafile, data_shape=myshape)

    fig, ax = plt.subplots()
    phigrid, rgrid = dh.get_meshgrid()
    Xgrid = rgrid*np.cos(phigrid)
    Ygrid = rgrid*np.sin(phigrid)

    im = ax.pcolormesh(Xgrid, Ygrid, dh['rho'], cmap='magma')
    fig.colorbar(im)
    ax.set_aspect('equal')

    #spiral detection
    maxkey = dh['rho'].argmax(axis=1)
    rvect = dh.get_axis(0)
    phivect = dh.get_axis(1)[maxkey]
    ax.scatter(
        rvect*np.cos(phivect),
        rvect*np.sin(phivect),
        s=2, marker='+', c='b'
    )

    fig.savefig(str(out/'2dplot_cartesian.png'))


def test_profile():
    dh = VacDataSorter(datafile, data_shape=myshape)
    fig, ax = plt.subplots()
    ax.plot(
        dh.get_axis(0),
        dh['rho'].mean(axis=1)
    )
    fig.savefig(str(out/'1dplot_profile.png'))


def test_coupe():
    dh = VacDataSorter(datafile, data_shape=myshape)
    fig, ax = plt.subplots()
    ax.plot(
        dh.get_axis(0),
        dh['rho'][:,50]
    )
    fig.savefig(str(out/'1dplot_coupe.png'))

