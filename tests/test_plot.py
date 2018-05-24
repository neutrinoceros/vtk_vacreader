from os import mkdir
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from vtk_vacreader import VacDataSorter

out = Path(__file__).parent / 'out/'
if not out.exists():
    mkdir(out)

datafile = Path(__file__).absolute().parent.parent / 'data' / 'hd142527_dusty0029.vtu'
assert Path(datafile).exists()
datafile = str(datafile)

myshape = (512, 128)

def test_2dplot():
    dh = VacDataSorter(datafile, data_shape=myshape)

    fig, ax = plt.subplots()
    phigrid, rgrid = dh.get_meshgrid()

    im = ax.pcolormesh(rgrid, phigrid, dh['rho'])
    fig.colorbar(im)

    #spiral detection:
    maxkey = dh['rho'].argmax(axis=1)

    ax.scatter(
        dh.get_axis(0),
        dh.get_axis(1)[maxkey],
        s=15, marker='+', c='r'
    )

    #stream lines
    # m2_pert = np.array([line - line.mean() for line in dh['m2']])
    # m2_pert2 = dh['m2'] - dh['m2'].mean(axis=0)
    # print (m2_pert - m2_pert2)
    # ax.streamplot(
    #     rgrid.T, phigrid.T,
    #     (dh['m1']/dh['rho']).T,
    #     (m2_pert/dh['rho']).T,
    #     density=2
    # )

    # density contour
    ax.contour(rgrid, phigrid, dh['rho'], colors='k')

    #local maximam detection
    from scipy.signal import argrelextrema
    maxima_keys = argrelextrema(dh['rho'], np.greater)
    ax.scatter(
        dh.get_axis(0)[maxima_keys[0]],
        dh.get_axis(1)[maxima_keys[1]],
        s=12, marker='x'
    )
    fig.savefig(str(out/'2dplot_polar.pdf'))


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

