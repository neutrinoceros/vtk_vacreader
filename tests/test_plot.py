from os import mkdir
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from vtk_vacreader import VacDataSorter

out = Path(__file__).parent / 'out/'
if not out.exists():
    mkdir(out)

datafile = '/home/crobert/dev/amrvac_nice/tests/disk/transition/out/hd142527_dusty0029.vtu'
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

