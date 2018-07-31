'''A minimal vtk wrapper to automate (vtk -> numpy) conversions.'''

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class VacDataSorter:
    '''A simple data holder class for non-amr runs.'''

    _readers = {
        #note : StructuredGrid can not be used because dimensions
        #are not stored by AMRVAC (as they usually don't make sense)
        'vtu': vtk.vtkXMLUnstructuredGridReader
    }

    def __init__(self, file_name:str, shape:tuple=None):
        if not Path(file_name).exists():
            raise FileNotFoundError(file_name)

        self.file_name = file_name

        #init vtk reader
        file_type = file_name.split('.')[-1]
        self.reader = __class__._readers[file_type]()
        self.reader.SetFileName(file_name)
        self.reader.Update()

        #set fields dictionnary
        cd = self.reader.GetOutput().GetCellData()
        sort_key = self._get_sort_key()
        self.fields = {}
        for i in range(cd.GetNumberOfArrays()):
            arrname = cd.GetArrayName(i)
            self.fields.update({arrname: vtk_to_numpy(cd.GetArray(arrname))[sort_key]})

        #optional reshaping (shape can not be read internally)
        self.shape = shape
        if self.shape:
            for k in self.fields:
                self.fields[k].shape = shape
        else:
            k0 = [k for k in self.fields.keys()][0]
            self.shape = (len(self.fields[k0]),)

    def __getitem__(self, key) -> np.ndarray:
        '''Mimic the behavior of the embedded dictionnary for scrapping'''
        return self.fields[key]

    def __iter__(self) -> (str, np.ndarray):
        '''Allow iteration over fields.items() at base level'''
        for k,v in self.fields.items():
            yield k,v

    def get_ticks(self, axis:int=0) -> np.ndarray:
        '''Reconstruct an array with cell coordinates along given axis.'''
        if isinstance(axis, str):
            axis = {'r': 0, 'phi': 1}[axis]

        axis_bounds = self.reader.GetOutput().GetBounds()[2*axis : 2*(axis+1)]
        npoints = self.shape[axis]
        if axis == 1: #phi
            ticks = np.linspace(*axis_bounds, npoints)
        elif axis == 0: #r
            #emulate AMRVAC
            base = np.linspace(*axis_bounds, npoints+1)
            ticks = np.empty(npoints)
            for i in range(npoints):
                ticks[i] = 0.5*(base[i] + base[i+1])
        return ticks

    def get_meshgrid(self, dim:int=2) -> list:
        '''Reconstruct an evenly spaced (2d) grid to locate cells.

        return is meant to be unpacked and used as input for
        matplotlib.pyplot.pcolormesh()
        '''
        if dim != 2:
            raise NotImplementedError
        vectors = [self.get_ticks(ax) for ax in range(dim)]
        vectors.reverse()
        return np.meshgrid(*vectors)

    def _get_sort_key(self) -> np.array:
        '''Allow reordering of data points against location in the physical space.'''
        data = self.reader.GetOutput()
        raw_cell_coords = np.empty((data.GetNumberOfCells(), 3))
        for i in range(data.GetNumberOfCells()):
            cell_corners = vtk_to_numpy(data.GetCell(i).GetPoints().GetData())
            raw_cell_coords[i] = np.array(
                [cell_corners[:,n].mean() for n in range(cell_corners.shape[1])]
            )

        cell_coords = np.array(
            [tuple(line) for line in raw_cell_coords],
            dtype=[('r', 'f4'), ('phi', 'f4'), ('z', 'f4')]
        )
        return cell_coords.argsort(order=['r', 'phi'])


    # deprecated!
    def get_axis(self, axis:int=0) -> np.array:
        '''Reconstruct an evenly spaced array for cell coordinates along given axis.'''
        from warnings import warn
        message = 'This method <VacDataSorter.get_axis()> is deprecated, use <VacDataSorter.get_ticks()> instead'
        warn(message, DeprecationWarning, stacklevel=2)
        axis_bounds = self.reader.GetOutput().GetBounds()[2*axis : 2*(axis+1)]
        return np.linspace(*axis_bounds, self.shape[axis])
