'''A simple data holder class for non amr runs.'''

import numpy as np
import vtk
from vtk.util import numpy_support as nps


class VacDataSorter:
    _readers = {
        'vtu': vtk.vtkXMLUnstructuredGridReader #note : StructuredGrid can not be used because dimensions are not stored by AMRVAC (as they usually don't make sense)
    }

    def __init__(self, file_name:str, data_shape:tuple=None):
        file_type = file_name.split('.')[-1]
        self.reader = __class__._readers[file_type]()
        self.reader.SetFileName(file_name)
        self.reader.Update()

        self.data_shape = data_shape

        cd = self.reader.GetOutput().GetCellData()

        sort_key = self._get_sort_key()

        self.fields = {}
        for i in range(cd.GetNumberOfArrays()):
            varname = cd.GetArrayName(i)
            self.fields.update({varname: nps.vtk_to_numpy(cd.GetArray(varname))[sort_key]})

        if data_shape:
            for k in self.fields.keys():
                self.fields[k].shape = data_shape

    def __getitem__(self, key):
        return self.fields[key]

    def _get_sort_key(self):
        data = self.reader.GetOutput()
        raw_cell_coords = np.empty((data.GetNumberOfCells(), 3))
        for i in range(data.GetNumberOfCells()):
            cell_corners = nps.vtk_to_numpy(data.GetCell(i).GetPoints().GetData())
            raw_cell_coords[i] = np.array([cell_corners[:,n].mean() for n in range(cell_corners.shape[1])])

        cell_coords = np.array(
            [tuple(line) for line in raw_cell_coords],
            dtype=[('r', 'f4'), ('phi', 'f4'), ('z', 'f4')]
        )
        return cell_coords.argsort(order=['r', 'phi'])

    def get_axis(self, axis:int=0):
        axis_bounds = self.reader.GetOutput().GetBounds()[2*axis : 2*(axis+1)]
        return np.linspace(*axis_bounds, self.data_shape[axis])

    def get_meshgrid(self, dim:int=2):
        if dim != 2:
            raise NotImplementedError
        vectors = [self.get_axis(ax) for ax in range(dim)]
        vectors.reverse()
        return np.meshgrid(*vectors)
