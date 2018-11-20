from pathlib import Path
from collections import defaultdict
from vtk_vacreader import VacDataSorter as VDS


class VacDataLoader:
    def __init__(self, basename:str, shape:tuple=None):
        self.basename = basename
        self.shape = shape
        self._loaded = defaultdict(bool)
        self._data = {}

    def __getitem__(self, key:int):
        targetfile = f"{self.basename}{str(key).zfill(4)}.vtu"
        if not self._loaded[key]:
            if not Path(targetfile).exists():
                raise FileNotFoundError(targetfile)
            else:
                self._data.update({key: VDS(file_name=targetfile, shape=self.shape)})
                self._loaded[key] = True
        return self._data[key]
