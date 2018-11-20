from pathlib import Path
from vtk_vacreader import VacDataLoader, VacDataSorter

basename = Path(__file__).absolute().parent.parent / 'data' / 'graindrift'
myshape = (512, 128)

def test_load_existing_data():
    vdl = VacDataLoader(
        basename=basename,
        outputs=[0,1,10],
        shape=myshape
    )

    ntest = 0
    vds = VacDataSorter(
        file_name=f"{basename}{str(ntest).zfill(4)}.vtu",
        shape=myshape
    )
    assert vdl[ntest] == vds
