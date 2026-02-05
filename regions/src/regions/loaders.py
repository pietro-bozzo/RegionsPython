from pathlib import Path
import numpy as np


def regionDataPath():
    # get path to regions/data/ folder

    # find scr/ directory, data/ must be at same level
    file_path = Path(__file__).resolve()
    parts = file_path.parts
    if 'src' not in parts:
        raise ValueError(f"src/ not found in path")
    idx = parts.index('src')

    return Path(*parts[:idx]) / 'data'


def loadAnatomyFile(file_path=None):
    # load .anat file, whose columns must be [rat, electrode, brain region]
    #
    # arguments:
    #     file_path    string = None, path to .anat file, default is nonlateral.anat located in regions/data/

    if file_path is None:
            file_path = regionDataPath() / 'nonlateral.anat'

    return np.genfromtxt(file_path,delimiter=",",comments="%",dtype=[("rat",int),("electrode",int),("region","U50")])