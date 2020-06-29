# AUTHORS: CARLOR BARAJAS, CHARLIE BECKER, BIN WANG, WILL MAYFIELD, SARAH MURPHY
# DATE: 08/19/2019

# THIS SCRIPT DEMONSTRATES A METHOD OF TUNING HYPERPARAMETERS OF A DEEP 
# NEURAL NETWORK IN PARALLEL IN AN HPC ENVIRONMENT USING A COMBINATION 
# OF POPULAR PYTHON MODULES - DASK, SCIKIT-LEARN AND KERAS. DATA AND BASE
# CONVOLUTIONAL MODEL STRUCTURE IS BORROWED FROM THE 2019 AMS SHORT COURSE
# ON MAHCINE LEARNING IN PYTHON TAUGHT BY JOHN GAGNE FROM NCAR. THE SHORT 
# COURSE GITHUB CAN BE FOUND AT https://github.com/djgagne/ams-ml-python-course

################### DISTRIBUTE MODELS AND GATHER RESULTS ##################

from model import create_model

if __name__ == "__main__":
    from sys import argv
    # MPI INIT called here
    from mpi4py import MPI
    from mpigridsearch import HPCGridSearch
    import numpy as np
    # from mpi_logic import fullSyncro, masterWorker
    argvi = lambda x: argv[argv.index(x) + 1]
    if '-f' not in argv:
        print("No json file given!")
        exit()
    else:
        params = argvi('-f')
    if '-fs' in argv:
        # Mode: full syncro
        mode = "fs"
    elif '-mw' in argv:
        mode = "mw"

    if '-d' in argv:
        root = argvi('-d')
    else:
        root = "../data/"

    if '-x' in argv:
        x = argvi('-x')
    else:
        x = None
    if '-y' in argv:
        y = argvi('-y')
    else:
        y = None
    if '-xt' in argv:
        xt = argvi('-xt')
    else:
        xt = None
    if '-yt' in argv:
        yt = argvi('-yt')
    else:
        yt = None

    if '-a' in argv:
        val = argvi("-a")
        if val.lower() == "false" or val.lower() == 'f':
            augmentation = False
        elif val.lower() == "true" or val.lower() == 't':
            augmentation = True
        else: # lets you use 0 or 1 for T/F
            try:
                augmentation = bool(int(val))
            except:
                augmentation = False
    else:
        augmentation = False
    # Initialize the searching object
    grid = HPCGridSearch(params, pschema="fs")
    # def search(self, x1=None, y1=None, x2=None, y2=None, augmentation=False, 
    # Load the data
    results = grid.search(x1=x, y1=y, x2=xt, y2=yt, 
            augmentation=augmentation, build_fn=create_model)
