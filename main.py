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

# Load the data with this function
def loadData(root="../data/"):
    # Gather data
    # train_out =         np.load('{}/train_out.npy'.format(root))
    test_out =          np.load('{}/test_out.npy'.format(root))
    # train_norm_2d =     np.load('{}/train_norm_2d.npy'.format(root))
    test_norm_2d =      np.load('{}/test_norm_2d.npy'.format(root))
    train_norm_2d_new = np.load('{}/train_norm_2d_new.npy'.format(root))
    train_out_new =     np.load('{}/train_out_new.npy'.format(root))
    return train_norm_2d_new, train_out_new, test_norm_2d, test_out

if __name__ == "__main__":
    from sys import argv
    # MPI INIT called here
    from mpi4py import MPI
    from mpigridsearch import HPCGridSearch
    import numpy as np
    # from mpi_logic import fullSyncro, masterWorker

    if '-f' not in argv:
        print("No json file given!")
        exit()
    else:
        params = argv[argv.index('-f') + 1]
    if '-fs' in argv:
        # Mode: full syncro
        mode = "fs"
    elif '-mw' in argv:
        mode = "mw"
    if '-d' in argv:
        root = argv[argv.index('-d') + 1]
    else:
        root = "../data/"
    if '-a' in argv:
        val = argv[argv.index('-a') + 1 ]
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
    x1, y1, x2, y2 = loadData(root=root)
    results = grid.search(x1=x1, y1=y1, x2=x2, y2=y2, 
            augmentation=augmentation, build_fn=create_model)
