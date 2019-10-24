# AUTHORS: CARLOR BARAJAS, CHARLIE BECKER, BIN WANG, WILL MAYFIELD, SARAH MURPHY
# DATE: 08/19/2019

# THIS SCRIPT DEMONSTRATES A METHOD OF TUNING HYPERPARAMETERS OF A DEEP 
# NEURAL NETWORK IN PARALLEL IN AN HPC ENVIRONMENT USING A COMBINATION 
# OF POPULAR PYTHON MODULES - DASK, SCIKIT-LEARN AND KERAS. DATA AND BASE
# CONVOLUTIONAL MODEL STRUCTURE IS BORROWED FROM THE 2019 AMS SHORT COURSE
# ON MAHCINE LEARNING IN PYTHON TAUGHT BY JOHN GAGNE FROM NCAR. THE SHORT 
# COURSE GITHUB CAN BE FOUND AT https://github.com/djgagne/ams-ml-python-course

################### DISTRIBUTE MODELS AND GATHER RESULTS ##################
if __name__ == "__main__":
    from sys import argv
    # MPI INIT called here
    from mpi4py import MPI
    from mpi_logic import fullSyncro, masterWorker

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if '-f' not in argv:
        print("No data file given!")
        exit()
    if '-fs' in argv:
        # Mode: full syncro
        fullSyncro(  argv, rank, size, comm)
    elif '-mw' in argv:
        if size > 1:
            masterWorker(argv, rank, size, comm)
        else:
            print("Size ({}) is too small for master-worker!".format(size))
            print("Switching to fullSyncro for single process work")
            fullSyncro(  argv, rank, size, comm)
    else:
        if rank == 0:
            print("No mode selected!")
            exit()
    print("[{:3d}] All tasks completed!".format(rank))
