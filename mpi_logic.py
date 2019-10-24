from mpi4py import MPI
from json import load as loadf

# Simple globals
KFolds = 3

# My helper tools file
# from codetools import generateAllCombinations
from codetools import comboGenerator, getMaxCombos
from model_tools import train_model


def fullSyncro(argv, rank, size, comm):
    # Determine additional input parameters
    if '-d' in argv:
        root = argv[argv.index('-d' ) + 1]
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

    if augmentation:
        print("[{:3d}] Running augmented training now, with augmentation".format(rank), flush=True)
    else:
        print("[{:3d}] Running training now, with no augmentation".format(rank), flush=True)
    # Generate a full collection of combinations
    filename = argv[argv.index('-f') + 1]
    with open(filename, 'r') as inFile:
        data = loadf(inFile)
    # allCombos = generateAllCombinations(data)
    allCombos = getMaxCombos(data)
    cgen = lambda n: comboGenerator(n, params=data)
    if rank == 0:
        print("[{:3d}] {} processes handling {} tasks".format(rank, size,
            allCombos))
        # print(allCombos)
    # Map over all things belonging to the process
    allResults = []
    with open('result-{:03d}.txt'.format(rank), 'w') as outfile:
        # Clear out the old file
        outfile.write("")

    for i in range(rank, allCombos, size):
            print("[{:3d}] Moving onto cgen({})".format(rank, i),  flush=True)
            allResults.append(train_model(cgen(i),
                augmentation=augmentation, cv=KFolds, root=root, rank=rank))
            with open('result-{:03d}.txt'.format(rank), 'a+') as outfile:
                outfile.write("{}\n".format(allResults[-1]))

    # comm.barrier()
    # with open('result-{:03d}.txt'.format(rank), 'w') as outfile:
        # for item in allResults:
            # outfile.write("{}\n".format(item))

    # MPI_Finalize()

def masterWorker(argv, rank, size, comm):
    # Determine additional input parameters
    if '-d' in argv:
        root = argv[argv.index('-d' + 1)]
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


    # Control the processes
    if rank == 0:
        print("[{:3d}] Process {} ready to control!".format(rank, rank))
        filename = argv[argv.index('-f') + 1]
        with open(filename, 'r') as inFile:
            data = loadf(inFile)
        cgen = lambda n: comboGenerator(n, params=data)
        # First generate all work
        # allCombos = generateAllCombinations(data)
        allCombos = getMaxCombos(data)
        print("[{:3d}] {} processes handling {} tasks".format(rank, size, allCombos))
        # print("[{:3d}] {}".format(rank, allCombos))
        # Do this with a generator later!

        # Send work to all processes
        for irank in range(1,size):
            item = cgen(irank - 1)
            comm.send(item, dest=irank, tag=0)
            print("[{:3d}] Work sent to process {}".format(rank, irank))
        totalWork = allCombos

        for workNum in range(size - 1, totalWork):
            # Now wait for completed work to be sent
            status = MPI.Status()
            res = comm.recv(source=MPI.ANY_SOURCE, tag=0,status=status)
            irank = status.source
            item = cgen(workNum)
            print("[{:3d}] Sending {} to process {}".format(rank,
                item, irank))
            comm.send(item, dest=irank, tag=0)
            # Write work to file
            with open('results.txt', 'w') as outfile:
                outfile.write("{}\n".format(res))
        # Shut down!
        # Collect remaining work
        for irank in range(1,size):
            # Collect all remaining work!
            res = comm.recv(source=irank, tag=0, status=None)
            with open('results.txt', 'w') as outfile:
                outfile.write("{}\n".format(res))
            # Send shutdown signal
            comm.send(None, dest=irank, tag=0)
        # For book keeping sake, still tally completed work
        print("[{:3d}] Complete {} tasks!".format(rank, totalWork))

    # Do work!
    else:
        print("[{:3d}] Process {} ready to work!".format(rank, rank))
        res = {}
        master = 0
        res = comm.recv(source=master, tag=0, status=None)
        print("[{:3d}] Received {} from Process {}".format(rank, res, master))
        while res is not None:
            # Do job
            res = train_model(res, augmentation=augmentation, cv=KFolds,
                    root=root, rank=rank)
            # Send Completed Work
            comm.send(res, dest=master, tag=0)
            # Receive Job 
            res = comm.recv(source=master, tag=0, status=None)
            print("[{:3d}] Received {} from Process {}".format(rank, res, master))
        # Shut down!

