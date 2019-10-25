# from sys import argv
# MPI INIT called here
from mpi4py import MPI

# Determine number of threads
from os import environ

class HPCGridSearch:
    def __init__(self, param_grid, rank=None, size=None, comm=None, pschema="fs"):
        self.params = param_grid
        if comm is None:
            self.comm = MPI.COMM_WORLD
        if rank is None:
            self.rank = self.comm.Get_rank()
        if size is None:
            self.size = self.comm.Get_size()
        self.pschema = pschema
        self.threads = None

    def setThreads(self, threads=None):
        if threads is None:
            if 'OMP_NUM_THREADS' in environ.keys():
                threads = environ['OMP_NUM_THREADS']
                try:
                    self.threads = int(threads)
                except:
                    self.threads = cpu_count()
            else:
                self.threads = cpu_count()
        else:
            self.threads = threads
        try:
            K.set_session(
                K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=threads,
                    # inter_op_parallelism_threads=threads, log_device_placement=True)))
                    inter_op_parallelism_threads=threads)))
        except:
            import tensorflow as tf
            config = tf.ConfigProto(intra_op_parallelism_threads=threads,
                inter_op_parallelism_threads=threads)
            K.tensorflow_backend.set_session(tf.Session(config=config))

    def search(self, x1=None, y1=None, x2=None, y2=None, augmentation=False, 
            validation_split=0.0, validation_data=None, shuffle=True,build_fn=None):
        if build_fn is None:
            self.rprint("ERROR: build_fn is None!")
        else:
            self.cm = build_fn=None
        self.setThreads()
        self.idata = x1
        self.odata = y1
        self.idatae = x2
        self.odatae = y2
        self.augmentation = augmentation
        if pschema == 'fs':
            self.fullSyncro()
        elif pschema == 'mw':
            self.masterWorker()
        else:
            self.rprint("ERROR: {} is not a valid option for a parallel schema!".format(self.pschema))
    # Simple print commands to cut down on typing
    def rprint(self, string, *args):
        if self.rank == 0:
            print("[{:4d}] {}".format(self.rank, string), flush=True)

    def aprint(self, string, *args):
        print("[{:4d}] {}".format(self.rank, string), flush=True)
    ##

    def fullSyncro(self):
        # Determine additional input parameters
        if self.augmentation:
            self.rprint("Running augmented training now, with augmentation")
        else:
            self.rprint("Running training now, with no augmentation")
        # Generate a full collection of combinations
        allCombos = getMaxCombos(self.params)
        cgen = lambda n: comboGenerator(n, params=self.params)
        self.rprint("{} processes handling {} tasks".format(size, allCombos))
        # Map over all things belonging to the process
        allResults = []
        with open('result-{:03d}.txt'.format(rank), 'w') as outfile:
            # Clear out the old file
            outfile.write("")
            # pass

        for i in range(rank, allCombos, size):
                print("[{:3d}] Moving onto cgen({})".format(rank, i),  flush=True)
                allResults.append(self.train_model(cgen(i), cv=KFolds)

                with open('result-{:03d}.txt'.format(rank), 'a+') as outfile:
                    outfile.write("{}\n".format(allResults[-1]))
        return allResults


    def masterWorker(self):
        """

        Sends out initial work orders.
        Runs a for loop over additional work.
        When ANY MPI process reports back as finished (confirmed via status)
            That process is immediately send new work.
        After all work has been sent out the master process waits for all
            processes to finish.
        The master process then collects the remaining work.

        """
        # Control the processes
        if self.rank == 0:
            self.rprint("Process {} ready to control!".format(rank))
            cgen = lambda n: comboGenerator(n, params=self.params)
            # First generate all work
            # allCombos = generateAllCombinations(data)
            allCombos = getMaxCombos(data)
            self.rprint("{} processes handling {} tasks".format(size, allCombos))
            # print("[{:3d}] {}".format(rank, allCombos))
            # Do this with a generator later!

            # Send work to all processes
            for irank in range(1,size):
                item = cgen(irank - 1)
                comm.send(item, dest=irank, tag=0)
                self.rprint("Work sent to process {}".format(irank))
            totalWork = allCombos

            for workNum in range(size - 1, totalWork):
                # Now wait for completed work to be sent
                status = MPI.Status()
                res = comm.recv(source=MPI.ANY_SOURCE, tag=0,status=status)
                irank = status.source
                item = cgen(workNum)
                self.rprint("Sending {} to process {}".format(item, irank))
                self.comm.send(item, dest=irank, tag=0)
                # Write work to file
                with open('results.txt', 'w') as outfile:
                    outfile.write("{}\n".format(res))
            # Shut down!
            # Collect remaining work
            for irank in range(1,size):
                # Collect all remaining work!
                res = self.comm.recv(source=irank, tag=0, status=None)
                with open('results.txt', 'w') as outfile:
                    outfile.write("{}\n".format(res))
                # Send shutdown signal
                self.comm.send(None, dest=irank, tag=0)
            # For book keeping sake, still tally completed work
            self.rprint("Completed {} tasks!".format(totalWork))

        # Do work!
        else:
            self.aprint("Process {} ready to work!".format(rank))
            res = {}
            master = 0
            res = comm.recv(source=master, tag=0, status=None)
            self.aprint("Received {} from Process {}".format(res, master))
            while res is not None:
                # Do job
                res = self.train_model(res, augmentation=augmentation, cv=KFolds,
                        root=root, rank=rank)
                # Send Completed Work
                comm.send(res, dest=master, tag=0)
                # Receive Job 
                res = comm.recv(source=master, tag=0, status=None)
                self.aprint("Received {} from Process {}".format(res, master))
            # Shut down!

    def train_model(self):
        # Check for some basic defaults
        if 'batch_size' in self.params:
            batch_size = self.params['batch_size'][0]
        else:
            batch_size = 1
        if "epochs" in params.keys():
            epochs = params['epochs'][0]
        else:
            epochs = 1
        if "learning_rate" in params.keys():
            lr = float(params["learning_rate"][0])
            self.cm = lambda : self.cm(learning_rate=lr)
        else:
            lr=0.001
        # train_norm_2d_new, train_out_new, test_norm_2d, test_out = loadData(root=root)
        model = self.cm()
        if self.augmentation:
            # Create a primitive augmentation object
            datagen = ImageDataGenerator(
                 rotation_range=5,
                 width_shift_range=0,
                 height_shift_range=0,
                 shear_range=0,
                 zoom_range=0,
                 horizontal_flip=True,
                 fill_mode='nearest')
            self.rprint("Training a network {}...".format(datetime.now()))
            # Fit the new model
            start_time = datetime.now()
            history=model.fit_generator(
                datagen.flow(self.idata, 
                    self.odata, batch_size=batch_size,shuffle=True),
                steps_per_epoch=coeff*self.idata.shape[0]//batch_size,
                epochs=epochs, verbose=0,
                use_multiprocessing=True,
                workers=threads)
            end_time = datetime.now()
            # Test accuracy
            loss, accuracy = model.evaluate( x=self.idatae, y=self.odatae, batch_size=batch_size, verbose=0) 
        else:
            self.rprint("Training a network {}...".format(datetime.now()))
            # Fit the new model
            start_time = datetime.now()
            model.fit(x=self.idata, y=self.odata, batch_size=batch_size,
                   epochs=epochs, verbose=0, shuffle=True)
            end_time = datetime.now()
            # Test accuracy
            loss, accuracy = model.evaluate( x=self.idatae, y=self.odatae,
                   batch_size=batch_size, verbose=0) 
        # Compute timing metrics
        run_time = deltaToString(end_time - start_time)
        params['acc']  = accuracy
        params['time'] = str(run_time)
        results = str(params)
        self.aprint("Trained! {}".format(results))
        return results
### THESE ARE HELPER FUNCTIONS
def getMaxCombos(params):
    return reduce(lambda x, y: x*y, (map(lambda key: len(params[key]),
        params.keys())))

def comboGenerator(n, params={}):
    """(int, dictionary) -> dictionary

    Acts as a pseudo-generator. Given an index it returns a dictionary.
    If you were to generate a list of all possible combinations of keys
    in the dictionary then this would return the ith item in that list.

    While there is a lot more sorting, it should be very very fast especially
    when compared the time and memory it takes to fully generate the whole list
    of combos instead.
    """
    keys = sorted(params.keys(), reverse=True)
    maxCombos = getMaxCombos(params)
    # print(maxCombos)
    if n >= maxCombos:
        return None

    if "data_multiplier" in keys:
        del keys[keys.index("data_multiplier")]
    else:
        params["data_multiplier"] = [1]
    # keys.insert(0, "data_multiplier")
    keys.append("data_multiplier")
    result = {}
    for key in keys:
        val  = len(params[key])
        result[key] = [params[key][n % val]]
        n = n // val
    # return keys, totals
    return result
