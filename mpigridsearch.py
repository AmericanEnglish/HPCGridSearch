# from sys import argv
# MPI INIT called here
from mpi4py import MPI

# Determine number of threads
from os import environ
from json import load as loadf
import numpy as np
from multiprocessing import cpu_count
import keras.backend as K
from functools import reduce
from datetime import datetime

#
from keras.utils.multi_gpu_utils import multi_gpu_model

class HPCGridSearch:
    def __init__(self, param_grid, rank=None, size=None, comm=None, pschema="fs"):
        # Load the json file provided in the string!
        if isinstance(param_grid, str):
            with open(param_grid, 'r') as inFile:
                self.params = loadf(inFile)
        elif isinstance(param_grid, dict):
            self.params = param_grid


        if comm is None:
            self.comm = MPI.COMM_WORLD
        if rank is None:
            self.rank = self.comm.Get_rank()
        if size is None:
            self.size = self.comm.Get_size()
        self.pschema = pschema
        self.threads = None
        self.agpu = None
    def setThreads(self, threads=None):
        if threads is None:
            if 'OMP_NUM_THREADS' in environ.keys():
                threads = environ['OMP_NUM_THREADS']
                try:
                    self.threads = int(threads)
                except:
                    self.threads = int(cpu_count())
            else:
                self.threads = int(cpu_count())
        else:
            self.threads = threads
        try:
            K.set_session(
                K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=self.threads,
                    # inter_op_parallelism_threads=threads, log_device_placement=True)))
                    inter_op_parallelism_threads=self.threads)))
        except:
            import tensorflow as tf
            config = tf.ConfigProto(intra_op_parallelism_threads=self.threads,
                inter_op_parallelism_threads=self.threads)
            K.tensorflow_backend.set_session(tf.Session(config=config))
    # In the case of preferring CPU only, merely set autoGPU to false
    # Otherwise for manual management of GPUs, use this!
    def setAutoGPU(self, autoGPU):
        self.agpu = autoGPU

    def search(self, x1=None, y1=None, x2=None, y2=None, augmentation=False, 
            validation_split=0.0, validation_data=None, autoGPU=True, shuffle=True,build_fn=None):
        # x1, y1, x2, y2 can also be strings from which the data should be
        # loaded, assume they are numpy arrays
        if self.agpu is None:
            self.agpu = autoGPU
        # Load data file if a string is provided instead
        if isinstance(x1, str):
            x1 =     np.load(x1)
        if isinstance(x2, str):
            x2 =     np.load(x2)
        if isinstance(y1, str):
            y1 =     np.load(y1)
        if isinstance(y1, str):
            y2 =     np.load(y2)

        if build_fn is None:
            self.rprint("ERROR: build_fn is None!")
        else:
            self.cm = build_fn
        self.setThreads()
        self.idata = x1
        self.odata = y1
        self.idatae = x2
        self.odatae = y2
        self.augmentation = augmentation
        if self.pschema == 'fs':
            self.fullSyncro()
        elif self.pschema == 'mw':
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
        self.rprint("{} processes handling {} tasks".format(self.size, allCombos))
        # Map over all things belonging to the process
        allResults = []
        with open('result-{:04d}.txt'.format(self.rank), 'w') as outfile:
            # Clear out the old file
            outfile.write("")
            # pass

        for i in range(self.rank, allCombos, self.size):
            self.aprint("Moving onto cgen({})".format(i))
            allResults.append(self.train_model(cgen(i))) #, cv=KFolds)
            with open('result-{:04d}.txt'.format(self.rank), 'a+') as outfile:
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
            self.rprint("Process {} ready to control!".format(self.rank))
            cgen = lambda n: comboGenerator(n, params=self.params)
            # First generate all work
            # allCombos = generateAllCombinations(data)
            allCombos = getMaxCombos(data)
            self.rprint("{} processes handling {} tasks".format(self.size, allCombos))
            # print("[{:3d}] {}".format(rank, allCombos))
            # Do this with a generator later!

            # Send work to all processes
            for irank in range(1,self.size):
                item = cgen(irank - 1)
                comm.send(item, dest=irank, tag=0)
                self.rprint("Work sent to process {}".format(irank))
            totalWork = allCombos

            for workNum in range(self.size - 1, totalWork):
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
            for irank in range(1,self.size):
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
            self.aprint("Process {} ready to work!".format(self.rank))
            res = {}
            master = 0
            res = comm.recv(source=master, tag=0, status=None)
            self.aprint("Received {} from Process {}".format(res, master))
            while res is not None:
                # Do job
                # res = self.train_model(res, augmentation=augmentation,
                        # cv=KFolds)
                res = self.train_model(res)
                # Send Completed Work
                comm.send(res, dest=master, tag=0)
                # Receive Job 
                res = comm.recv(source=master, tag=0, status=None)
                self.aprint("Received {} from Process {}".format(res, master))
            # Shut down!

    def train_model(self, params):
        # Check for some basic defaults
        if 'batch_size' in params.keys():
            batch_size = params['batch_size'][0]
        else:
            batch_size = 1
        if "epochs" in params.keys():
            epochs = params['epochs'][0]
        else:
            epochs = 1
        if "learning_rate" in params.keys():
            lr = float(params["learning_rate"][0])
            cm = lambda : self.cm(learning_rate=lr)
        else:
            lr=0.001
        model = cm()
        if "gpu" in params.keys():
            num_gpus = params['gpu'][0]
            if self.agpu:
                if num_gpus > 1:
                    model = multi_gpu_model(num_gpus)
                    model.compile(opt, "binary_crossentropy", metrics=['accuracy'])
        # else:
            # num_gpus = 0
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
                    self.odata, batch_size=batch_size,shuffle=shuffle),
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

    def calc_verification_scores(self,test_labels,predictions):
        
        model_auc = roc_auc_score(test_labels, predictions)
        model_brier_score = mean_squared_error(test_labels, predictions)
        climo_brier_score = mean_squared_error(test_labels, np.ones(test_labels.size) * test_labels.sum() / test_labels.size)
        model_brier_skill_score = 1 - model_brier_score / climo_brier_score
        # print(f"AUC: {model_auc:0.3f}")
        # print(f"Brier Score: {model_brier_score:0.3f}")
        # print(f"Brier Score (Climatology): {climo_brier_score:0.3f}")
        # print(f"Brier Skill Score: {model_brier_skill_score:0.3f}")
        return model_auc

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

def deltaToString(tme):
    sec = tme.total_seconds()
    hours = int(sec) // 60 // 60
    minutes = int(sec - hours* 60*60) // 60
    sec = sec - hours* 60*60 - minutes * 60
    return "{:02d}:{:02d}:{:010.7f}".format(hours, minutes, sec)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
