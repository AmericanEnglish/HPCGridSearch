# from sys import argv
# MPI INIT called here
from mpi4py import MPI

# Determine number of threads
from os import environ
from json import load as loadf
import numpy as np
from multiprocessing import cpu_count
# For creating references to weights
from hashlib import sha256
from jsonpickle import encode as dencode
from os import mkdir
from os.path import isdir
# Autogeneration of training sets
from sklearn.model_selection import train_test_split
# Determine TensorFlow version for compatability
from tensorflow import __version__ as tfversion
tfversion = tfversion.split(".")
tfversion = list(map(lambda x: int(x), tfversion))
twoOh = [2, 0, 0]
if tfversion >= twoOh:
    from tensorflow.keras import backend as K
    from tensorflow.keras.optimizers import Adam
    from tensorflow import distribute as D
    import tensorflow as tf
    from   tensorflow.keras.callbacks import CSVLogger
    from tensorflow.keras.callbacks import Callback as KCallback  
    from tensorflow.keras.callbacks import LearningRateScheduler
else: # Assuming 1.XX
    import keras.backend as K
    from keras.utils.multi_gpu_utils import multi_gpu_model
    # from keras.optimizers import Adam

from functools import reduce
from datetime import datetime
import hashlib

# Original from: -> Adapted to use better real time metric
# https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
# Post about making custom metrics show up in the CSV
# https://stackoverflow.com/questions/48488549/keras-append-to-logs-from-callback
class TimeHistory(KCallback):
    def on_epoch_begin(self, epoch, logs):
        self.epoch_time_start = datetime.now()

    def on_epoch_end(self, epoch, logs):
        logs["epoch_time"] = deltaToString(datetime.now() - self.epoch_time_start)

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

    def setEnv(self, threads=None, ngpus=None):
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
        self.setGPUsPerTask()
        if len(self.grange) == 0:
            if tfversion < twoOh:
                try:  # very old version of tensorflow do weird things
                    K.set_session(
                        K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=self.threads,
                            # inter_op_parallelism_threads=threads, log_device_placement=True)))
                            inter_op_parallelism_threads=self.threads)))
                except:
                    import tensorflow as tf
                    config = tf.ConfigProto(intra_op_parallelism_threads=self.threads,
                        inter_op_parallelism_threads=self.threads)
                    K.tensorflow_backend.set_session(tf.Session(config=config))
            else: # assumed >= 2.0
                import tensorflow as tf
                tf.config.threading.set_inter_op_parallelism_threads(self.threads)
                tf.config.threading.set_intra_op_parallelism_threads(self.threads)
        else: # With GPUs
            import tensorflow as tf
            if tfversion < twoOh: # 2.0 reworks and strategies replace this method
                visible_devices = ",".join(map(lambda s: str(s), self.grange))
                gpu_options = tf.GPUOptions(visible_device_list=visible_devices)
                config = tf.ConfigProto(intra_op_parallelism_threads=self.threads,
                    inter_op_parallelism_threads=self.threads,
                    gpu_options=gpu_options)
                # config.gpu_options.visible_device_list = visible_devices
                K.tensorflow_backend.set_session(tf.Session(config=config))
                self.aprint("Visible devices set to {}".format(visible_devices))
                self.aprint("New session has been set")
            else: # This should still be done for 2.0+ gpus
                physical_devices = tf.config.list_physical_devices('GPU')
                physical_devices.sort()
                physical_devices = [physical_devices[int(i)] for i in self.grange]
                tf.config.set_visible_devices(physical_devices, 'GPU')
                # f = lambda x: tf.config.experimental.set_memory_growth(x, True)
                # map(f, physical_devices)
                tf.config.threading.set_inter_op_parallelism_threads(self.threads)
                tf.config.threading.set_intra_op_parallelism_threads(self.threads)
                self.aprint("Visible devices set to {}: {}".format(self.grange, physical_devices))
                self.phys = physical_devices
                self.vis  = list(map(lambda s: str(s), self.grange))
    # In the case of preferring CPU only, merely set autoGPU to false
    # Otherwise for manual management of GPUs, use this!
    def setGPUsPerTask(self, ngpus=None):
        if ngpus is None:
            if 'NGPUS' in environ.keys():
                ngpus = environ['NGPUS']
                try:
                    self.ngpus= int(ngpus)
                except:
                    self.ngpus = gpu_count()
            else:
                self.ngpus = gpu_count() 
        else:
            self.ngpus = gpu_count()
        lnode = MPI.Get_processor_name()
        self.aprint("Determined {} gpus for my node, {}".format(self.ngpus, lnode))
        # Create a NODE ONLY communicator to determine how many processes via 
        # Comm_split currently exist on your node. 
        #* This is by hostname so it only works in environments where hostnames are unique.

        # Create an mpi color
        # color = bytes(lnode, 'utf-8')
        color = int(hashlib.sha1(lnode.encode()).hexdigest(), 16) % (10 ** 8)
        # color = int(hashlib.sha256(lnode.encode()).hexdigest(), 16) % (10 ** 8)
        # color = int.from_bytes(color, byteorder='little')
        # Create node only communicator 
        self.aprint("My color is {}".format(color))
        lcomm = self.comm.Split(color, self.rank)
        # Get size of mpi processes from this node only communicator
        tasks_per_lnode = lcomm.Get_size()
        self.lrank = lcomm.Get_rank()
        self.lcomm = lcomm
        # Compute local GPUs per task. There is not a more obvious way of doing
        # this without passing even more environment variables.
        # Determine number of gpus to be used
        self.grange = list(range(self.lrank, self.ngpus, tasks_per_lnode))
        self.aprint("Local rank {}, my gpus are {}".format(self.lrank, self.grange))

    def setAutoGPU(self, autoGPU):
        self.agpu = autoGPU

    def search(self, x1=None, y1=None, x2=None, y2=None, augmentation=False, 
            validation_split=0.0, validation_data=None, autoGPU=True, shuffle=True,
            build_fn=None):
        # x1, y1, x2, y2 can also be strings from which the data should be
        # loaded, assume they are numpy arrays
        if self.agpu is None:
            self.agpu = autoGPU
            # self.setGPUsPerTask()
        # Load data file if a string is provided instead
        if x1 is None:
            print("ERROR: input data is None!")
            exit()
        elif isinstance(x1, str):
            x1 =     np.load(x1)
    
        if x2 is not None and isinstance(x2, str):
            x2 =     np.load(x2)
        if y1 is None:
            print("ERROR: output data is None!")
            exit()
        elif isinstance(y1, str):
            y1 =     np.load(y1)

        if y2 is not None and isinstance(y1, str):
            y2 =     np.load(y2)
        # If no training sets were provided them generate them    
        if x2 is None or y2 is None:
            x1, x2, y1, y2 = train_test_split(
                    x1, y1, test_size=None)

        if build_fn is None:
            self.rprint("ERROR: build_fn is None!")
        else:
            self.cm = build_fn
        self.setEnv()
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
            d = cgen(i)
            try:
                allResults.append(self.train_model(d)) #, cv=KFolds)
                with open('result-{:04d}.txt'.format(self.rank), 'a+') as outfile:
                    outfile.write("{}\n".format(allResults[-1]))
            # except Exception as e:
            except TypeError as e:
                self.aprint("Error: Failed to train {}, {}".format(d,e))
                d['acc'] = ''
                d['time'] = ''
                d['tacc'] = ''
                allResults.append(d)
                with open('result-{:04d}.txt'.format(self.rank), 'a+') as outfile:
                    outfile.write("{}\n".format(allResults[-1]))
        self.aprint('All tasks completed!')
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
        # Compute dictionary hash
        ident = dencode(params)
        chk = sha256(ident.encode("utf-8")).hexdigest()
        params['hash'] = chk
        # Do not forget to extract it
        for key in params.keys():
            if isinstance(key, list):
                params[key] = params[key][0]
        # Check for some basic defaults
        if 'batch_size' in params.keys():
            batch_size = params['batch_size']
        else:
            batch_size = 1
        if "epochs" in params.keys():
            epochs = params['epochs']
        else:
            epochs = 1
        if "learning_schedule" in params.keys():
            # Assume there exists some file
            # myfile.py
            # Which contains the learning schedule function
            # def fun(epoch, lr)
            # Then params.json would contain
            # myfile.fun
            all_things = params["learning_schedule"]
            all_things = all_things.split(".")
            baseImport, func = ".".join(all_things[:-1]), all_things[-1]
            # Import function dynamically
            baseImport = __import__(baseImport, globals(), locals(), [func], 0)
            # Extract function from returned object
            func = getattr(baseImport, func)
            schedule = LearningRateScheduler(func)
        else:   
            # This is the standard keras learning rate schedule
            def_schedule = lambda epoch, lr: lr
            schedule = LearningRateScheduler(def_schedule)
        # Just pass everything to create, the function may need it.
        model = self.cm(**params)
        if "gpu" in params.keys():
            num_gpus = params['gpu']
            if self.agpu:
                # opt = Adam(lr=lr)
                if num_gpus > 1 and tfversion < twoOh:
                    opt = model.optimizer
                    model = multi_gpu_model(model,num_gpus)
                    model.compile(opt, model.loss, metrics=['accuracy'])
                elif num_gpus > 1 and tfversion >= twoOh:
                    # strat = D.MirroredStrategy(devices=self.phys)
                    vis = list(map(lambda x: "/gpu:{}".format(x), self.vis))
                    # print("vis", vis)
                    # Currently crashing due to NCCL error
                    # strat = D.MirroredStrategy(devices=vis)
                    # Possible fix is using separate merge technique
                    strat = D.MirroredStrategy(devices=vis,
                            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
                    # strat = D.MirroredStrategy()
                    with strat.scope():
                        model = self.cm(**params)
                        opt = model.optimizer
                        model.compile(opt, model.loss, metrics=['accuracy'])
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
            self.aprint("Training a network {}...".format(datetime.now()))
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
            self.aprint("Training network {} :: {}...".format(chk, datetime.now()))
            # Fit the new model
            # history = model.fit(x=self.idata, y=self.odata, batch_size=batch_size,
                    # [self.idata, self.idata, self.idata],
            if not isdir("./training_logs/"):
                try:
                    mkdir("./training_logs/")
                except:
                    pass
            csv = CSVLogger("./training_logs/{}.log".format(chk),
                    append=False)
            start_time = datetime.now()
            history = model.fit(
                    x=self.idata,
                    y=self.odata,#, self.odata[:,1,:], self.odata[:,2,:]], 
                    batch_size=batch_size,
                    validation_split=0.2,
                   epochs=epochs, verbose=0, shuffle=True,
                   callbacks=[TimeHistory(),schedule,csv])
                   # epochs=epochs, verbose=1, shuffle=True)
            end_time = datetime.now()
            # Test accuracy
            # loss, accuracy = model.evaluate( x=self.idatae, y=self.odatae,
                    # [self.idatae, self.idatae, self.idatae],
            loss, accuracy  = model.evaluate( 
                x=self.idatae,
                y=self.odatae,# self.odatae[:,1,:], self.odatae[:,2,:]],
               batch_size=batch_size, verbose=0) 
        # Compute timing metrics
        run_time = deltaToString(end_time - start_time)
        params['time'] = str(run_time)
        # Consider accuracy to be the average of all three layer accuracies
        # acc = list(map(lambda x: sum(x)/len(x), zip(layer1_acc, layer2_acc, layer3_acc)))
        # accuracy = layer1_acc #+ layer2_acc + layer3_acc
        # accuracy /= 3
        params['acc']  = str(accuracy)
        # Gather all output accuracies
        keys = history.history.keys()
        # taccs = [history.history[key] for key in keys if "accuracy" in key]
        # loss = [history.history[key] for key in keys if "loss" in key]
        # print(history.history.keys())
        # params['tacc'] = list(
                # map(lambda x: sum(x)/len(x), zip(*taccs)))
        # params['tacc'] = history.history['accuracy']
        # params['loss'] = history.history['loss']
        # params['vloss'] = history.history['val_loss']
        # params['vacc'] = history.history['val_accuracy'] 
            # list( map(lambda x: sum(x)/len(x), zip(*loss)))
        # params['acc1'] = taccs[0]
        # params['acc2'] = taccs[1]
        # params['acc3'] = taccs[2]
        if not isdir("./weights/"):
            try:
                mkdir("./weights/")
            except:
                pass
        results = str(params)
        model.save("./weights/{}".format(chk))
        self.aprint("Trained! {}".format(results))
        # model.summary()
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
    def comboHelper(d):
        if isinstance(d, list):
            return len(d)
        else:
            return 1

    return reduce(
            lambda x, y: x*y, 
                (map(lambda key: comboHelper(params[key]), params.keys())))

def comboGenerator(n, params=None):
    if params is None:
        params = {}
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

    # if "data_multiplier" in keys:
        # del keys[keys.index("data_multiplier")]
    # else:
        # params["data_multiplier"] = [1]
    # keys.insert(0, "data_multiplier")
    # keys.append("data_multiplier")
    result = {}
    for key in keys:
        q = params[key]
        if isinstance(q, list):
            val  = len(params[key])
            result[key] = q[n % val]
        else:
            val = 1
            result[key] = q
        n = n // val
    # return keys, totals
    return result

def deltaToString(tme):
    sec = tme.total_seconds()
    hours = int(sec) // 60 // 60
    minutes = int(sec - hours* 60*60) // 60
    sec = sec - hours* 60*60 - minutes * 60
    return "{:02d}:{:02d}:{:05.2f}".format(hours, minutes, sec)

# from tensorflow.python.client import device_lib
def get_available_gpus():
    # local_device_protos = device_lib.list_local_devices()
    # return [x.name for x in local_device_protos if x.device_type == 'GPU']
    # SLURM sets this automatically, prevents a bad sessions
    if "CUDA_VISIBLE_DEVICES" in environ:
        gpus = environ['CUDA_VISIBLE_DEVICES'].split(",")
    else:
        gpus = []
    return gpus

def gpu_count():
    return len(get_available_gpus())
