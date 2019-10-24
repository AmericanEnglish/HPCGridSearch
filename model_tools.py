from keras.models import Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, Dropout
from keras.layers import SpatialDropout2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K
# Enable all cores?
from multiprocessing import cpu_count
from os import environ
from keras.utils.multi_gpu_utils import multi_gpu_model
if 'OMP_NUM_THREADS' in environ.keys():
    threads = environ['OMP_NUM_THREADS']
    try:
        threads = int(threads)
    except:
        threads = cpu_count()
else:
    threads = cpu_count()
# Works on tensorflow 1.13 but not on 1.12
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
# except Exception as e:
    # print(e)
# Enable all cores done
import numpy as np
# For machine learning
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

#from imblearn.over_sampling import RandomOverSampler
from datetime import datetime

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def create_model1(learning_rate=0.001):
    
    # Deep convolutional neural network
    num_conv_filters = 8
    filter_width = 5
    conv_activation = "relu"
    
    # Input data in shape (instance, y, x, variable)
    conv_net_in = Input(shape=(32, 32, 3))  # train_norm_2d.shape[1:]
    
    # First 2D convolution Layer
    conv_net = Conv2D(num_conv_filters, (filter_width, filter_width), padding="same")(conv_net_in)
    conv_net = Activation(conv_activation)(conv_net)
    
    # Average pooling takes the mean in a 2x2 neighborhood to reduce the image size
    conv_net = AveragePooling2D()(conv_net)
    
    # Second set of convolution and pooling layers
    conv_net = Conv2D(num_conv_filters * 2, (filter_width, filter_width), padding="same")(conv_net)
    conv_net = Activation(conv_activation)(conv_net)
    conv_net = AveragePooling2D()(conv_net)
    
    # Third set of convolution and pooling layers
    conv_net = Conv2D(num_conv_filters * 4, (filter_width, filter_width), padding="same")(conv_net)
    conv_net = Activation(conv_activation)(conv_net)
    conv_net = AveragePooling2D()(conv_net)
    
    # Flatten the last convolutional layer into a long feature vector
    conv_net = Flatten()(conv_net)
    
    # Dense output layer, equivalent to a logistic regression on the last layer
    conv_net = Dense(1)(conv_net)
    conv_net = Activation("sigmoid")(conv_net)
    conv_model = Model(conv_net_in, conv_net)
    
    # Use the Adam optimizer with default parameters
    opt = Adam(lr=learning_rate)
    
    # Compile
    conv_model.compile(opt, "binary_crossentropy", metrics=['accuracy'])
    num_gpus = len(get_available_gpus())
    # print(conv_model.summary())
    model = multi_gpu_model(conv_model, num_gpus)
    model.compile(opt, "binary_crossentropy", metrics=['accuracy'])
    return model

# Import the new GAN as the model I care about
from wgan import create_model


#################### LOAD DATA AND DISTRIBUTE MODEL ########################
def loadData(root="../data/", batch_size=None, aug=False):
    # Gather data
    # train_out =         np.load('{}/train_out.npy'.format(root))
    test_out =          np.load('{}/test_out.npy'.format(root))
    # train_norm_2d =     np.load('{}/train_norm_2d.npy'.format(root))
    test_norm_2d =      np.load('{}/test_norm_2d.npy'.format(root))
    train_norm_2d_new = np.load('{}/train_norm_2d_new.npy'.format(root))
    train_out_new =     np.load('{}/train_out_new.npy'.format(root))

    return train_norm_2d_new, train_out_new, test_norm_2d, test_out


def generateAugmentedData(coeff=1,root="./data/"):
    # from multiporcessing import Pool, cpu_count

    train_norm_2d_new, train_out_new, test_norm_2d, test_out = loadData(root=root)
    datagen = ImageDataGenerator(
              rotation_range=5,
              width_shift_range=0,
              height_shift_range=0,
              shear_range=0,
              zoom_range=0,
              horizontal_flip=True,
              fill_mode='nearest')
    seed=1
    # total_size = train_out_new.shape[0] + test_out.shape[0]
    # print(total_size)
    data_generator = datagen.flow(train_norm_2d_new, 
            train_out_new, batch_size=train_out_new.shape[0],
            shuffle=True, seed=seed),
    data_generator = data_generator[0]
    ins = []
    outs = []
    for i in range(coeff):
        res = data_generator.next()
        ins.append(res[0])
        outs.append(res[1])
    return [np.concatenate(ins), np.concatenate(outs)]

###################### BUILD MODEL STRUCTURE ##############################
def train_model(params, augmentation=False,cv=3, root="../data/", rank=0):
    out_threshold = 0.005

    # Extract variables from the dictionary
    if "data_multiplier" in params:
        coeff = params["data_multiplier"][0]
        # del params['data_multiplier']
    else:
        coeff = 1
    # have to  pull out this so that it works with CVGridSearch
    batch_size = params['batch_size'][0]
    # del params['batch_size']

    if "learning_rate" in params.keys():
        lr = float(params["learning_rate"][0])
    else:
        lr=0.001

    if "epochs" in params.keys():
        epochs = params['epochs'][0]
    else:
        epochs = 1
    # Create Model
    cm = lambda : create_model(learning_rate=lr)
    train_norm_2d_new, train_out_new, test_norm_2d, test_out = loadData(root=root)
    model = cm()

    # Perform live augmentation if requested
    if augmentation == True:
       # print("[{:3d}] Running augmented training now, with augmentation".format(rank), flush=True)
       datagen = ImageDataGenerator(
                 rotation_range=5,
                 width_shift_range=0,
                 height_shift_range=0,
                 shear_range=0,
                 zoom_range=0,
                 horizontal_flip=True,
                 fill_mode='nearest')
       # datagen.fit(train_norm_2d_new)
       # We don't get grid search so we pull them out manually

       print("[{:3d}] Training a network {}...".format(rank, datetime.now()), flush=True)
       start_time = datetime.now()
       history=model.fit_generator(
            datagen.flow(train_norm_2d_new, 
                train_out_new, batch_size=batch_size,shuffle=True),
            steps_per_epoch=coeff*train_norm_2d_new.shape[0]//batch_size,
            epochs=epochs, verbose=0,
            use_multiprocessing=True,
            workers=threads)
       # What is this even for?  The verification score perhaps?
       # indices_test=np.where(test_out>out_threshold)[0]
       # test_out_pos=test_out[indices_test]
       # test_out_new=np.tile(test_out_pos,18)
       # test_out_new=np.concatenate((test_out,test_out_new),axis=0)
       # test_norm_2d_pos=test_norm_2d[indices_test,:,:,:]
       # test_norm_2d_new=np.tile(test_norm_2d_pos,(18,1,1,1))
       # test_norm_2d_new=np.concatenate((test_norm_2d,test_norm_2d_new),axis=0)
       end_time = datetime.now()
       # print("[{:3d}] Trained!".format(rank), flush=True)
       # start_time = datetime.now()
       # loss, accuracy, f1_score, precision, recall = model.evaluate( x=test_norm_2d, y=test_out, batch_size=batch_size, verbose=0) 
       loss, accuracy = model.evaluate( x=test_norm_2d, y=test_out, batch_size=batch_size, verbose=0) 
       run_time = deltaToString(end_time - start_time)
       # run_time_str = datetime.strptime(run_time, '%H:hour')
       # result = "{} :: acc -> {} :: {}".format(params, accuracy,
               # run_time)
       params['acc']  = accuracy
       params['time'] = str(run_time)
       results = str(params)
       # print("{} :: {}".format(params, run_time))
            # l.append(x)
       # return calc_verification_scores(model, test_norm_2d_new, test_out_new), t1-t0
       print("[{:3d}] Trained! {}".format(rank, results), flush=True)
       # return result

    else:
       print("[{:3d}] Training a network {}...".format(rank, datetime.now()), flush=True)
       # Train the model
       start_time = datetime.now()
       model.fit(x=train_norm_2d_new, y=train_out_new, batch_size=batch_size,
               epochs=epochs, verbose=0, shuffle=True)
                # steps_per_epoch=coeff*train_norm_2d_new.shape[0]//batch_size)

       end_time = datetime.now()
       # Check model
       loss, accuracy = model.evaluate( x=test_norm_2d, y=test_out,
               batch_size=batch_size, verbose=0) 
       run_time = deltaToString(end_time - start_time)
       # result = "{} :: acc -> {} :: {}".format(params, accuracy,
               # run_time)
       params['acc']  = accuracy
       params['time'] = str(run_time)
       results = str(params)
       print("[{:3d}] Trained! {}".format(rank, results), flush=True)

    return results


def calc_verification_scores(test_labels,predictions):
    
    model_auc = roc_auc_score(test_labels, predictions)
    model_brier_score = mean_squared_error(test_labels, predictions)
    climo_brier_score = mean_squared_error(test_labels, np.ones(test_labels.size) * test_labels.sum() / test_labels.size)
    model_brier_skill_score = 1 - model_brier_score / climo_brier_score
    print(f"AUC: {model_auc:0.3f}")
    print(f"Brier Score: {model_brier_score:0.3f}")
    print(f"Brier Score (Climatology): {climo_brier_score:0.3f}")
    print(f"Brier Skill Score: {model_brier_skill_score:0.3f}")
    return model_auc

def deltaToString(tme):
    sec = tme.total_seconds()
    hours = int(sec) // 60 // 60
    minutes = int(sec - hours* 60*60) // 60
    sec = sec - hours* 60*60 - minutes * 60
    return "{:02d}:{:02d}:{:010.7f}".format(hours, minutes, sec)
