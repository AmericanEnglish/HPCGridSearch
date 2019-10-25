# AUTHORS: CARLOR BARAJAS, CHARLIE BECKER, BIN WANG, WILL MAYFIELD, SARAH MURPHY
# DATE: 08/19/2019

# THIS SCRIPT DEMONSTRATES A METHOD OF TUNING HYPERPARAMETERS OF A DEEP 
# NEURAL NETWORK IN PARALLEL IN AN HPC ENVIRONMENT USING A COMBINATION 
# OF POPULAR PYTHON MODULES - DASK, SCIKIT-LEARN AND KERAS. DATA AND BASE
# CONVOLUTIONAL MODEL STRUCTURE IS BORROWED FROM THE 2019 AMS SHORT COURSE
# ON MAHCINE LEARNING IN PYTHON TAUGHT BY JOHN GAGNE FROM NCAR. THE SHORT 
# COURSE GITHUB CAN BE FOUND AT https://github.com/djgagne/ams-ml-python-course

################### DISTRIBUTE MODELS AND GATHER RESULTS ##################
def create_model(learning_rate=0.001):
    
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
    # print(conv_model.summary())
    # if num_gpus > 1:
        # model = multi_gpu_model(conv_model, num_gpus)
    # else:
        # model = conv_model
    # model.compile(opt, "binary_crossentropy", metrics=['accuracy'])

    return conv_model


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
    # from mpi_logic import fullSyncro, masterWorker

    if '-f' not in argv:
        print("No json file given!")
        exit()
    if '-fs' in argv:
        # Mode: full syncro
        mode = "fs"
    elif '-mw' in argv:
        mode = "mw"
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
    # Initialize the searching object
    grid = HPCGridSearch("./technical_report.json", pschema="fs")
    # def search(self, x1=None, y1=None, x2=None, y2=None, augmentation=False, 
    # Load the data
    x1, y1, x2, y2 = loadData(root=root)
    results = grid.search(x1=x1, y1=y1, x2=x2, y2=y2, 
            augmentation=augmentation, build_fn=create_model)
