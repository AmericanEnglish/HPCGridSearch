try:
    from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, Dropout
    from keras.optimizers import SGD, Adam
    from keras.layers import SpatialDropout2D
    from keras.models import Model, save_model, load_model
except: # lazily assume 2.0 is why it failed
    from tensorflow.keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D, Flatten, LeakyReLU, Dropout
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.layers import SpatialDropout2D
    from tensorflow.keras.models import Model, save_model, load_model

def create_model(**kwargs):
    model = create_model_tornado(kwargs)
    return model

def create_model_tornado(learning_rate=0.001):
    
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

