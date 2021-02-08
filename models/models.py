from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, BatchNormalization, UpSampling2D, Concatenate, Activation, \
    Cropping2D, MaxPooling2D, Conv2DTranspose, Dropout, Reshape


def ConvBlock(n_conv, nfilter, layerin):
    conv = layerin 
    
    for i in range(n_conv): 
        conv = Convolution2D(nfilter, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.2)(conv)
    return conv 

    
def context_target_unet(context_input, target_input, nClass, n_conv=2, nfilter0=32, growth=2):
    context_inputlayer = Input((context_input[0], context_input[1], context_input[2]), name="context")
    target_inputlayer = Input((target_input[0], target_input[1], target_input[2]), name="target")

    # Block 1 - down: context: 2.0 (256, 256)
    nfilter = nfilter0
    c_conv1_d = ConvBlock(n_conv, nfilter, context_inputlayer)
    c_pool1_d = MaxPooling2D(pool_size=(2, 2))(c_conv1_d)

    # Block 2 - down: context: 4.0 (128, 128)
    nfilter *= growth
    c_conv2_d = ConvBlock(n_conv, nfilter, c_pool1_d)
    c_pool2_d = MaxPooling2D(pool_size=(2, 2))(c_conv2_d)

    # Block 3 - down: context: 8.0 (64, 64)
    nfilter *= growth
    c_conv3_d = ConvBlock(n_conv, nfilter, c_pool2_d)
    c_pool3_d = MaxPooling2D(pool_size=(2, 2))(c_conv3_d)

    # Block 3_1 - down: context: 16.0 (32, 32)
    c_conv3_1_d = ConvBlock(n_conv, nfilter, c_pool3_d)
    c_pool3_1_d = MaxPooling2D(pool_size=(2, 2))(c_conv3_1_d)

    # Block 4: context: 32.0 (16, 16)
    # nfilter *= growth
    c_conv4_d = ConvBlock(n_conv, nfilter, c_pool3_1_d)
    c_conv4_d = ConvBlock(n_conv, nfilter, c_conv4_d)

    # Block 5_0 - up: context: 16.0 (32, 32)
    c_up5_0 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(c_conv4_d)
    c_merge5_0 = Concatenate()([c_conv3_1_d, c_up5_0])
    c_conv5_0_u = ConvBlock(n_conv, nfilter, c_merge5_0)

    # Block 5 - up: context: 8.0 (64, 64)
    c_up5 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(c_conv5_0_u)
    c_merge5 = Concatenate()([c_conv3_d, c_up5])
    c_conv5_u = ConvBlock(n_conv, nfilter, c_merge5)
    c_conv5_u_crop = Cropping2D(cropping=((24, 24)))(c_conv5_u)

    # Block 6 - up: context: 4.0 (128, 128)
    nfilter /= growth
    c_up6 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(c_conv5_u)
    c_merge6 = Concatenate()([c_conv2_d, c_up6])
    c_conv6_u = ConvBlock(n_conv, nfilter, c_merge6)

    # Block 7 - up: context: 2.0 (256, 256)
    # up7 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv6_u)
    nfilter /= growth
    c_up7 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(c_conv6_u)
    c_merge7 = Concatenate()([c_conv1_d, c_up7])
    c_conv7_u = ConvBlock(n_conv, nfilter, c_merge7)
    # c_conv7_u_crop = Cropping2D(cropping=((120, 120)))(c_conv7_u)


    #### target branch

    # Block 1 - down: target: 0.50 (256, 256)
    nfilter = nfilter0
    t_conv1_d = ConvBlock(n_conv, nfilter, target_inputlayer)
    t_pool1_d = MaxPooling2D(pool_size=(2, 2))(t_conv1_d)

    # Block 2 - down: target: 1.0 (128, 128)
    nfilter *= growth
    t_conv2_d = ConvBlock(n_conv, nfilter, t_pool1_d)
    t_pool2_d = MaxPooling2D(pool_size=(2, 2))(t_conv2_d)

    # Block 3 - down: target: 2.0 (64, 64)
    nfilter *= growth
    t_conv3_d = ConvBlock(n_conv, nfilter, t_pool2_d)
    t_pool3_d = MaxPooling2D(pool_size=(2, 2))(t_conv3_d)

    # Block 3_1 - down: target: 4.0 (32, 32)
    t_conv3_1_d = ConvBlock(n_conv, nfilter, t_pool3_d)
    t_pool3_1_d = MaxPooling2D(pool_size=(2, 2))(t_conv3_1_d)

    # Block 4: target: 8.0 (16, 16)
    # nfilter *= growth
    t_conv4_d = ConvBlock(n_conv, nfilter, t_pool3_1_d)
    t_merge4_d = Concatenate()([t_conv4_d, c_conv5_u_crop])  # c_conv7_u_crop
    t_conv4_d = ConvBlock(n_conv, nfilter, t_merge4_d)

    # Block 5_0 - up: target: 4.0 (32, 32)
    t_up5_0 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(t_conv4_d)
    t_merge5_0 = Concatenate()([t_conv3_1_d, t_up5_0])
    t_conv5_0_u = ConvBlock(n_conv, nfilter, t_merge5_0)

    # Block 5 - up: target: 2.0 (64, 64)
    t_up5 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(t_conv5_0_u)
    t_merge5 = Concatenate()([t_conv3_d, t_up5])
    t_conv5_u = ConvBlock(n_conv, nfilter, t_merge5)

    # Block 6 - up: target: 1.0 (128, 128)
    nfilter /= growth
    t_up6 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(t_conv5_u)
    t_merge6 = Concatenate()([t_conv2_d, t_up6])
    t_conv6_u = ConvBlock(n_conv, nfilter, t_merge6)

    # Block 7 - up: target: 0.5 (256, 256)
    nfilter /= growth
    t_up7 = Conv2DTranspose(nfilter, kernel_size=(2, 2), strides=(2, 2))(t_conv6_u)
    t_merge7 = Concatenate()([t_conv1_d, t_up7])
    t_conv7_u = ConvBlock(n_conv, nfilter, t_merge7)

    context_outputlayer = Convolution2D(nClass, 1, activation='softmax')(c_conv7_u)  # name="out_context"
    target_outputlayer = Convolution2D(nClass, 1, activation='softmax')(t_conv7_u)  # name='out_target'

    context_outputlayer = Reshape((context_input[0] * context_input[1], nClass), name="out_context")(context_outputlayer)  # to be able to use sample_weights
    target_outputlayer = Reshape((target_input[0] * target_input[1], nClass), name="out_target")(target_outputlayer)  # to be able to use sample_weights

    context_target_model = Model(inputs=[context_inputlayer, target_inputlayer],
                                 outputs=[context_outputlayer, target_outputlayer])

    context_target_model.summary(line_length=124)

    return context_target_model




