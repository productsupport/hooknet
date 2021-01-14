import os
import numpy as np
import h5py
from tensorflow import keras

from skimage.transform import resize, rotate
from skimage.filters import gaussian
from skimage.color import rgb2hed, hed2rgb, hsv2rgb, rgb2hsv

from skimage.color import rgb2gray

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, list_IDs, data_folder, batch_size=32, dim=(256, 256),
                 n_channels=3, n_classes=11, shuffle=True):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_folder = data_folder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y, sample_weights  = self.__data_generation(list_IDs_temp)

        return X, Y, sample_weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # print(i, ID)
            
            # Store sample
            with h5py.File(os.path.join(self.data_folder, ID), 'r') as f:
                patch = f['patches_20x']['patch'][:]
                seg = f['patches_20x']['segmentation'][:]
                patch = patch.astype('float') / 255

                X[i, ] = patch  # f['patches_20x']['patch'][:]
                Y[i, ] = seg  # f['patches_20x']['segmentation'][:]

#         # combine first and second dimentsions
#         X = X.reshape((-1, ) + (*self.dim, self.n_channels))
#         Y = X.reshape((-1, ) + (*self.dim, self.n_channels))
#         self.num_samples = len(X)

        # sample weights: approach 1: 0 for unknown class and 1 for the rest
        sample_weights = np.ones_like(Y)
        sample_weights = sample_weights.astype(dtype='float32')
        sample_weights[np.where(Y == 0)] = 0.0

        # sample weights: approach 2
        # class_weight_norm_dict = self.__sample_weights(Y)
        #
        # sample_weights = np.zeros_like(Y)
        # for key in class_weight_norm_dict.keys():
        #     sample_weights[np.where(Y == np.float(key))] = class_weight_norm_dict[key]


        Y = keras.utils.to_categorical(Y, num_classes=self.n_classes)

        sample_weights = np.reshape(
            sample_weights, (sample_weights.shape[0],
                             sample_weights.shape[1] * sample_weights.shape[2]))

        # reshape to be able to use sample_weights
        Y = np.reshape(Y, (Y.shape[0], Y.shape[1] * Y.shape[2], -1))

        return X, Y, sample_weights

    def __sample_weights(self, y):
        'Calculate class weights per batch'

        unique, counts = np.unique(y, return_counts=True)
        dict_count = dict(zip(unique, counts))
        print(dict_count)
        dict_count.pop(0, None)

        # if a class is not there, add count 0 for that class ???
        for i in range(1, self.n_classes):
            if i not in dict_count.keys():
                dict_count[i] = 0

        print(dict_count)

        counts_sum = 0
        for i in dict_count.keys():
            counts_sum += dict_count[i]

        print("counts_sum: ", counts_sum)

        # calculate weights
        class_weight_dict = {}
        class_weight_dict_2 = {}
        for key in dict_count.keys():
            class_weight_dict[key] = 1.0 - dict_count[key] / counts_sum
            class_weight_dict_2[key] = dict_count[key] / counts_sum
        print("class_weight_dict: ", class_weight_dict)
        print("class_weight_dict_2: ", class_weight_dict_2)


        weights_sum = 0
        for i in class_weight_dict.keys():
            weights_sum += class_weight_dict[i]

        weights_sum_2 = 0
        for i in class_weight_dict_2.keys():
            weights_sum_2 += class_weight_dict_2[i]

        print("weights_sum: ", weights_sum)
        print("weights_sum_2: ", weights_sum_2)

        class_weight_norm_dict = {}
        for key in class_weight_dict.keys():
            class_weight_norm_dict[key] = class_weight_dict[key] / weights_sum

        class_weight_norm_dict_2 = {}
        for key in class_weight_dict_2.keys():
            class_weight_norm_dict_2[key] = class_weight_dict_2[key] / weights_sum_2

        print("class_weight_norm_dict: ", class_weight_norm_dict)
        print("class_weight_norm_dict_2: ", class_weight_norm_dict_2)

        return class_weight_norm_dict


class DataGenerator_metaData(keras.utils.Sequence):
    'Generates data for Keras. without using sample weights. using metaData'

    def __init__(self, metaData, data_folder, batch_size=10, dim=(256, 256),
                 n_channels=3, n_classes=10, shuffle=True, blur_augment=0, colour_augment=True,
                 flip_augment=True, colour_scheme='hed'):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.metaData = metaData
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_folder = data_folder
        self.on_epoch_end()

        # Basic augmentation settings
        self.blur = blur_augment
        self.colour_augment = colour_augment
        self.colour_scheme = colour_scheme
        self.flip = flip_augment

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        list_IDs_temp = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        classmeta_temp = self.metaData.groupby(["class_id"]).sample(n=1000, replace=True)  # random_state=1
        self.indexes = list(classmeta_temp.patch_name.values)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            with h5py.File(os.path.join(self.data_folder, ID), 'r') as f:
                patch = f['patches_20x']['patch'][:]
                seg = f['patches_20x']['segmentation'][:]
                patch = patch.astype('float') / 255

                if self.blur > 0:
                    sigma = np.random.rand(1) * (1 + self.blur)
                    patch = gaussian(patch, sigma=sigma, multichannel=True)

                if self.colour_augment:
                    if self.colour_scheme == 'hsv':
                        patch = np.clip(patch, -1, 1)
                        sample_hue = (np.random.rand(1) - 0.5) * 1
                        sample_saturation = (np.random.rand(1) - 0.5) * 1
                        hsv = rgb2hsv(patch)
                        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + sample_hue, 0, 1)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sample_saturation, 0, 1)
                        patch = hsv2rgb(hsv)
                        patch = np.clip(patch, 0, 1.0).astype(np.float32)
                    elif self.colour_scheme == 'hed':
                        ah = 0.95 + np.random.random() * 0.1
                        bh = -0.05 + np.random.random() * 0.1
                        ae = 0.95 + np.random.random() * 0.1
                        be = -0.05 + np.random.random() * 0.1
                        patch = np.clip(patch, -1, 1)
                        hed = rgb2hed(patch)
                        hed[:, :, 0] = ah * hed[:, :, 0] + bh
                        hed[:, :, 1] = ae * hed[:, :, 1] + be
                        patch = hed2rgb(hed)
                        patch = np.clip(patch, 0, 1.0).astype(np.float32)

                if self.flip:
                    # if whatflip == 5, nothing happens
                    whatflip = np.random.randint(6)
                    if whatflip == 0:  # Rotate 90
                        patch = rotate(patch, 90)
                        seg = rotate(seg, 90)
                    elif whatflip == 1:  # Rotate 180
                        patch = rotate(patch, 180)
                        seg = rotate(seg, 180)
                    elif whatflip == 2:  # Rotate 270
                        patch = rotate(patch, 270)
                        seg = rotate(seg, 270)
                    elif whatflip == 3:  # Flip left-right
                        patch = patch[:, -1::-1, :]
                        seg = seg[:, -1::-1]
                    elif whatflip == 4:  # Flip up-down
                        patch = patch[-1::-1, :, :]
                        seg = seg[-1::-1, :]

                X[i,] = patch  # f['patches_20x']['patch'][:]
                Y[i,] = seg  # f['patches_20x']['segmentation'][:]

        Y = keras.utils.to_categorical(Y, num_classes=self.n_classes)

        return X, Y


    