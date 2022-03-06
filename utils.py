import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomContrast, RandomFlip, RandomTranslation, RandomZoom
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras import Sequential
import random
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform, RicianNoiseTransform
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
import math
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import roc_curve, auc


def margin_scheduler(a, b, t, max_t, mode='exp', step=None):
    '''
    Margin scheduler for gradual increment of margin exponentially.
    @param a: lower bound margin
    @param b: upper bound margin
    @param t: time step/epochs
    @param max_t: maximum time step
    @param mode: 'exp' for exponential function or 'step' for step function
    @param step: Step interval for step function
    @return: scheduled margin
    '''
    assert mode=='exp' or mode=='step', "Please input mode:'exp' for exponential growth or mode: 'step' for step growth!"
    # exponential function
    if mode=='exp':
        # growth-rate
        r = (1/max_t)*np.log(b/a)
        # output function
        margin = a*np.exp(r*t)
    # step function
    if mode=='step':
        assert step is not None, "Step size must be defined!"
        t_range = np.linspace(0, max_t, num=step+1, dtype=int)
        margin_range = np.linspace(a, b, num=step)
        for i in range(step):
            if t>=t_range[i]:
                margin = margin_range[i]
    return margin


def data_aug(imgs, choice=None):
    '''
    Random data augmentation with random choice
    @param imgs: input imgs
    @param choice: "None" as default for random sampling, else specify choice
    @return: data augmentated images
    '''
    if choice is None:
        choice = np.random.randint(0,7)
    # no augmentation
    if choice==0:
        x = imgs
    # flip up and down
    if choice==1:
        x = RandomRotation(factor=0.25)(imgs)
    # flip left and right
    if choice==2:
        x = RandomContrast(factor=0.5)(imgs)
    # rotation based on angle
    if choice==3:
        x = RandomFlip()(imgs)
    if choice==4:
        x = RandomTranslation(height_factor=0.2, width_factor=0.2)(imgs)
    if choice==5:
        x = RandomZoom(height_factor=0.2, width_factor=0.2)(imgs)
    if choice==6:
        x= GaussianNoise(0.1)(imgs)
    return x


def augment_layers(x):
    '''
    Random data augmentation with sequential layer
    @param x: input batch images of at least rank 4
    @return: augmented input x
    '''
    aug_model = Sequential([
        GaussianNoise(0.05),
        RandomRotation(factor=0.25),
        RandomContrast(factor=0.5),
        RandomFlip(),
        RandomTranslation(height_factor=0.2, width_factor=0.2),
        RandomZoom(height_factor=0.2, width_factor=0.2)
    ])
    return aug_model(x)


def z_score_norm(modality):
    """
    Removes 1% of the top and bottom intensities and perform
    normalization on the input 2D slice.
    """
    b = np.percentile(modality, 99)
    t = np.percentile(modality, 1)
    modality = np.clip(modality, t, b)
    if np.std(modality) == 0:
        return modality
    else:
        modality = (modality - np.mean(modality)) / np.std(modality)
        return modality


def nn_array2list(nn_array, idx2img_array, nrows, k=1):
    '''
    Convert nearest neighbour array to image list for grid plot

    nn_array: Nearest neightbour array in 2D
    idx2img_array: An array that maps idices to corresponding image
    k: k-th number of nearest neighbor

    return: list of images corresponding to the indices of nearest neighbours
    '''
    assert nn_array.ndim <= 2, "Only 2D array are accepted!"
    img_list = []
    # loop through each row of the nn_array
    for i in range(nrows):
        img_idx_row = nn_array[i]
        for idx in img_idx_row[:k+1]:
            img_list.append(idx2img_array[idx])
    return img_list


def hstack(input):
    """
    Recursive loops for stacking mapping function
    @param input:
    @return: stacked tensors from generation loop
    """
    for i,(x,y) in enumerate(input):
        #intantiate for first step
        if i==0:
            y_0 = np.zeros_like(x)
            x_0 = np.zeros_like(y)
        y_0 = np.vstack((y_0, y))
        x_0 = np.vstack((x_0, x))
    # remove batch number
    y_stack = y_0[y.shape[0]:]
    x_stack = x_0[x.shape[0]:]
    return x_stack, y_stack


class BalancedDataGenerator:
    '''
    Generator for balanced data set use for deep metric learning.
    For each class we sample based on sample_per_class.
    '''
    def __init__(self, sample_per_class: int):
        '''
        @param sample_per_class: the number of samples per class
        '''
        self.sample_per_class = sample_per_class

    def __call__(self, x: tf.float32, y: tf.int32):
        '''
        Call function for the generator
        @param x: features
        @param y: labels
        @return: generator that generates batch size (class*number_per_class) with even distribution
        '''
        # to ensure that the input dimension is 1 for label dataset
        y = np.squeeze(y)
        # array of non-duplicate class in the label dataset
        class_ = np.unique(y)
        # number of classes
        n_class = len(class_)
        while True:
            # initiate to store sample class index for each classes
            total_sample_idx = []
            # loop through the classes
            for i in range(n_class):
                # collect the index where label is class
                idx = tf.where(y == class_[i])
                # shuffle the index
                idx = tf.random.shuffle(idx)
                # sample the number of classes from the index list where label is class
                sample_idx = np.squeeze(random.sample(idx.numpy().tolist(), self.sample_per_class))
                total_sample_idx.append(sample_idx)
            # batch size is define as the (number of class * sample per class)
            # the size of total_sample_idx is essentially the number of batch
            # since we gather index based on sample_per_class for each class
            total_sample_idx = np.ndarray.flatten(np.array(total_sample_idx)) # [batch size, ]
            total_sample_idx = tf.random.shuffle(total_sample_idx)
            # map index to parameter
            y_batch = tf.gather(y, total_sample_idx) # [batch size, ]
            x_batch = tf.gather(x, total_sample_idx) # [batch size, *x.shape[1:]]
            yield x_batch, tf.cast(y_batch, tf.int32)


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, max_epochs, decay_rate=0.9):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs

    def __call__(self, epochs):
        return self.initial_learning_rate * self.decay_rate ^ (epochs / self.max_epochs)


class LRDecay:
    def __init__(self, initial_learning_rate, max_epochs, decay_rate=0.1):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs

    def exponentialDecay(self, epochs):
        return self.initial_learning_rate * math.exp(-self.decay_rate * epochs)

    def polynomialDecay(self, epochs, power):
        decay = (1 - (epochs / float(self.max_epochs))) ** power
        lr = self.initial_learning_rate * decay
        # return the new learning rate
        return float(lr)

    def stepDecay(self, epochs, drop_every):
        exp = np.floor((1 + epochs) / drop_every)
        lr = self.initial_learning_rate * (self.decay_rate ** exp)
        return float(lr)

    def stepWise(self, epochs, epochs_list, learning_rates):
        for i in range(len(epochs_list)):
            if epochs < epochs_list[i]:
                lr = learning_rates[i]
                break
            if epochs >= epochs_list[-1]:
                lr = learning_rates[-1]
                break
        return float(lr)

    def __call__(self, epochs, mode="exp", power=2.0, drop_every=20,
                 epochs_list=None, learning_rates=None):
        if mode == "exp":
            return self.exponentialDecay(epochs)
        if mode == "poly":
            return self.polynomialDecay(epochs, power)
        if mode == "step":
            return self.stepDecay(epochs, drop_every)
        if mode == "step_wise":
            return self.stepWise(epochs, epochs_list, learning_rates)


def get_split_fold(data, label_col='labels', fold_col='fold', val_fold=0, test_fold=-1):
    """
    If the data set is already split according to folds with indices [-1, 0, 1, 2, 3]
    @param data: csv file where the data sets are stored
    @param label_col: label column name
    @param fold_col: there can be multi fold columns name, by default "fold"
    @param val_fold: validation fold to be popped
    @param test_fold: test fold to be popped
    @return: dictionaries of train, val, train dictionary
    """
    # return indices of fold data
    test_idx = np.where(data[fold_col]==test_fold)[0]
    val_idx = np.where(data[fold_col]==val_fold)[0]
    train_idx = np.where((data[fold_col] != test_fold) & (data[fold_col] != val_fold))[0]

    # create dictionary for each data set
    train_ds = {'npy_path':[data['npy_path'].tolist()[i] for i in train_idx],
                'labels':[data[label_col].tolist()[i] for i in train_idx],
                'patient_id':[data['patient_id'].tolist()[i] for i in train_idx]}
    val_ds = {'npy_path':[data['npy_path'].tolist()[i] for i in val_idx],
                'labels':[data[label_col].tolist()[i] for i in val_idx],
                'patient_id':[data['patient_id'].tolist()[i] for i in val_idx]}
    test_ds = {'npy_path':[data['npy_path'].tolist()[i] for i in test_idx],
                'labels':[data[label_col].tolist()[i] for i in test_idx],
                'patient_id':[data['patient_id'].tolist()[i] for i in test_idx]}

    return {'train_ds':train_ds, 'val_ds': val_ds, 'test_ds':test_ds}


class DataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded,
                 crop_status=True, crop_type="center",
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True,
                 infinite=True, margins=(0,0,0)):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        # original patch size with [slices, width, height]
        self.patch_size = patch_size
        self.num_modalities = 3
        self.indices = list(range(len(data['labels'])))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.margins = margins

    @staticmethod
    def load_patient(img_path):
        img = np.load(img_path, mmap_mode="r")
        return img

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['npy_path'][i] for i in idx]
        patients_labels = [self._data['labels'][i] for i in idx]
        patient_id = [self._data['patient_id'][i] for i in idx]
        # initialize empty array for data and seg
        img = np.zeros((len(patients_for_batch), self.num_modalities, *self.patch_size), dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data = self.load_patient(j)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            # swap axes for crop function, (z, x, y, m) => (m, x, y, z)
            patient_data = np.swapaxes(patient_data,0,-1)
            if self.crop_status:
                patient_data = crop(patient_data[None], seg=None, crop_size=self.patch_size,
                                    margins=self.margins, crop_type=self.crop_type)
            img[i] = patient_data[0]
        # img = np.swapaxes(img, 1, -1)
        patients_labels = np.array(patients_labels)
        return {'data': img, 'lbl': patients_labels, 'patient_id': patient_id}


class BalancedDataLoader3D(DataLoader):
    def __init__(self, data, sample_size, sample_per_class, patch_size,
                 num_threads_in_multithreaded, crop_status=True, crop_type="center", seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, margins=(0,0,0)):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, sample_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        assert sample_size<=len(data['labels']), "Sample size should be less than data set size!"
        self.patch_size = patch_size
        self.num_modalities = 3
        self.indices = list(range(len(data['labels'])))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.sample_per_class = sample_per_class
        self.n_class = np.unique(self._data['labels'])
        self.margins = margins

    @staticmethod
    def load_patient(img_path):
        img = np.load(img_path, mmap_mode="r")
        return img
      # need to fix!
    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        balanced_class_indx = []
        for c in self.n_class:
            class_sample = 0
            while class_sample<=self.sample_per_class:
                idx = self.get_indices()
                # retrieve the patients label
                patients_labels = [self._data['labels'][i] for i in idx]
                # evaluate the position of the patients where the label match the class
                class_indx = np.where(np.array(patients_labels)==c)[0]
                # map back to the indices of the patient labels where it match
                class2labels = [idx[j] for j in class_indx]
                # check if the length match
                class_sample = len(class2labels)
                if class_sample>self.sample_per_class:
                    class2labels = np.random.choice(class2labels, self.sample_per_class, replace=False).tolist()
            balanced_class_indx.append(class2labels)
        # flatten list of list
        bci = sum(balanced_class_indx, [])
        # shuffle the list
        random.shuffle(bci)
        balanced_patients_labels = [self._data['labels'][i] for i in bci]
        balanced_patients_for_batch = [self._data['npy_path'][i] for i in bci]
        balanced_patient_id = [self._data['patient_id'][i] for i in bci]
        # initialize empty array for data and seg
        img = np.zeros((self.sample_per_class*len(self.n_class), self.num_modalities, *self.patch_size), dtype=np.float32)
        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(balanced_patients_for_batch):
            patient_data = self.load_patient(j)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            # swap axes for crop function, (z, x, y, m) => (m, x, y, z)
            patient_data = np.swapaxes(patient_data,0,-1)
            if self.crop_status:
                patient_data = crop(patient_data[None], seg=None, crop_size=self.patch_size,
                                    margins=self.margins, crop_type=self.crop_type)
            img[i] = patient_data[0]
        # img = np.swapaxes(img, 1, -1)
        patients_labels = np.array(balanced_patients_labels)
        return {'data': img, 'lbl': patients_labels, 'patient_id': balanced_patient_id}


class ValidationDataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, crop=True, seed_for_shuffle=1234,
                 return_incomplete=True, shuffle=True, infinite=False):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        # original patch size with [slices, width, height]
        self.patch_size = patch_size
        self.num_modalities = 3
        self.indices = list(range(len(data['labels'])))
        self.crop = crop

    @staticmethod
    def load_patient(img_path):
        img = np.load(img_path, mmap_mode="r")
        return img

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['npy_path'][i] for i in idx]
        patients_labels = [self._data['labels'][i] for i in idx]
        patient_id = [self._data['patient_id'][i] for i in idx]
        # initialize empty array for data and seg
        img = np.zeros((len(patients_for_batch), self.num_modalities, *self.patch_size), dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data = self.load_patient(j)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            # swap axes for crop function, (z, x, y, m) => (m, x, y, z)
            patient_data = np.swapaxes(patient_data,0,-1)
            if crop:
                patient_data = crop(patient_data[None], seg=None, crop_size=self.patch_size, crop_type="center")
            img[i] = patient_data[0]
        # img = np.swapaxes(img, 1, -1)
        patients_labels = np.array(patients_labels)
        return {'data': img, 'lbl': patients_labels, 'patient_id': patient_id}



def get_train_transform(patch_size, prob=0.5):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform(
            patch_size,
            [i // 2 for i in patch_size],
            do_elastic_deform=True,
            alpha=(0., 300.),
            sigma=(20., 40.),
            do_rotation=True,
            angle_x=(0.,0.),
            angle_y=(0.,0.),
            angle_z=(-np.pi/15., np.pi/15.),
            do_scale=True,
            scale=(1/1.15, 1.15),
            random_crop=False,
            border_mode_data='constant',
            border_cval_data=0,
            order_data=3,
            p_el_per_sample=prob, p_rot_per_sample=prob, p_scale_per_sample=prob
        )
    )

    # now we mirror along the y-axis
    tr_transforms.append(MirrorTransform(axes=(1,)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=prob))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=prob))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.5), p_per_sample=prob))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 2.0), different_sigma_per_channel=True,
                                               p_per_channel=prob, p_per_sample=prob))
    tr_transforms.append(ContrastAugmentationTransform(contrast_range=(0.75, 1.25), p_per_sample=prob))
    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


class CustomModelCheckpoint:
    def __init__(self, filepath, model,
                patience=20, save_best_only=True):
        '''
        Custom Model Checkpoint to store previous metrics
        with a length of patience and compared in every epochs
        to save the best model or save model for every run
        @param filepath: where metrics should be stored
        @param model: tensorflow model
        @param run: refer to wandb: https://docs.wandb.ai/guides/track/advanced/save-restore
                run = wandb.init(...)
                assesible to run.name, run.dir...
        @param patience:
        @param save_best_only:
        '''
        self.model = model
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.patience = patience

    def reset(self):
        """
        Delete existing filepath where the
        monitor value is being stored
        """
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def on_epochs_end(self, monitor_val, mode, threshold=None, weights_dir=None,
                      model_weights_name='best_model.h5'):
        '''
        @param monitor_val: metrics to be monitored
        @param mode: {'min', 'max', 'min_threshold', 'max_threshold'}
            min: store the minimum value of the monitored metrics
            max: store the maximum value of the monitored metrics
            min_threshold: save weights if monitored metrics > min_threshold
            max_threshold: save weights if monitored metrics < max_threshold
        @param threshold: Specific threshold for 'min_threshold' and 'max_threshold'
        @param weights_dir: save weights in local directory
        @param model_weights_name: model weights in .h5 format
        '''
        if weights_dir!=None:
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

        if self.save_best_only:
            # goal=> minimize target metrics
            if mode=='min':
                # check if file path exist
                if not os.path.isfile(self.filepath):
                    filepath_dir = os.path.split(self.filepath)[0]
                    if not os.path.exists(filepath_dir):
                        os.umask(0)
                        os.makedirs(filepath_dir)
                    init_val = np.ones(self.patience)*np.inf
                    np.save(self.filepath, init_val)
                    store_val = np.load(self.filepath)
                else:
                    store_val = np.load(self.filepath)
                current_max = np.max(store_val)
                current_max_idx = np.argmax(store_val)
                if monitor_val<current_max:
                    store_val[current_max_idx] = monitor_val
                    self.model.save_weights(os.path.join(weights_dir, model_weights_name))
                    np.save(self.filepath, store_val)

            # goal=> maximize target metrics
            if mode=='max':
                if not os.path.isfile(self.filepath):
                    filepath_dir = os.path.split(self.filepath)[0]
                    if not os.path.exists(filepath_dir):
                        os.umask(0)
                        os.makedirs(filepath_dir)
                    init_val = np.zeros(self.patience)
                    np.save(self.filepath, init_val)
                    store_val = np.load(self.filepath)
                else:
                    store_val = np.load(self.filepath)
                current_min = np.min(store_val)
                current_min_idx = np.argmin(store_val)
                if monitor_val>current_min:
                    store_val[current_min_idx] = monitor_val
                    self.model.save_weights(os.path.join(weights_dir, model_weights_name))
                    np.save(self.filepath, store_val)

            # goal=> exceed min_threshold
            if mode=='min_threshold':
                assert threshold!=None, "please input the value for parameter: 'threshold'!"
                if monitor_val>threshold:
                    self.model.save_weights(os.path.join(weights_dir, 'best_model.h5'))

            # goal=> does not exceed max_threshold
            if mode=='max_threshold':
                assert threshold!=None, "please input the value for parameter: 'threshold'!"
                if monitor_val<threshold:
                    self.model.save_weights(os.path.join(weights_dir, model_weights_name))
        else:
            self.model.save_weights(os.path.join(weights_dir, model_weights_name))


class CustomModelCheckpointBool:
    def __init__(self, filepath,
                patience=5, save_best_only=True):
        '''
        Custom Model Checkpoint to store previous metrics
        with a length of patience and compared in every epochs
        to save the best model or save model for every run by returning
        boolean as a gating outcome
        @param filepath: where metrics should be stored
        @param model: tensorflow model
        @param run: refer to wandb: https://docs.wandb.ai/guides/track/advanced/save-restore
                run = wandb.init(...)
                assesible to run.name, run.dir...
        @param patience:
        @param save_best_only:
        '''
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.patience = patience

    def reset(self):
        """
        Delete existing filepath where the
        monitor value is being stored
        """
        os.remove(self.filepath)

    def on_epochs_end(self, monitor_val, mode, threshold=None):
        '''
        @param monitor_val: metrics to be monitored
        @param mode: {'min', 'max', 'min_threshold', 'max_threshold'}
            min: store the minimum value of the monitored metrics
            max: store the maximum value of the monitored metrics
            min_threshold: save weights if monitored metrics > min_threshold
            max_threshold: save weights if monitored metrics < max_threshold
        @param threshold: Specific threshold for 'min_threshold' and 'max_threshold'
        '''

        if self.save_best_only:
            # goal=> minimize target metrics
            if mode=='min':
                # check if file path exist
                if not os.path.isfile(self.filepath):
                    filepath_dir = os.path.split(self.filepath)[0]
                    if not os.path.exists(filepath_dir):
                        os.umask(0)
                        os.makedirs(filepath_dir)
                    init_val = np.ones(self.patience)*np.inf
                    np.save(self.filepath, init_val)
                    store_val = np.load(self.filepath)
                else:
                    store_val = np.load(self.filepath)
                current_max = np.max(store_val)
                current_max_idx = np.argmax(store_val)
                if monitor_val<current_max:
                    store_val[current_max_idx] = monitor_val
                    np.save(self.filepath, store_val)
                    return True
                else:
                    return False
            # goal=> maximize target metrics
            if mode=='max':
                if not os.path.isfile(self.filepath):
                    filepath_dir = os.path.split(self.filepath)[0]
                    if not os.path.exists(filepath_dir):
                        os.umask(0)
                        os.makedirs(filepath_dir)
                    init_val = np.zeros(self.patience)
                    np.save(self.filepath, init_val)
                    store_val = np.load(self.filepath)
                else:
                    store_val = np.load(self.filepath)
                current_min = np.min(store_val)
                current_min_idx = np.argmin(store_val)
                if monitor_val>current_min:
                    store_val[current_min_idx] = monitor_val
                    np.save(self.filepath, store_val)
                    return True
                else:
                    return False

            # goal=> exceed min_threshold
            if mode=='min_threshold':
                assert threshold!=None, "please input the value for parameter: 'threshold'!"
                if monitor_val>threshold:
                    return True
                else:
                    return False

            # goal=> does not exceed max_threshold
            if mode=='max_threshold':
                assert threshold!=None, "please input the value for parameter: 'threshold'!"
                if monitor_val<threshold:
                    return True
                else:
                    return False


def min_max_norm(img, dwh=(1,2,3)):
    """
    Min max normalization for 3D images with input [batch size, slices, width, height, channel]
    @param img: Input image of 5D array
    @return: Min max norm of the image per slice
    """
    inp_shape = img.shape
    img_min = np.broadcast_to(img.min(axis=dwh, keepdims=True), inp_shape)
    img_max = np.broadcast_to(img.max(axis=dwh, keepdims=True), inp_shape)
    x = (img-img_min)/(img_max-img_min+float(1e-18))
    return x


def clip_fn(img, percentile, axis, keepdims):
    '''
    Clipping the values for each modalities
    '''
    t = np.percentile(img, percentile, axis, keepdims=keepdims)
    b = np.percentile(img, (100-percentile), axis, keepdims=keepdims)
    clip_img = np.clip(img, b, t)
    return clip_img


def standardization(img, axis=(0,1,2), percentile=99):
    clip_img = clip_fn(img, percentile, axis, True)
    m = clip_img.mean(axis=axis, keepdims=True)
    s = clip_img.std(axis=axis, keepdims=True)
    m_broadcast = np.broadcast_to(m, shape=clip_img.shape)
    s_broadcast = np.broadcast_to(s, shape=clip_img.shape)

    z_norm = (clip_img-m_broadcast)/(s_broadcast+float(1e-18))
    return z_norm



def acc_per_class(y_true, y_pred):
    n_class = len(np.unique(y_true))
    total_count = np.zeros(n_class)
    hit_count = np.zeros(n_class)
    for c in range(n_class):
        for i in range(len(y_true)):
            if y_true[i]==c:
                total_count[c]+=1
                hit = tf.cast(tf.equal(y_pred[i],y_true[i]), dtype=tf.float32)
                hit_count[c]+=hit

    return hit_count/(total_count + float(1e-8))


def acc_per_class_v2(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    n_class = len(np.unique(y_true))
    class_acc = np.zeros(n_class)
    for c in range(n_class):
        # class masking
        y_true_class_idx = np.where(y_true==c)[0]
        y_true_class = y_true[y_true_class_idx]
        y_pred_class = y_pred[y_true_class_idx]
        # y_true_class = np.take(y_true, y_true_class_idx)
        # y_pred_class = np.take(y_pred, y_true_class_idx)
        TP = np.sum(np.equal(y_pred_class, y_true_class).astype(float))
        total = y_true_class_idx.shape[0]
        class_acc[c] = TP/(total+ float(1e-8))
    return class_acc

def new_fold_col(csv_path, col_name, label_col="labels", n_split=5, seed=5243, overwrite=True, new_csv_name=None):
    '''
    Create new column for fold distribution
    @param csv_path: the path to the csv file where image is stored
    @param col_name: new column name to be replaced or to be created
    @param label_col: label column name
    @param seed: seeding for random status
    @param overwrite: overwrite original file by default
    @param new_csv_name: if overwrite is False, appoint new path for csv to be stored
    @return: new csv file
    '''
    tabular_data = pd.read_csv(csv_path)
    lbl_list = tabular_data[label_col].tolist()
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
    # reference index
    idx_ = np.arange(len(lbl_list))
    # superimpose the masking
    fold_ = np.zeros(len(lbl_list))
    for i, (_, split_idx) in enumerate(skf.split(X=np.zeros_like(lbl_list), y=lbl_list)):
        # gather index where the indices match the split fold indices
        mask = np.in1d(idx_, split_idx)
        # assign number to the fold
        fold_+=mask*(i-1)
        # idx2lbl = np.take(lbl_list, split_idx)
        # val, counts = np.unique(idx2lbl, return_counts=True)
        # print(val)
        # print(counts)
        # print()
    # val, counts = np.unique(fold_, return_counts=True)
    # print(val)
    # print(counts)
    tabular_data[col_name] = [int(f) for f in fold_]
    if overwrite:
        tabular_data.to_csv(csv_path, index=False)
    else:
        tabular_data.to_csv(new_csv_name, index=False)


class BalancedDataLoader3Dv2(DataLoader):
    def __init__(self, data, sample_size, sample_per_class, patch_size,
                 num_threads_in_multithreaded, crop_status=True, crop_type="center", seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, margins=(0,0,0)):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, sample_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        assert sample_size<=len(data['labels']), "Sample size should be less than data set size!"
        self.patch_size = patch_size
        self.num_modalities = 3
        self.indices = list(range(len(data['labels'])))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.sample_per_class = sample_per_class
        self.n_class = np.unique(self._data['labels'])
        self.margins = margins

    @staticmethod
    def load_patient(img_path):
        img = np.load(img_path, mmap_mode="r")
        return img
      # need to fix!
    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        balanced_class_indx = []
        for c in self.n_class[:-1]:
            class_sample = 0
            while class_sample<=self.sample_per_class:
                idx = self.get_indices()
                # retrieve the patients label
                patients_labels = [self._data['labels'][i] for i in idx]
                # evaluate the position of the patients where the label match the class
                class_indx = np.where(np.array(patients_labels)==c)[0]
                # map back to the indices of the patient labels where it match
                class2labels = [idx[j] for j in class_indx]
                # check if the length match
                class_sample = len(class2labels)
                if class_sample>self.sample_per_class:
                    class2labels = np.random.choice(class2labels, self.sample_per_class, replace=False).tolist()
            balanced_class_indx.append(class2labels)

        # over sampling unbalanced class
        unbalanced_class_idx = np.where(np.array(self._data['labels'])==self.n_class[-1])[0]
        over_sampling = np.random.choice(unbalanced_class_idx, self.sample_per_class, replace=False).tolist()
        balanced_class_indx.append(over_sampling)
        # flatten list of list
        bci = sum(balanced_class_indx, [])
        # shuffle the list
        random.shuffle(bci)
        balanced_patients_labels = [self._data['labels'][i] for i in bci]
        balanced_patients_for_batch = [self._data['npy_path'][i] for i in bci]
        balanced_patient_id = [self._data['patient_id'][i] for i in bci]
        # initialize empty array for data and seg
        img = np.zeros((self.sample_per_class*len(self.n_class), self.num_modalities, *self.patch_size), dtype=np.float32)
        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(balanced_patients_for_batch):
            patient_data = self.load_patient(j)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            # swap axes for crop function, (z, x, y, m) => (m, x, y, z)
            patient_data = np.swapaxes(patient_data,0,-1)
            if self.crop_status:
                patient_data = crop(patient_data[None], seg=None, crop_size=self.patch_size,
                                    margins=self.margins, crop_type=self.crop_type)
            img[i] = patient_data[0]
        # img = np.swapaxes(img, 1, -1)
        patients_labels = np.array(balanced_patients_labels)
        return {'data': img, 'lbl': patients_labels, 'patient_id': balanced_patient_id}


class CallBacks:
    def __init__(self, mode='min'):
        self.mode = mode
        self.counter = 0
        self.current_val = self.init_status()

    def reset(self):
        self.counter=0

    def init_status(self):
        if self.mode=='min':
            init_val = 0
        else:
            init_val = float(1e5)
        return init_val

    def status(self, monitor, patience):
        if self.mode=='min':
            if monitor>self.current_val:
                self.counter+=1
                self.current_val=monitor
            else:
                self.reset()
        if self.mode=='max':
            if monitor<self.current_val:
                self.counter+=1
                self.current_val=monitor
            else:
                self.reset()
        if self.counter==patience:
            self.reset()
            return True
        else:
            return False

class ReduceLRonPlateau(CallBacks):
    def __init__(self, initial_lr, max_lr, factor, **kwargs):
        super().__init__(**kwargs)
        self.initial_lr = float(initial_lr)
        self.max_lr = float(max_lr)
        self.factor = factor

    def __call__(self, monitor, patience):
        if self.status(monitor, patience):
            if self.initial_lr>self.max_lr:
                self.initial_lr*=self.factor
            if self.initial_lr<=self.max_lr:
                self.initial_lr=self.max_lr
            return self.initial_lr
        else:
            return self.initial_lr


def lbl_2_onehot(label):
    one_hot = [[1,0,0],[0,1,0],[0,0,1],[1,0,1],[1,1,0],[0,1,1],[1,1,1]]
    map_ = [one_hot[i] for i in label]
    return np.array(map_)


def onehot_2_lbl(one_hot):
    def one_hot_map(one_hot):
        if np.array_equal(one_hot,[1,0,0]):
            return 0
        if np.array_equal(one_hot,[0,1,0]):
            return 1
        if np.array_equal(one_hot,[0,0,1]):
            return 2
        if np.array_equal(one_hot,[1,0,1]):
            return 3
        if np.array_equal(one_hot,[1,1,0]):
            return 4
        if np.array_equal(one_hot,[0,1,1]):
            return 5
        if np.array_equal(one_hot,[1,1,1]):
            return 6
    true_label = [one_hot_map(j) for j in one_hot]
    return np.array(true_label)


def rescale_intensity(image, thres=(1.0, 99.0), method='noclip'):
    '''
        Rescale the image intensity using several possible ways

        Parameters
        ----------
        image: array
            Image to rescale
        thresh: list of two floats between 0. and 1., default (1.0, 99.0)
            Percentiles to use for thresholding (depends on the `method`)
        method: str, one of ['clip', 'mean', 'median', 'noclip']
            'clip': clip intensities between the thresh[0]th and the thresh[1]th
            percentiles, and then scale between 0 and 1
            'mean': divide by mean intensity
            'meadin': divide by meadian intensity
            'noclip': Just like 'clip', but wihtout clipping the extremes

        Returns
        -------
        image: array
    '''
    eps = 0.000001

    def rescale_single_channel_image(image):
        # Deal with negative values first
        min_value = np.min(image)
        if min_value < 0:
            image -= min_value
        if method == 'clip':
            val_l, val_h = np.percentile(image, thres)
            image2 = image
            image2[image < val_l] = val_l
            image2[image > val_h] = val_h
            image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        elif method == 'mean':
            image2 = image / max(np.mean(image), 1)
        elif method == 'median':
            image2 = image / max(np.median(image), 1)
        elif method == 'noclip':
            val_l, val_h = np.percentile(image, thres)
            image2 = image
            image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        else:
            image2 = image
        return image2

    # Process each channel independently
    if len(image.shape) == 4:
        for i in range(image.shape[-1]):
            image[..., i] = rescale_single_channel_image(image[..., i])
    else:
        image = rescale_single_channel_image(image)

    return image


def roc_threshold(y_true, y_pred_prob):
    '''
    Function to retrieve the optimum threshold for the ROC curve
    @param y_true: true label
    @param y_pred_prob: predicted label from model output, must be in probability form!
    @return: optimum threshold
    '''
    fpr, tpr, thr = roc_curve(y_true, y_pred_prob)
    optimum_idx = np.argmax(tpr-fpr)
    opt_threshold = thr[optimum_idx]
    return opt_threshold


def auc_score(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def get_embedding_aug(embeddings, labels, n_inner_pts, l2_norm=True):
    '''
    Embedding augmentation methods in tensorflow form, adapted from:
    https://github.com/clovaai/embedding-expansion/blob/1aec8955e2e9f8d15f857ccdc6cc25545892da3d/loss/embedding_aug.py#L14
    @param embeddings: Feature embedding with tensor shape [B, embedding dimension]
    @param labels: Labels correlated to the embedding with tensor shape [B, ]
    @param n_inner_pts: Number of synthetic points
    @param l2_norm: L2 normalisation
    @return: augmented embeddings and augmented labels
    '''
    batch_size = embeddings.shape[0]
    labels = tf.squeeze(labels)
    swap_axes_list = [i + 1 if i % 2 == 0 else i - 1 for i in range(batch_size)]
    swap_embeddings = tf.gather(embeddings, swap_axes_list)
    pos = embeddings
    anchor = swap_embeddings
    # create a copy of the original embedding
    concat_embeddings = tf.identity(embeddings)
    # create a copy of the original labels
    concat_labels = tf.identity(labels)
    n_pts = n_inner_pts
    l2_normalize = l2_norm
    total_length = float(n_pts + 1)
    for n_idx in range(n_pts):
        left_length = float(n_idx + 1)
        right_length = total_length - left_length
        # linear interpolation
        inner_pts = (anchor * left_length + pos * right_length) / total_length
        if l2_normalize:
            inner_pts = tf.math.l2_normalize(inner_pts, axis=1)
        concat_embeddings = tf.concat([concat_embeddings, inner_pts], axis=0)
        concat_labels = tf.concat([concat_labels, labels], axis=0)

    return concat_embeddings, concat_labels


def get_c_idx(labels):
    c_ = np.unique(labels)
    class_list = []
    for c in c_:
        c_idx = np.where(labels == c)[0]
        # c_idx_len = c_idx.shape[0]
        class_list.append(c_idx)
    return class_list


def gridplot_info(query_nn, imgs_list, pid_list, lbl_list):
    '''
    Unravel image list, patient id list and label list for grid plot
    @param query_nn: query id with dimension => [n_rows, topk_nn]
    @param imgs_list: image stack
    @param pid_list: patient id stack
    @param lbl_list: label stack
    @return: flatten image list and flatten caption list with {"pid", "class"}
    '''
    tmp_img = []
    tmp_cap = []
    mid_slice = imgs_list[0].shape[0]//2
    for i in range(len(query_nn)):
        img_list = [imgs_list[j, mid_slice, ..., 0] for j in query_nn[i]]
        captions = ["pid: {}, class: {}".format(str(pid_list[j]).lstrip("0"), str(lbl_list[j])) for j in query_nn[i]]
        tmp_img.append(img_list)
        tmp_cap.append(captions)
    # unravel list within list
    img_list_flatten = sum(tmp_img, [])
    captions_flatten = sum(tmp_cap, [])

    return img_list_flatten, captions_flatten


def multi_get_split_fold(data, label_col='labels', fold_col='fold', val_fold=0, test_fold=-1):
    """
    If the data set is already split according to folds with indices [-1, 0, 1, 2, 3]
    @param data: csv file where the data sets are stored
    @param label_col: label column name
    @param fold_col: there can be multi fold columns name, by default "fold"
    @param val_fold: validation fold to be popped
    @param test_fold: test fold to be popped
    @return: dictionaries of train, val, train dictionary
    """
    # return indices of fold data
    test_idx = np.where(data[fold_col]==test_fold)[0]
    val_idx = np.where(data[fold_col]==val_fold)[0]
    train_idx = np.where((data[fold_col] != test_fold) & (data[fold_col] != val_fold))[0]

    # create dictionary for each data set
    train_ds = {'npy_path':[data['npy_path'].tolist()[i] for i in train_idx],
                'labels':[data[label_col].tolist()[i] for i in train_idx],
                'prostatitis':[data['prostatitis'].tolist()[i] for i in train_idx],
                'maglinant': [data['maglinant'].tolist()[i] for i in train_idx],
                'ggg': [data['ggg'].tolist()[i] for i in train_idx],
                'risk': [data['risk'].tolist()[i] for i in train_idx],
                'ggg_s': [data['ggg_s'].tolist()[i] for i in train_idx],
                'tumour': [data['tumour'].tolist()[i] for i in train_idx],
                'patient_id':[data['patient_id'].tolist()[i] for i in train_idx]}
    val_ds = {'npy_path':[data['npy_path'].tolist()[i] for i in val_idx],
                'labels':[data[label_col].tolist()[i] for i in val_idx],
                'prostatitis': [data['prostatitis'].tolist()[i] for i in val_idx],
                'maglinant': [data['maglinant'].tolist()[i] for i in val_idx],
                'ggg': [data['ggg'].tolist()[i] for i in val_idx],
                'risk': [data['risk'].tolist()[i] for i in val_idx],
                'ggg_s': [data['ggg_s'].tolist()[i] for i in val_idx],
                'tumour': [data['tumour'].tolist()[i] for i in val_idx],
                'patient_id':[data['patient_id'].tolist()[i] for i in val_idx]}
    test_ds = {'npy_path':[data['npy_path'].tolist()[i] for i in test_idx],
                'labels':[data[label_col].tolist()[i] for i in test_idx],
                'prostatitis': [data['prostatitis'].tolist()[i] for i in test_idx],
                'maglinant': [data['maglinant'].tolist()[i] for i in test_idx],
                'ggg': [data['ggg'].tolist()[i] for i in test_idx],
                'risk': [data['risk'].tolist()[i] for i in test_idx],
                'ggg_s': [data['ggg_s'].tolist()[i] for i in test_idx],
                'tumour': [data['tumour'].tolist()[i] for i in test_idx],
                'patient_id':[data['patient_id'].tolist()[i] for i in test_idx]}

    return {'train_ds':train_ds, 'val_ds': val_ds, 'test_ds':test_ds}


class DataLoader3D_multi(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded,
                 crop_status=True, crop_type="center",
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True,
                 infinite=True, margins=(0,0,0)):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        # original patch size with [slices, width, height]
        self.patch_size = patch_size
        self.num_modalities = 3
        self.indices = list(range(len(data['labels'])))
        self.crop_status = crop_status
        self.crop_type = crop_type
        self.margins = margins

    @staticmethod
    def load_patient(img_path):
        img = np.load(img_path, mmap_mode="r")
        return img

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data['npy_path'][i] for i in idx]
        patients_labels = [self._data['labels'][i] for i in idx]
        patient_id = [self._data['patient_id'][i] for i in idx]
        prostatitis = [self._data['prostatitis'][i] for i in idx]
        maglinant = [self._data['maglinant'][i] for i in idx]
        ggg = [self._data['ggg'][i] for i in idx]
        risk = [self._data['risk'][i] for i in idx]
        ggg_s = [self._data['ggg_s'][i] for i in idx]
        tumour = [self._data['tumour'][i] for i in idx]
        # initialize empty array for data and seg
        img = np.zeros((len(patients_for_batch), self.num_modalities, *self.patch_size), dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data = self.load_patient(j)

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            # swap axes for crop function, (z, x, y, m) => (m, x, y, z)
            patient_data = np.swapaxes(patient_data,0,-1)
            if self.crop_status:
                patient_data = crop(patient_data[None], seg=None, crop_size=self.patch_size,
                                    margins=self.margins, crop_type=self.crop_type)
            img[i] = patient_data[0]
        # img = np.swapaxes(img, 1, -1)
        patients_labels = np.array(patients_labels)
        return {'data': img, 'lbl': patients_labels, 'patient_id': patient_id,
                'prostatitis': prostatitis, 'maglinant':maglinant, 'ggg':ggg,
                'risk':risk, 'ggg_s':ggg_s, 'tumour':tumour}


def inverse_frequency_weight(labels, n_class):
    labels = tf.cast(labels, dtype=tf.int32)
    bins = tf.math.bincount(labels, maxlength=n_class)
    # for miss out class bins to be 1
    bins = tf.where(bins==0, 1, bins)
    weights = 1/bins
    # normalize
    weights /=tf.math.reduce_sum(weights)
    return tf.cast(weights, dtype=tf.float32)


