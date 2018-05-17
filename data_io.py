import warnings
from glob import glob
import tensorflow as tf
import os
import nibabel as nb
import numpy as np
from nilearn import image
import pandas as pd


def _get_resize_arg(target_shape):
    mni_shape_mm = np.array([144.0, 174.0, 150.0])
    target_resolution_mm = mni_shape_mm / np.array(target_shape)
    target_affine = np.array([[4., 0., 0., -70.],
                              [0., 4., 0., -103.],
                              [0., 0., 4., -65.],
                              [0., 0., 0., 1.]])
    target_affine[0, 0] = target_resolution_mm[0]
    target_affine[1, 1] = target_resolution_mm[1]
    target_affine[2, 2] = target_resolution_mm[2]
    return target_affine, list(target_shape)


def _get_data(batch_size, src_folder, n_epochs, cache_prefix,
              shuffle, target_shape, balance_dataset=True, nthreads=None):
    if nthreads is None:
        import multiprocessing
        nthreads = multiprocessing.cpu_count()

    in_images = glob(os.path.join(src_folder, "*.nii*"))
    assert len(in_images) != 0

    paths = tf.constant(in_images)
    paths_ds = tf.data.Dataset.from_tensor_slices((paths,))

    target_affine, target_shape = _get_resize_arg(target_shape)

    target_affine_tf = tf.constant(target_affine)
    target_shape_tf = tf.constant(target_shape)

    def _read_and_resample(path, target_affine, target_shape):
        path_str = path.decode('utf-8')
        nii = nb.load(path_str)
        data = nii.get_data()
        data[np.isnan(data)] = 0
        nii = nb.Nifti1Image(data, nii.affine)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nii = image.resample_img(nii,
                                     target_affine=target_affine,
                                     target_shape=target_shape)
        data = nii.get_data().astype(np.float32)

        return data

    data_ds = paths_ds.map(
        lambda path: tuple(tf.py_func(_read_and_resample,
                                      [path, target_affine_tf,
                                       target_shape_tf],
                                      [tf.float32], name="read_and_resample")),
        num_parallel_calls=nthreads)

    def _reshape_images(data):
        return tf.reshape(data, target_shape)

    data_ds = data_ds.map(_reshape_images, num_parallel_calls=nthreads)

    def _get_label(path):
        path_str = path.decode('utf-8')
        df = pd.read_csv(
            "D:/data/PAC_Data/PAC_Data/PAC2018_Covariates_Upload.csv")
        PAC_ID = path_str.split(os.sep)[-1].split('.')[0]
        label = int(df[df.PAC_ID == PAC_ID]['Label']) - 1
        return label

    labels_ds = paths_ds.map(
        lambda path: tuple(tf.py_func(_get_label,
                                      [path],
                                      [tf.int32], name="get_label")),
        num_parallel_calls=nthreads)

    def _reshape_labels(labels):
        return tf.reshape(labels, [])

    labels_ds = labels_ds.map(_reshape_labels, num_parallel_calls=nthreads)

    dataset = tf.data.Dataset.zip((data_ds, labels_ds))

    # Sanity check uncomment those lines to zero values for one label
    # making the classfication task trivial
    #def _zero(data, label):
    #    return tf.multiply(data, tf.cast(label, tf.float32)), label
    #
    #dataset = dataset.map(_zero)

    if cache_prefix:
        dataset = dataset.cache(cache_prefix)

    if balance_dataset:
        # balance classes
        dataset = dataset.apply(tf.contrib.data.rejection_resample(lambda _, label: label,
                                                                   [0.5, 0.5]))
        # see https://stackoverflow.com/a/47056930/616300
        dataset = dataset.map(lambda _, data: (data))

    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100,
                                                                   count=n_epochs))
    else:
        dataset = dataset.repeat(n_epochs)
    return dataset


class InputFnFactory:

    def __init__(self, target_shape, batch_size,
                 n_epochs, train_src_folder,
                 train_cache_prefix, eval_src_folder, eval_cache_prefix):
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.train_src_folder = train_src_folder
        self.train_cache_prefix = train_cache_prefix
        self.eval_src_folder = eval_src_folder
        self.eval_cache_prefix = eval_cache_prefix

    def _get_iterator(self, cache_prefix, src_folder, shuffle, n_epochs,
                      balance_dataset):
        dataset = _get_data(batch_size=self.batch_size,
                            src_folder=src_folder,
                            n_epochs=n_epochs,
                            cache_prefix=cache_prefix,
                            shuffle=shuffle,
                            target_shape=self.target_shape,
                            balance_dataset=balance_dataset)
        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        training_iterator = dataset.make_one_shot_iterator()
        return training_iterator.get_next()

    def train_input_fn(self):
        return self._get_iterator(cache_prefix=self.train_cache_prefix,
                                  src_folder=self.train_src_folder,
                                  shuffle=True,
                                  n_epochs=self.n_epochs,
                                  balance_dataset=True)

    def eval_input_fn(self):
        return self._get_iterator(cache_prefix=self.eval_cache_prefix,
                                  src_folder=self.eval_src_folder,
                                  shuffle=False,
                                  n_epochs=1,
                                  balance_dataset=False)