import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from torch import utils
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        ''' Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        '''
        ids = np.stack([x[0] for x in data_holder])
        inputs = [resample_pcd(x[1], self.input_size) if x[1].shape[0] > self.input_size else x[1]
            for x in data_holder]
        inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(np.float32)
        npts = np.stack([x[1].shape[0] if x[1].shape[0] < self.input_size else self.input_size
            for x in data_holder]).astype(np.int32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        return ids, inputs, npts, gts


def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)
        df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = BatchData(df, batch_size, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size

def to_var(x):
    if torch.cuda.is_available():
        x=x.cuda()
    return Variable(x)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def plot_loss(losses):
    plt.figure(figsize=(16,9))
    plt.plot(losses)
    plt.title('Training losses')


def loadDataFile(filename):
    return load_h5(filename)


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    write_point_cloud(filename, pcd)