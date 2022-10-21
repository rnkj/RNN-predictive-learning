import numpy as np

import torch
from torch.utils.data import Dataset


class SingleCircleDataset(Dataset):
    """
    A sample dataset which provides a single circle
    The radius of the circle is 2, and the center is (2, 2).
    """
    def __init__(self, stop=2*np.pi, length=50):
        """
        stop: the endpoint of numpy.linspace
        length: the number of data
        """
        super(SingleCircleDataset, self).__init__()

        ts = np.linspace(0, stop, num=length, dtype=np.float32)
        xs = np.cos(ts)
        ys = np.sin(ts)
        # center is (2, 2), and radius is 2
        data = 2 * (1 + np.c_[xs, ys]) # (`length`, 2) shape
        data = data[np.newaxis, ...] # (1, `length`, 2) shape

        self.inputs = torch.tensor(data[:, :-1])
        self.teachers = torch.tensor(data[:, 1:])

    def __len__(self):
        """
        return batch size.
        This dataset is composed of a single sequence drawing a circle.
        Therefore this dataset always returns 1.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        return training sequence (NOT data points).
        This dataset is composed of a single sequence drawing a circle.
        Therefore `index` is ignored here, and always returns the same sequence.
        """
        return self.inputs, self.teachers

    def get_sequence(self):
        return self[0] # Index 0 is a formal index, and ignored. See `__getitem__`

    @property
    def dims(self):
        return self.inputs.shape[-1]
