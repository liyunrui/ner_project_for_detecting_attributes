import copy
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):

    """Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.

    Args:
        columns: List of names corresponding to the matrices in data.
        data: List of n-dimensional data matrices ordered in correspondence with columns.
            All matrices must have the same leading dimension.  Data can also be fed a list of
            instances of np.memmap, in which case RAM usage can be limited to the size of a
            single batch.
    """

    def __init__(self, columns, data):
        '''
        columns: list of string.
        data: list of 2-D array
        '''
        assert len(columns) == len(data), 'columns length does not match data length'

        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths)) == 1, 'all matrices in data must have same first dimension'

        self.length = lengths[0] # length: number of training example
        self.columns = columns
        self.data = data
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length) # 1-D array with length ====> (self.length, )

    def shapes(self):
        return pd.Series(dict(zip(self.columns, [mat.shape for mat in self.data])))

    def dtypes(self):
        return pd.Series(dict(zip(self.columns, [mat.dtype for mat in self.data])))

    def shuffle(self):
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(int(time.time()))):
        '''
        It's built for train/val splitting.

        How to do train/val splitting.
            1.Split by number of rows (training examples), more easily way.
            2.Split by item_id,
        '''
        train_idx, test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)
        train_df = DataFrame(copy.copy(self.columns), [mat[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat[test_idx] for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=1000000, allow_smaller_final_batch=False):
        # num_epochs = 10,000 (by default)
        epoch_num = 0
        while epoch_num < num_epochs:
            '''
            Definition of one epoch is optimizing of all the training examples. 

            For example, if allow_smaller_final_batch = False
            training data = [1,2,3,4,5]
            batch size = 2
            ==> [1,2], [3,4] 
            otherwise(allow_smaller_final_batch = True)
            ===>
            '''
            if shuffle:
                self.shuffle()

            for i in range(0, self.length, batch_size):
                batch_idx = self.idx[i: i + batch_size] # 1-D array with length og batch_size
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    # teminate the loop, when facing smaller batch size. then it won't yield the bloew code.
                    break
                yield DataFrame(columns=copy.copy(self.columns), data=[mat[batch_idx].copy() for mat in self.data])

            epoch_num += 1

    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def __iter__(self):
        '''The iter() method creates an object which can be iterated one element at a time. 
        It's useful when coupled with loops'''
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        '''
        Make this object support indexing.

        Reference of understanding here: https://stackoverflow.com/questions/43627405/understanding-getitem-method
        '''
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        '''
        Make this object support item assignment, aka having dict-like functionality in python.

        It's a helper function to make the usage of the this class better, coupled with __getitem__.  

        '''
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        # item assignment
        self.dict[key] = value
