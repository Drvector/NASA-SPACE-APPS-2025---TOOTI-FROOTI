"""
dataset_generator.py
- builds tf.data.Dataset or Keras Sequence from lists of candidates
- yields (global_view, local_view), label batches
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class LCSequence(Sequence):
    def __init__(self, entries, batch_size=32, shuffle=True):
        """
        entries: list of dicts with keys:
           'global': numpy array shape (G,)
           'local': numpy array shape (L,)
           'label': 0 or 1
        """
        self.entries = entries
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.entries) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.entries[idx*self.batch_size:(idx+1)*self.batch_size]
        g = np.array([e['global'] for e in batch])[:,:,None]
        l = np.array([e['local'] for e in batch])[:,:,None]
        y = np.array([e['label'] for e in batch]).astype(np.float32)
        return {'global_input': g, 'local_input': l}, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.entries)
