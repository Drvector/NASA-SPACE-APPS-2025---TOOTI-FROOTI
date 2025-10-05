"""
train.py
- Example training driver: load a prepared CSV/NPZ list of processed candidates
- trains and saves model weights
"""

import os
import numpy as np
from model import build_astronet
from dataset_generator import LCSequence
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_PREPARED = "data/prepared_candidates.npz"  # produced by your data pipeline
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_prepared_npz(path):
    """
    expects arrays: global_views (N,G), local_views (N,L), labels (N,)
    """
    data = np.load(path)
    return data['global_views'], data['local_views'], data['labels']

def main():
    global_views, local_views, labels = load_prepared_npz(DATA_PREPARED)
    entries = [{'global': global_views[i], 'local': local_views[i], 'label': int(labels[i])} for i in range(len(labels))]

    train_e, val_e = train_test_split(entries, test_size=0.15, stratify=labels, random_state=42)
    batch_size = 64
    train_seq = LCSequence(train_e, batch_size=batch_size, shuffle=True)
    val_seq = LCSequence(val_e, batch_size=batch_size, shuffle=False)

    model = build_astronet(global_length=global_views.shape[1], local_length=local_views.shape[1], lr=3e-4)

    cb = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR, "astronet_best.h5"),
                                           save_weights_only=True, save_best_only=True, monitor='val_auc', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
    ]

    model.fit(train_seq, validation_data=val_seq, epochs=60, callbacks=cb)
    model.save(os.path.join(OUT_DIR, "astronet_final.h5"))

if __name__ == "__main__":
    main()
