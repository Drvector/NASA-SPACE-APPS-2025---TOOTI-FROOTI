"""
model.py
- dual-input 1D CNN in Keras (Astronet-like)
- two inputs: global and local views (1D arrays)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

def conv_block(x, filters, kernel_size=5, pool=2):
    x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool)(x)
    return x

def build_astronet(global_length=400, local_length=200, lr=1e-4):
    # global branch
    g_input = layers.Input(shape=(global_length,1), name='global_input')
    xg = conv_block(g_input, 16, kernel_size=11, pool=2)
    xg = conv_block(xg, 32, kernel_size=9, pool=2)
    xg = conv_block(xg, 64, kernel_size=7, pool=2)
    xg = layers.Flatten()(xg)
    xg = layers.Dense(128, activation='relu')(xg)
    xg = layers.Dropout(0.4)(xg)

    # local branch
    l_input = layers.Input(shape=(local_length,1), name='local_input')
    xl = conv_block(l_input, 16, kernel_size=7, pool=2)
    xl = conv_block(xl, 32, kernel_size=5, pool=2)
    xl = conv_block(xl, 64, kernel_size=3, pool=2)
    xl = layers.Flatten()(xl)
    xl = layers.Dense(128, activation='relu')(xl)
    xl = layers.Dropout(0.4)(xl)

    # merge
    merged = layers.concatenate([xg, xl])
    dense = layers.Dense(256, activation='relu')(merged)
    dense = layers.Dropout(0.5)(dense)
    out = layers.Dense(1, activation='sigmoid', name='output')(dense)

    model = models.Model(inputs=[g_input, l_input], outputs=out)
    model.compile(optimizer=optimizers.Adam(lr),
                  loss=losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])
    return model

if __name__ == "__main__":
    m = build_astronet()
    m.summary()
