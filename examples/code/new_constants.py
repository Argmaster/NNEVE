from tensorflow import keras

from nneve.quantum_oscillator import QOConstants, QOTracker

constants = QOConstants(
    optimizer=keras.optimizers.Adam(),
    tracker=QOTracker(),
    k=4.0,
    mass=1.0,
    x_left=-6.0,
    x_right=6.0,
    fb=1.0,
    sample_size=500,
    v_f=1.0,
    v_lambda=1.0,
    v_drive=1.0,
)
