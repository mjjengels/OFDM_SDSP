import numpy as np

# Zero-Forcing
def zero_forcing(received_symbols, channel_estimation, noise_variance):
    return received_symbols / channel_estimation

# Minimum Mean Square Error
def mmse(received_symbols, channel_estimation, noise_variance):
    return np.conj(channel_estimation) / (np.abs(channel_estimation)**2 + noise_variance) * received_symbols