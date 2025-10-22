import numpy as np
from methods import pilot_estimation


def decode(signal: np.array, pilot_estimation_method: callable, interpolation_method: callable, equalizing_method: callable, **kwargs) -> np.array:
    '''
    Decode the received signal using variable methods for pilot estimation, channel interpolation, and equalization.
    
    :param np.array signal: Received OFDM signal
    :param function pilot_estimation_method: Method for pilot estimation
    :param function interpolation_method: Method for channel interpolation
    :param function equalizing_method: Method for equalization
    :param dict kwargs: Additional arguments passed on to the various methods
    :return np.array: Estimated QPSK symbols    
    '''
    
    # Extract parameters from kwargs
    fft_size = kwargs.get('fft_size')
    prefix_length = kwargs.get('prefix_length')
    frame_length = kwargs.get('frame_length')
    data_indices = kwargs.get('data_indices')
    pilot_indices = kwargs.get('pilot_indices')
    pilot_symbol = kwargs.get('pilot_symbol')
    channel_length = kwargs.get('channel_length')
    corr_coeff = kwargs.get('corr_coeff', 0.99)
    
    # Removing the cyclic prefix and splitting the signal into the OFDM symbols
    signals = [signal[i * fft_size + (i + 1) * prefix_length : (i + 1) * (fft_size + prefix_length)] for i in range(frame_length)]
    
    # Calculate the FFT for each OFDM symbol
    ffts = np.array([np.fft.fft(s) / np.sqrt(fft_size) for s in signals])
    
    # Estimate the noise variance by using zero subcarriers from the zero padding of the FFTs
    zero_subcarriers = np.setdiff1d(np.arange(fft_size), np.concatenate([data_indices, pilot_indices]))
    zero_samples = np.concatenate([ffts[i][zero_subcarriers] for i in range(frame_length)])
    noise_variance = np.mean(np.abs(zero_samples)**2)
    
    # Estimate the process variance for the Kalman filter if not provided (WIP, no idea if this actually works)
    all_pilot_estimations_naive = pilot_estimation.naive(ffts, pilot_indices, pilot_symbol).flatten()
    tap_variance_estimate = fft_size * (np.mean(np.abs(all_pilot_estimations_naive) ** 2) - noise_variance) / channel_length
    
    # Estimate the channel for all pilot subcarriers using the provided pilot estimation method
    pilot_kwargs = {'corr_coeff': corr_coeff, 'tap_variance': tap_variance_estimate, 'noise_variance': noise_variance}
    pilot_estimations = pilot_estimation_method(ffts, pilot_indices, pilot_symbol, **pilot_kwargs)
    
    # Interpolate the channel for all subcarriers using the provided interpolation method considering the pilot estimations
    gamma = tap_variance_estimate
    interpolation_kwargs = {'noise_variance': noise_variance, 'gamma': gamma, 'channel_length': channel_length}
    channel_estimations = [interpolation_method(fft_size, pilot_indices, pilot_estimations[i], **interpolation_kwargs) for i in range(frame_length)]
    
    # Use the channel estimations and the provided equalizer to estimate the QPSK symbols for all data subcarriers
    qpsk_estimations = [equalizing_method(ffts[i], channel_estimations[i], noise_variance)[data_indices] for i in range(frame_length)]    

    # Combine all estimated QPSK symbols into a single array and return
    return np.concatenate(qpsk_estimations)


def calculate_mse(qpsk_estimations: np.array, true_symbols: np.array) -> float:
    ''' 
    Calculate Mean Squared Error between estimated and true symbols

    :param np.array qpsk_estimations: Estimated QPSK symbols
    :param np.array true_symbols: True QPSK symbols

    :return float: Mean Squared Error
    '''
    
    errors = qpsk_estimations - true_symbols
    return np.mean(np.abs(errors)**2)


def calculate_ber(qpsk_estimations: np.array, true_symbols: np.array) -> float:
    '''
    Calculate Bit Error Rate between demodulated estimated and true symbols

    :param np.array qpsk_estimations: Estimated QPSK symbols
    :param np.array true_symbols: True QPSK symbols

    :return float: Bit Error Rate
    '''

    # Demodulate estimated QPSK symbols to bits
    estimated_bits = np.array([(1 if s.real > 0 else 0, 1 if s.imag > 0 else 0) for s in qpsk_estimations]).flatten()
    
    # Demodulate true QPSK symbols to bits
    true_bits = np.array([(1 if s.real > 0 else 0, 1 if s.imag > 0 else 0) for s in true_symbols]).flatten()

    # Calculate Bit Error Rate
    bit_error_rate = np.sum(estimated_bits != true_bits) / len(true_bits)
    return bit_error_rate