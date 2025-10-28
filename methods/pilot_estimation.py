import numpy as np

'''
This file contains the methods for the estimation of the channel at the pilot subcarriers. 
All methods have the same signature, so they can be used interchangeably.
'''

def naive(ffts: np.array, pilot_indices: np.array, pilot_symbol: np.complex128, **kwargs) -> np.array:
    '''
    Naive channel estimation at the pilot subcarriers by dividing the received pilot symbols by the known transmitted pilot symbol.
    
    :param np.array ffts: Received OFDM symbols in frequency domain (shape: frame_length x fft_size)
    :param np.array pilot_indices: Indices of the pilot subcarriers
    :param np.complex128 pilot_symbol: Known transmitted pilot value
    :return np.array: Estimated channel values at the pilot subcarriers (shape: frame_length x number_of_pilots)
    '''
    
    # Initialization of the list to hold the channel estimations for each OFDM symbol
    estimations = []
    
    # Iterate over each OFDM symbol in the frame
    for fft in ffts:        
        # Estimate the channel at the pilot subcarriers
        estimations.append(fft[pilot_indices] / pilot_symbol)
        
    return np.array(estimations)


def kalman_filter(ffts, pilot_indices, pilot_symbol, **kwargs):
    '''
    Kalman filter for channel estimation at the pilot subcarriers. 
    The Kalman filter assumes an AR(1) model for the temporal evolution of the channel. (as given in the assignment description)
    Filter parameters can be provided via kwargs:
        - corr_coeff: Correlation coefficient of the AR(1) model
        - process_noise_var: Process noise variance
        - noise_var: Gaussian noise variance due the transmission
    
    :param np.array ffts: Received OFDM symbols in frequency domain (shape: frame_length x fft_size)
    :param np.array pilot_indices: Indices of the pilot subcarriers
    :param np.complex128 pilot_symbol: Known transmitted pilot value
    :return np.array: Estimated channel values at the pilot subcarriers (shape: frame_length x number_of_pilots)        
    '''

    # Initialization of the list to hold the channel estimations for each OFDM symbol
    estimations = []
    
    # Set previous estimation to 0 as the mean of h is modeled to be 0
    channel_estimation_prev = np.zeros_like(pilot_indices, dtype=complex)
    # Set the previous error covariance to sigma_h
    error_cov_prev = np.ones_like(pilot_indices, dtype=float) * kwargs['tap_variance']

    # Iterate over each OFDM symbol in the frame
    for fft in ffts:      
        # Calculate the naive estimate for the OFDM symbol
        naive_estimate = fft[pilot_indices] / pilot_symbol
        
        # Intialize the estimation and error covariance arrays
        channel_estimate = np.zeros_like(naive_estimate)
        error_cov_estimate = np.zeros_like(error_cov_prev)
        
        # Iterate over the pilot subcarriers
        for k in range(len(pilot_indices)):
            # Calculate an intermediate prediction of the channel and error covariance by considering the correlation of the AR(1) model
            channel_prediction = channel_estimation_prev[k] * kwargs['corr_coeff']
            error_cov_prediction = kwargs['corr_coeff'] ** 2 * error_cov_prev[k] + kwargs['tap_variance'] * (1 - kwargs['corr_coeff'] ** 2)
            
            # Calculate the Kalman gain
            kalman_gain = error_cov_prediction / (error_cov_prediction + kwargs['noise_variance'])

            # Update the channel estimate and error covariance using the Kalman gain and the naive estimate
            channel_estimate[k] = channel_prediction + kalman_gain * (naive_estimate[k] - channel_prediction)
            error_cov_estimate[k] = (1 - kalman_gain) * error_cov_prediction

        # Update the previous estimation and error covariance for the next iteration
        channel_estimation_prev = channel_estimate
        error_cov_prev = error_cov_estimate

        estimations.append(channel_estimate)
    return np.array(estimations)