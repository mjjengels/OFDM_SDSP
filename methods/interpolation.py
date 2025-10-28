import numpy as np
from scipy.interpolate import CubicSpline

def linear(fft_size: int, pilot_indices: np.array, pilot_channel_estimation: np.array, **kwargs) -> np.array:
    '''
    Linear interpolation of the channel estimates
    
    :param int fft_size: Total number of subcarriers
    :param np.array pilot_indices: Indices of the pilot subcarriers
    :param np.array pilot_channel_estimation: Estimated channel values at the pilot subcarriers
    :return np.array: Estimated channel values at all subcarriers
    '''

    # Initialize the channel estimation array
    channel_est = np.zeros(shape=(fft_size,), dtype=np.complex128)
    channel_est[pilot_indices] = pilot_channel_estimation
    
    for k in range(fft_size):
        if k in pilot_indices:
            continue
        # Find next smallest and next largest pilot indices
        k1 = pilot_indices[pilot_indices < k].max() if np.any(pilot_indices < k) else pilot_indices[0]
        k2 = pilot_indices[pilot_indices > k].min() if np.any(pilot_indices > k) else pilot_indices[-1]
        
        # Avoid zero-division case
        if k1 == k2:
            channel_est[k] = channel_est[k1]
            continue
        
        # Calculate remaining channel estimates by linear interpolation
        channel_est[k] = channel_est[k1] + (k - k1) / (k2 - k1) * (channel_est[k2] - channel_est[k1])
        
    return channel_est


def wiener(fft_size: int, pilot_indices: np.array, pilot_channel_estimation: np.array, **kwargs) -> np.array:
    '''
    Wiener filter in frequency domain to estimate the channel at all subcarriers
    
    :param int fft_size: Total number of subcarriers
    :param np.array pilot_indices: Indices of the pilot subcarriers
    :param np.array pilot_channel_estimation: Estimated channel values at the pilot subcarriers
    
    :return np.array: Estimated channel values at all subcarriers    
    '''

    ell = np.arange(kwargs.get('channel_length'))[None, :]
    subcarrier_idx = np.arange(fft_size)[:, None]
    pilot_idx = pilot_indices[:, None]
    
    phiX = np.exp(-1j * 2 * np.pi * (subcarrier_idx @ ell) / fft_size) / np.sqrt(fft_size)
    phiP = np.exp(-1j * 2 * np.pi * (pilot_idx @ ell) / fft_size) / np.sqrt(fft_size)

    gamma = kwargs.get('gamma')
    R_pp = gamma * phiP @ phiP.conj().T
    R_xp = gamma * phiX @ phiP.conj().T
    
    sigma2 = kwargs.get('noise_variance')
    try:
        alpha = np.linalg.solve(R_pp + sigma2 * np.eye(R_pp.shape[0]), pilot_channel_estimation)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(R_pp + sigma2 * np.eye(R_pp.shape[0])) @ pilot_channel_estimation

    return R_xp @ alpha


def spline(fft_size: int, pilot_indices: np.array, pilot_channel_estimation: np.array, **kwargs) -> np.array:
    '''
    Spline interpolation of the channel estimates
    
    :param int fft_size: Total number of subcarriers
    :param np.array pilot_indices: Indices of the pilot subcarriers
    :param np.array pilot_channel_estimation: Estimated channel values at the pilot subcarriers
    :return np.array: Estimated channel values at all subcarriers    
    '''
    
    # Intialize the channel estimation array 
    channel_est = np.zeros(shape=(fft_size,), dtype=np.complex128)
    
    # Fit cubic spline separately to real and imaginary parts
    cs_real = CubicSpline(pilot_indices, pilot_channel_estimation.real, bc_type='natural')
    cs_imag = CubicSpline(pilot_indices, pilot_channel_estimation.imag, bc_type='natural')
    
    # Evaluate the spline at all subcarrier indices
    all_idx = np.arange(fft_size)
    channel_est = cs_real(all_idx) + 1j * cs_imag(all_idx)

    return channel_est