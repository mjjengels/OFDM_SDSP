'''
This file contains the code for the terminal interface to evaluate the performance of a specific combination of methods.
The methods can be specified using command line arguments.
'''

import argparse
import os
import scipy.io
import numpy as np
from methods import pilot_estimation, interpolation, equalizer
from pipeline import decode, calculate_mse, calculate_ber


PILOT_CHOICES = ['naive', 'kalman']
INTERP_CHOICES = ['linear', 'spline', 'wiener']
EQUALIZER_CHOICES = ['mmse', 'zf']
DATASET_CHOICES = [1, 2, 3]
NOISE_LEVELS = ['no_noise', 'low_noise', 'high_noise']

parser = argparse.ArgumentParser(description="Evaluate performance of specified methods.")

parser.add_argument('--p', type=str, choices=PILOT_CHOICES, required=True, help='Pilot estimation method')
parser.add_argument('--i', type=str, choices=INTERP_CHOICES, required=True, help='Interpolation method')
parser.add_argument('--e', type=str, choices=EQUALIZER_CHOICES, required=True, help='Equalizer method')
parser.add_argument('--d', type=int, required=True, choices=DATASET_CHOICES, help='Dataset Id')
parser.add_argument('--n', type=str, required=True, choices=NOISE_LEVELS, help='Noise level for evaluation')

args = parser.parse_args()

if __name__ == "__main__":
    # Map method names to actual functions
    pilot_methods = {
        'naive': pilot_estimation.naive,
        'kalman': pilot_estimation.kalman_filter
    }
    
    interp_methods = {
        'linear': interpolation.linear,
        'spline': interpolation.spline,
        'wiener': interpolation.wiener
    }
    
    equalizer_methods = {
        'mmse': equalizer.mmse,
        'zf': equalizer.zero_forcing
    }
    
    noise_levels = {
        'no_noise': 'NoNoise_RxSignal',
        'low_noise': 'LowNoise_RxSignal',
        'high_noise': 'HighNoise_RxSignal'
    }
    
    dataset = scipy.io.loadmat(f'Dataset/DataSet{args.d}.mat')    
    signal = dataset[noise_levels[args.n]].flatten()
    
    ofdm = dataset['OFDM']
    channel = dataset['Channel']
    image_size = [int(i) for i in dataset['ImageSize'][0]]

    channel = { 'Name': str(channel[0][0][0][0]),
                'Length': int(channel[0][0][1][0][0]),
                'DopplerFreq': int(channel[0][0][2][0][0]),
                'CorrCoefficient': float(channel[0][0][3][0][0])}

    ofdm = {'Name': str(ofdm[0][0][0][0]),
            'Bandwidth': int(ofdm[0][0][0][0].split('Mz')[0]) * 10**6,
            'SamplingFrq': int(ofdm[0][0][1][0][0]),
            'CarrierFreq': int(ofdm[0][0][2][0][0]),
            'FFTSize': int(ofdm[0][0][3][0][0]),
            'DataSubcarriers': int(ofdm[0][0][4][0][0]),
            'Modulation': str(ofdm[0][0][5][0]),
            'PilotIndices': ofdm[0][0][6][0],
            'DataIndices': ofdm[0][0][7][0],
            'FrameLen': int(ofdm[0][0][8][0][0]),
            'PilotSymbols': ofdm[0][0][9][0]}

    pilot_indices = ofdm['PilotIndices'] - 1  # Convert to zero-based indexing
    data_indices = ofdm['DataIndices'] - 1    # Convert to zero-based indexing

    channel_length = channel['Length']
    prefix_length = channel_length - 1
    fft_size = ofdm['FFTSize']
    frame_length = ofdm['FrameLen']

    num_pilots = len(pilot_indices)
    pilot_symbol = ofdm['PilotSymbols'][0]

    corr_coeff = channel['CorrCoefficient']
    
    qpsk_estimations = decode(
        signal,
        pilot_estimation_method=pilot_methods[args.p],
        interpolation_method=interp_methods[args.i],
        equalizing_method=equalizer_methods[args.e],
        fft_size=fft_size,
        prefix_length=prefix_length,
        frame_length=frame_length,
        data_indices=data_indices,
        pilot_indices=pilot_indices,
        pilot_symbol=pilot_symbol,
        channel_length=channel_length,
        corr_coeff=corr_coeff
    )
    
    true_symbols = np.loadtxt(f'recovered_signals/recovered_signal_dataset{args.d}.txt', dtype=np.complex128)
    ber = calculate_ber(qpsk_estimations[:image_size[0] * image_size[1] // 2], true_symbols)
    mse = calculate_mse(qpsk_estimations[:image_size[0] * image_size[1] // 2], true_symbols)
    
    pilot_method_name = {
        'naive': 'Naive',
        'kalman': 'Kalman Filter'
    }
    
    interpolation_method_name = {
        'linear': 'Linear Interpolation',
        'spline': 'Spline Interpolation',
        'wiener': 'Wiener Filter'
    }
    
    equalizer_method_name = {
        'zf': 'Zero-Forcing',
        'mmse': 'MMMSE'
    }
    
    print(
        f"Evaluated dataset {args.d} at {args.n.replace('_', ' ')} with\n"
        f"\t- Pilot Estimation:\t{pilot_method_name[args.p]}\n"
        f"\t- Interpolation:\t{interpolation_method_name[args.i]}\n"
        f"\t- Equalizer:\t\t{equalizer_method_name[args.e]}\n"
        f"-- Results --\n"
        f"BER: {ber}\n"
        f"MSE: {mse}"
    )