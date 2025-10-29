# OFDM Channel Estimation and Detection

This repository contains the code for the assignment “Channel Estimation and Data Detection in OFDM Systems”. It lets you evaluate and compare different pilot-channel estimation, interpolation, and equalization methods on provided datasets and noise levels.

## Setup

1) Clone this repository.
2) (Recommended) Create a virtual environment.
   - Windows:
     - Create: `python -m venv .venv`
     - Activate: `.\.venv\Scripts\activate`
   - macOS/Linux:
     - Create: `python3 -m venv .venv`
     - Activate: `source .venv/bin/activate`
3) Install dependencies: `pip install -r requirements.txt`

## Evaluate a single configuration

Use `evaluate_method.py` to evaluate one specific combination of methods on one dataset and noise level. The script prints Bit Error Rate (BER) and Mean Squared Error (MSE).

Available arguments (see `--help`):

```
        -h, --help            			show this help message and exit
        -p {naive,kalman}    			Pilot estimation method
        -i {linear,spline,wiener} 		Interpolation method                                                                 					   
        -e {mmse,zf}         			Equalizer method
        -d {1,2,3}           			Dataset Id
        -n {no_noise,low_noise,high_noise} 	Noise level for evaluation   
        --image 				Displays the recovered image   
```

Example:

```
 python .\evaluate_method.py -p kalman -i wiener -e mmse -d 2 -n low_noise
```

Example output:

```
Evaluated dataset 2 at low noise with
                                - Pilot Estimation:     Kalman Filter
                                - Interpolation:        Wiener Filter
                                - Equalizer:            MMSE
-- Results --
BER: 0.01256198347107438
MSE: 0.07887508467908878
```

## Compare multiple methods

The notebook `performance_eval.ipynb` evaluates all combinations of methods across all datasets and noise levels, then renders comparison tables.

What it produces

- A 12×9 results grid (2 pilot methods × 3 interpolation methods × 2 equalizers) × (3 datasets × 3 noise levels).
- Three DataFrames: `mse_df`, `ber_df`, `be_df` (Bit Errors), plus a styled view to highlight column-wise best/worst.

How to use

1) Open and run all cells in `performance_eval.ipynb`.
2) In the last cell, switch which DataFrame is displayed:

```python
# Show MSE (default in the notebook)
styled_df = mse_df.style.apply(highlight_max, axis=0).apply(highlight_min, axis=0)

# Or show BER
# styled_df = ber_df.style.apply(highlight_max, axis=0).apply(highlight_min, axis=0)

# Or show absolute Bit Errors
# styled_df = be_df.style.apply(highlight_max, axis=0).apply(highlight_min, axis=0)
```

