import streamlit as st
import joblib
import pandas as pd
import numpy as np
import scipy.io
from scipy.stats import skew, kurtosis, iqr
from scipy.signal import welch
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / (len(signal) - 1)
def rms(signal):
    signal = np.nan_to_num(signal)
    return np.sqrt(np.mean(signal**2))
def spectral_entropy(psd):
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))
def bandpower(psd, freqs, band):
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.sum(psd[idx])
def extract_features_from_segment(mat_file_path):
    mat = scipy.io.loadmat(mat_file_path)
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    data_struct = mat[key]
    eeg_data = data_struct['data'][0, 0]
    fs = int(data_struct['sampling_frequency'].squeeze().item())
    window_features = []
    for channel_data in eeg_data:
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        median_val = np.median(channel_data)
        skewness_val = skew(channel_data)
        kurtosis_val = kurtosis(channel_data)
        range_val = max_val - min_val
        iqr_val = iqr(channel_data)
        variance_val = np.var(channel_data)
        rms_val = rms(channel_data)
        zcr_val = zero_crossing_rate(channel_data)
        freqs, psd = welch(channel_data, fs=fs, nperseg=min(256, len(channel_data)//2))
        delta_power = bandpower(psd, freqs, (0.5, 4))
        theta_power = bandpower(psd, freqs, (4, 8))
        alpha_power = bandpower(psd, freqs, (8, 13))
        beta_power = bandpower(psd, freqs, (13, 30))
        gamma_power = bandpower(psd, freqs, (30, 100))
        spec_entropy = spectral_entropy(psd)
        window_features.extend([
            mean_val, std_val, min_val, max_val, median_val,
            skewness_val, kurtosis_val, range_val, iqr_val,
            variance_val, rms_val, zcr_val,
            delta_power, theta_power, alpha_power, beta_power, gamma_power,
            spec_entropy
        ])
    return window_features
num_channels_to_keep = 15
num_stats = 18
num_features_to_keep = num_channels_to_keep * num_stats
stats = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt', 'range', 'iqr', 'var', 'rms', 'zcr',
         'bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta', 'bp_gamma', 'entropy']
col_names = [f'ch{i+1}_{stat}' for i in range(num_channels_to_keep) for stat in stats]
model = joblib.load("final_model.pkl")
def predict_segment(mat_file_path):
    features = extract_features_from_segment(mat_file_path)
    trimmed = features[:num_features_to_keep]
    df = pd.DataFrame([trimmed], columns=col_names)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability
st.set_page_config(page_title="Seizure Predictor", layout="centered")
st.title("EEG Seizure Prediction")
st.markdown("Upload a `.mat` EEG test segment file for prediction.")
uploaded_file = st.file_uploader("Choose a `.mat` file", type="mat")
if uploaded_file:
    try:
        st.info("Processing uploaded file...")
        prediction, probability = predict_segment(uploaded_file)
        st.subheader("Prediction Result")
        st.write("**Preictal**:\n Alert!! Seizure About To occur" if prediction == 1 else "**Interictal**:\n Safe, No signs of seizure to occur shortly")
        st.write("**Seizure Probability** (0.5 and less is the safe state)")
        st.write(f"{probability:.3f}")

    except Exception as e:
        st.error(f"An error occurred!!!: {e}")