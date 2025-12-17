import os
import re
import random
import numpy as np
import pandas as pd
import streamlit as st

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# -----------------------------
# Utilities: loading spectra
# -----------------------------
SUPPORTED_EXTS = (".csv", ".txt", ".dat")

def list_spectra_files(folder: str):
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if f.lower().endswith(SUPPORTED_EXTS):
                files.append(os.path.join(root, f))
    files.sort()
    return files

def load_spectrum(path: str):
    """
    Loads a 2-column spectrum and returns (x, y) as float arrays.
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            cols = [c.lower().strip() for c in df.columns]
            df.columns = cols
            energy_candidates = ["energy", "e", "en", "ev"]
            mu_candidates = ["mu", "i", "intensity", "xmu"]

            def pick_col(cands):
                for c in cands:
                    if c in df.columns: return c
                return None

            e_col = pick_col(energy_candidates)
            mu_col = pick_col(mu_candidates)

            if e_col is None or mu_col is None:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(numeric_cols) >= 2:
                    e_col, mu_col = numeric_cols[0], numeric_cols[1]
                else:
                    e_col, mu_col = df.columns[0], df.columns[1]

            x = pd.to_numeric(df[e_col], errors="coerce").to_numpy()
            y = pd.to_numeric(df[mu_col], errors="coerce").to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if len(x) < 10: raise ValueError("Too few valid points.")
            return x.astype(float), y.astype(float)
    except Exception:
        pass

    arr = np.loadtxt(path)
    x, y = arr[:, 0], arr[:, 1]
    m = np.isfinite(x) & np.isfinite(y)
    return x[m].astype(float), y[m].astype(float)

# -----------------------------
# Signal processing
# -----------------------------
def smooth_moving_average(y, w=9):
    w = max(3, int(w) | 1)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def interpolate_uniform(x, y, n=2048):
    x_min, x_max = float(np.min(x)), float(np.max(x))
    xu = np.linspace(x_min, x_max, n)
    yu = np.interp(xu, x, y)
    return xu, yu

# -----------------------------
# Gaussian fitting
# -----------------------------
def gaussian(x, amp, cen, sig):
    return amp * np.exp(-0.5 * ((x - cen) / sig) ** 2)

def multi_gaussian(x, *params):
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i+1], params[i+2])
    return y

def first_major_peak_region(x, y, smooth_window=11, prominence=0.02):
    ys = smooth_moving_average(y, w=smooth_window)
    yr = np.max(ys) - np.min(ys)
    prom = max(prominence * yr, 1e-12)
    peaks, _ = find_peaks(ys, prominence=prom)
    pk = int(peaks[0]) if len(peaks) > 0 else int(np.argmax(ys))
    
    span = float(np.max(x) - np.min(x))
    half = max(0.03 * span, 20.0)
    lo, hi = x[pk] - half, x[pk] + half
    mask = (x >= lo) & (x <= hi)
    return pk, mask, ys

def fit_gaussians_to_peak(x, y, n_gauss=1):
    pk_idx, mask, _ = first_major_peak_region(x, y)
    x_fit, y_fit = x[mask], y[mask]
    baseline = float(np.min(y_fit))
    y0 = y_fit - baseline

    # Initial Guesses
    x0, amp0 = float(x_fit[np.argmax(y0)]), float(np.max(y0))
    w = float(np.max(x_fit) - np.min(x_fit))
    sig0 = max(w / (6 * n_gauss), 1e-6)
    
    p0 = []
    lower, upper = [], []
    for i in range(n_gauss):
        p0 += [amp0/n_gauss, x0 + (i*0.1*w), sig0]
        lower += [0.0, float(np.min(x_fit)), 1e-6]
        upper += [np.inf, float(np.max(x_fit)), w]

    try:
        popt, _ = curve_fit(lambda xx, *pp: multi_gaussian(xx, *pp), x_fit, y0, p0=p0, bounds=(lower, upper))
        y_pred = multi_gaussian(x_fit, *popt) + baseline
        rmse = float(np.sqrt(np.mean((y_pred - y_fit) ** 2)))
        return {"pk_index": pk_idx, "x_fit": x_fit, "y_fit": y_fit, "params": popt, "y_pred": y_pred, "rmse": rmse}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Fourier transform (FFT)
# -----------------------------
def compute_fft(x, y, n=4096, detrend=True, window=True):
    xu, yu = interpolate_uniform(x, y, n=n)
    if detrend: yu = yu - np.mean(yu)
    if window: yu = yu * np.hanning(len(yu))
    
    dx = float(xu[1] - xu[0])
    Y = np.fft.rfft(yu)
    freq = np.fft.rfftfreq(len(yu), d=dx)
    mag = np.abs(Y)
    return xu, yu, freq, mag

# -----------------------------
# Prompt parsing
# -----------------------------
def parse_user_intent(text: str):
    t = text.lower()
    wants_fit = any(k in t for k in ["fit", "gauss", "peak"])
    wants_fft = any(k in t for k in ["fourier", "fft", "transform"])
    
    n = None
    m = re.search(r"\b(\d+)\s*(gauss|peak)", t)
    if m: n = int(m.group(1))
    
    return {"fit": wants_fit, "fft": wants_fft, "n_gauss": n}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Dr. XAS AI Agent", layout="wide")
st.title("🧪 Dr. XAS AI Agent")

with st.sidebar:
    st.header("Database & Settings")
    data_folder = st.text_input("Spectra folder path", value="data")
    if "files" not in st.session_state or st.button("🔄 Scan folder"):
        st.session_state.files = list_spectra_files(data_folder) if os.path.isdir(data_folder) else []
    
    if st.button("🎲 Randomly load spectrum") and st.session_state.files:
        path = random.choice(st.session_state.files)
        st.session_state.x, st.session_state.y = load_spectrum(path)
        st.session_state.current_path = path

    default_gauss = st.slider("Default #Gaussians", 1, 4, 1)
    fft_n = st.selectbox("FFT points", [1024, 2048, 4096, 8192], index=2)

if "chat" not in st.session_state: st.session_state.chat = []

colA, colB = st.columns([1, 1])
with colA:
    if "x" in st.session_state:
        st.subheader("Current Spectrum")
        st.line_chart(pd.DataFrame({"Energy": st.session_state.x, "Mu": st.session_state.y}), x="Energy", y="Mu")
    else:
        st.info("Load a spectrum from the sidebar to begin.")

with colB:
    st.subheader("Chat")
    for role, msg in st.session_state.chat:
        with st.chat_message(role): st.markdown(msg)
    
    user_text = st.chat_input("Ex: 'Fit with 2 gaussians' or 'Compute FFT'")

if user_text:
    st.session_state.chat.append(("user", user_text))
    if "x" not in st.session_state:
        st.session_state.chat.append(("assistant", "Please load a spectrum first."))
    else:
        intent = parse_user_intent(user_text)
        res_msgs = []
        
        if intent["fit"]:
            n = intent["n_gauss"] or default_gauss
            fit = fit_gaussians_to_peak(st.session_state.x, st.session_state.y, n_gauss=n)
            if "error" in fit:
                res_msgs.append(f"Fit error: {fit['error']}")
            else:
                st.session_state.last_fit = pd.DataFrame({"Energy": fit["x_fit"], "Measured": fit["y_fit"], "Fit": fit["y_pred"]})
                res_msgs.append(f"✅ Gaussian fit complete (RMSE: {fit['rmse']:.4f}).")
        
        if intent["fft"]:
            xu, yu, freq, mag = compute_fft(st.session_state.x, st.session_state.y, n=fft_n)
            st.session_state.last_fft = pd.DataFrame({"Freq": freq, "Magnitude": mag})
            st.session_state.last_pre_fft = pd.DataFrame({"Energy": xu, "Signal": yu})
            res_msgs.append(f"✅ FFT calculated using {fft_n} points.")

        if not res_msgs: res_msgs.append("I can perform Gaussian fitting or FFT. Try 'Fit 1 peak' or 'Do FFT'.")
        st.session_state.chat.append(("assistant", "\n\n".join(res_msgs)))
    st.rerun()

# -----------------------------
# Results Display
# -----------------------------
st.divider()
out1, out2 = st.columns(2)
with out1:
    st.markdown("### Peak Fit Results")
    if "last_fit" in st.session_state:
        st.line_chart(st.session_state.last_fit, x="Energy")
with out2:
    st.markdown("### FFT Results")
    if "last_fft" in st.session_state:
        st.line_chart(st.session_state.last_fft, x="Freq", y="Magnitude")
