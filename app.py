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
    Loads a 2-column spectrum:
      - CSV/TXT/DAT
      - header optional
      - tries to infer which columns correspond to energy and mu
    Returns (x, y) as float arrays.
    """
    # Try CSV with header
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            cols = [c.lower().strip() for c in df.columns]
            df.columns = cols

            # pick likely names
            energy_candidates = ["energy", "e", "en", "ev"]
            mu_candidates = ["mu", "i", "intensity", "xmu"]

            def pick_col(cands):
                for c in cands:
                    if c in df.columns:
                        return c
                return None

            e_col = pick_col(energy_candidates)
            mu_col = pick_col(mu_candidates)

            if e_col is None or mu_col is None:
                # fallback: first 2 numeric columns
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(numeric_cols) >= 2:
                    e_col, mu_col = numeric_cols[0], numeric_cols[1]
                else:
                    # fallback: just first 2 columns
                    e_col, mu_col = df.columns[0], df.columns[1]

            x = pd.to_numeric(df[e_col], errors="coerce").to_numpy()
            y = pd.to_numeric(df[mu_col], errors="coerce").to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if len(x) < 10:
                raise ValueError("Too few valid points.")
            return x.astype(float), y.astype(float)
    except Exception:
        pass

    # Try whitespace-delimited with no header
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Could not parse 2-column data from: {path}")
    x, y = arr[:, 0], arr[:, 1]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    return x.astype(float), y.astype(float)

# -----------------------------
# Signal processing
# -----------------------------
def smooth_moving_average(y, w=9):
    w = max(3, int(w) | 1)  # odd >= 3
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
    # params = [amp1, cen1, sig1, amp2, cen2, sig2, ...]
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        y += gaussian(x, params[i], params[i+1], params[i+2])
    return y

def first_major_peak_region(x, y, smooth_window=11, prominence=0.02):
    """
    Find "first major peak" in x-order:
      - smooth
      - find peaks
      - pick the first peak with sufficient prominence
      - return a fitting window around that peak
    """
    ys = smooth_moving_average(y, w=smooth_window)

    # scale prominence relative to y-range
    yr = np.max(ys) - np.min(ys)
    prom = max(prominence * yr, 1e-12)

    peaks, props = find_peaks(ys, prominence=prom)
    if len(peaks) == 0:
        # fallback: global max
        pk = int(np.argmax(ys))
    else:
        pk = int(peaks[0])

    xpk = x[pk]
    # pick a local window: e.g. +/- 3% of total x-span or at least some points
    span = float(np.max(x) - np.min(x))
    half = max(0.03 * span, 20.0)  # 20 eV default-ish; adjust if needed
    lo, hi = xpk - half, xpk + half

    m = (x >= lo) & (x <= hi)
    if m.sum() < 30:
        # expand if too narrow
        half = max(0.06 * span, 40.0)
        lo, hi = xpk - half, xpk + half
        m = (x >= lo) & (x <= hi)

    return pk, m, ys

def guess_initial_params(x_fit, y_fit, n_gauss=1):
    """
    Build simple initial guesses:
      - centers spaced around max
      - amplitudes split
      - sigmas as fraction of window
    """
    x0 = float(x_fit[np.argmax(y_fit)])
    amp0 = float(np.max(y_fit) - np.min(y_fit))
    w = float(np.max(x_fit) - np.min(x_fit))
    sig0 = max(w / (6 * n_gauss), 1e-6)

    params = []
    if n_gauss == 1:
        params = [amp0, x0, sig0]
    else:
        # spread centers slightly around x0
        offsets = np.linspace(-0.2*w, 0.2*w, n_gauss)
        for i in range(n_gauss):
            params += [amp0 / n_gauss, x0 + offsets[i], sig0]
    return params

def fit_gaussians_to_peak(x, y, n_gauss=1):
    pk_idx, mask, y_smooth = first_major_peak_region(x, y)
    x_fit = x[mask]
    y_fit = y[mask]

    # baseline: subtract local minimum (simple)
    baseline = float(np.min(y_fit))
    y0 = y_fit - baseline

    p0 = guess_initial_params(x_fit, y0, n_gauss=n_gauss)

    # bounds: amplitude>=0, sigma>0; centers within fit window
    lower = []
    upper = []
    for i in range(n_gauss):
        lower += [0.0, float(np.min(x_fit)), 1e-6]
        upper += [np.inf, float(np.max(x_fit)), float(np.max(x_fit) - np.min(x_fit))]
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    try:
        popt, pcov = curve_fit(
            lambda xx, *pp: multi_gaussian(xx, *pp),
            x_fit,
            y0,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000
        )
        y_pred = multi_gaussian(x_fit, *popt) + baseline
        rmse = float(np.sqrt(np.mean((y_pred - y_fit) ** 2)))
        return {
            "pk_index": pk_idx,
            "fit_mask": mask,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "baseline": baseline,
            "params": popt,
            "cov": pcov,
            "y_pred": y_pred,
            "rmse": rmse
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Fourier transform (basic FFT)
# -----------------------------
def compute_fft(x, y, n=4096, detrend=True, window=True):
    """
    Basic FFT of y(x) after interpolating to a uniform x-grid.
    Returns frequency axis (in 1/x units) and magnitude spectrum.
    """
    xu, yu = interpolate_uniform(x, y, n=n)

    if detrend:
        yu = yu - np.mean(yu)

    if window:
        w = np.hanning(len(yu))
        yu = yu * w

    dx = float(xu[1] - xu[0])
    Y = np.fft.rfft(yu)
    freq = np.fft.rfftfreq(len(yu), d=dx)
    mag = np.abs(Y)
    return xu, yu, freq, mag

# -----------------------------
# Prompt parsing (the "agent")
# -----------------------------
def parse_user_intent(text: str):
    t = text.lower()

    # detect task
    wants_fit = any(k in t for k in ["fit", "fitting", "gaussian", "gauss", "peak"])
    wants_fft = any(k in t for k in ["fourier", "fft", "transform"])

    # detect how many gaussians requested
    # examples: "use 2 gaussians", "two gaussians", "3 peaks"
    n = None
    m = re.search(r"\b(\d+)\s*(gauss|gaussian|gaussians|peaks|peak)\b", t)
    if m:
        n = int(m.group(1))
    else:
        words = {"one": 1, "two": 2, "three": 3, "four": 4}
        for w, val in words.items():
            if re.search(rf"\b{w}\s*(gauss|gaussian|gaussians|peaks|peak)\b", t):
                n = val
                break

    # if user asked "multiple" assume 2 by default
    if n is None and "multiple" in t and wants_fit:
        n = 2

    return {
        "fit": bool(wants_fit) and not bool(wants_fft) or ("fit" in t and wants_fit),
        "fft": bool(wants_fft) and not bool(wants_fit) or ("fft" in t or "fourier" in t),
        "n_gauss": n
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Dr. XAS AI Agent (Basic)", layout="wide")
st.title("🧪 Dr. XAS AI Agent (Basic Web App)")

with st.sidebar:
    st.header("Database")
    data_folder = st.text_input("Spectra folder path", value="data")
    st.caption("Put CSV/TXT/DAT spectra in this folder. Each file should have 2 columns: energy and mu/intensity.")

    if "files" not in st.session_state or st.button("🔄 Scan folder"):
        if os.path.isdir(data_folder):
            st.session_state.files = list_spectra_files(data_folder)
        else:
            st.session_state.files = []

    files = st.session_state.get("files", [])
    st.write(f"Found **{len(files)}** spectra files.")

    if st.button("🎲 Randomly load a spectrum"):
        if len(files) == 0:
            st.warning("No spectra files found. Check the folder path.")
        else:
            path = random.choice(files)
            x, y = load_spectrum(path)
            st.session_state.current_path = path
            st.session_state.x = x
            st.session_state.y = y

    st.divider()
    st.header("Fit/FFT Settings")
    default_gauss = st.slider("Default #Gaussians (if not specified in prompt)", 1, 4, 1)
    fft_n = st.selectbox("FFT points (interpolation)", [1024, 2048, 4096, 8192], index=2)
    smooth_w = st.slider("Peak-detection smoothing window", 5, 41, 11, step=2)

# init chat
if "chat" not in st.session_state:
    st.session_state.chat = []

# show currently loaded spectrum
colA, colB = st.columns([1.1, 0.9])

with colA:
    st.subheader("Current Spectrum")
    if "x" in st.session_state and "y" in st.session_state:
        st.write(f"**Loaded:** `{st.session_state.current_path}`")
        df_plot = pd.DataFrame({"Energy": st.session_state.x, "Mu": st.session_state.y})
        st.line_chart(df_plot, x="Energy", y="Mu", height=320)
    else:
        st.info("Click **Randomly load a spectrum** in the sidebar (or scan folder first).")

with colB:
    st.subheader("Chat")
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    user_text = st.chat_input("Ask: 'Fit first peak with 2 Gaussians' or 'Do Fourier transform' ...")

if user_text:
    st.session_state.chat.append(("user", user_text))

    # Ensure we have a spectrum loaded
    if "x" no
