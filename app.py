import os
import json
import random
import re
import numpy as np
import pandas as pd
import streamlit as st

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from openai import OpenAI

# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(page_title="Dr. XAS AI Agent", layout="wide")
st.title("🧪 Dr. XAS AI Agent")

# ============================================================
# Utilities: spectrum loading
# ============================================================
SUPPORTED_EXTS = (".csv", ".txt", ".dat")

def list_spectra_files(folder):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(SUPPORTED_EXTS):
                files.append(os.path.join(root, n))
    return sorted(files)

def load_spectrum(path):
    try:
        df = pd.read_csv(path)
        if df.shape[1] >= 2:
            x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    except Exception:
        arr = np.loadtxt(path)
        x, y = arr[:, 0], arr[:, 1]

    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

# ============================================================
# Signal processing helpers
# ============================================================
def smooth(y, w=11):
    w = max(3, int(w) | 1)
    return np.convolve(y, np.ones(w)/w, mode="same")

# ============================================================
# Gaussian fitting
# ============================================================
def gaussian(x, a, x0, s):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2)

def multi_gaussian(x, *p):
    y = np.zeros_like(x)
    for i in range(0, len(p), 3):
        y += gaussian(x, p[i], p[i+1], p[i+2])
    return y

def fit_first_peak(x, y, n_gauss=1):
    ys = smooth(y, 11)
    peaks, _ = find_peaks(ys, prominence=0.05*(ys.max()-ys.min()))
    pk = peaks[0] if len(peaks) else np.argmax(ys)

    span = (x.max() - x.min()) * 0.05
    mask = (x > x[pk]-span) & (x < x[pk]+span)

    xfit, yfit = x[mask], y[mask]
    base = yfit.min()
    yfit = yfit - base

    p0 = []
    for i in range(n_gauss):
        p0 += [yfit.max()/n_gauss, x[pk], span/5]

    popt, _ = curve_fit(multi_gaussian, xfit, yfit, p0=p0, maxfev=20000)
    ypred = multi_gaussian(xfit, *popt) + base

    return xfit, yfit + base, ypred, popt

# ============================================================
# Fourier transform
# ============================================================
def compute_fft(x, y, n=4096):
    xu = np.linspace(x.min(), x.max(), n)
    yu = np.interp(xu, x, y)
    yu -= yu.mean()

    Y = np.fft.rfft(yu * np.hanning(len(yu)))
    freq = np.fft.rfftfreq(len(yu), xu[1]-xu[0])
    return freq, np.abs(Y)

# ============================================================
# GPT ROUTER
# ============================================================
ROUTER_SCHEMA = {
    "name": "xas_router",
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["fit", "fft", "both", "none"]},
            "n_gauss": {"type": "integer", "minimum": 1, "maximum": 4},
            "explanation": {"type": "string"}
        },
        "required": ["action", "n_gauss", "explanation"]
    }
}

def gpt_route(prompt, model, api_key):
    client = OpenAI(api_key="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")
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content":
             "You are Dr. XAS AI Agent. Decide analysis task for XAS data."},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": ROUTER_SCHEMA,
                "strict": True
            }
        },
    )
    return json.loads(resp.output_text)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Settings")

    data_dir = st.text_input("Spectra folder", "data")
    files = list_spectra_files(data_dir)
    st.write(f"📁 {len(files)} spectra found")

    if st.button("🎲 Load random spectrum"):
        if files:
            path = random.choice(files)
            st.session_state["x"], st.session_state["y"] = load_spectrum(path)
            st.session_state["path"] = path

    st.divider()
    use_gpt = st.toggle("Use GPT routing", value=True)
    api_key_input = st.text_input("OpenAI API key", type="password")
    model_name = st.text_input("Model", "gpt-4o-mini")

# ============================================================
# Chat
# ============================================================
if "chat" not in st.session_state:
    st.session_state.chat = []

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Spectrum")
    if "x" in st.session_state:
        df = pd.DataFrame({"Energy": st.session_state.x, "Mu": st.session_state.y})
        st.line_chart(df, x="Energy", y="Mu", height=320)
        st.caption(st.session_state.path)
    else:
        st.info("Load a spectrum first")

with right:
    st.subheader("Chat")
    for r, m in st.session_state.chat:
        with st.chat_message(r):
            st.markdown(m)

    user_msg = st.chat_input("e.g. 'Fit first peak with 2 Gaussians'")

# ============================================================
# Handle user message
# ============================================================
if user_msg:
    st.session_state.chat.append(("user", user_msg))

    if "x" not in st.session_state:
        st.session_state.chat.append(("assistant", "Please load a spectrum first."))
        st.rerun()

    # GPT or fallback routing
    try:
        if use_gpt:
            key = api_key_input or os.getenv("OPENAI_API_KEY")
            route = gpt_route(user_msg, model_name, key)
            action = route["action"]
            n_gauss = route["n_gauss"]
            note = f"🧠 GPT: {route['explanation']}"
        else:
            raise RuntimeError("GPT disabled")

    except Exception as e:
        txt = user_msg.lower()
        action = "fit" if "fit" in txt else "fft" if "fft" in txt else "none"
        n_gauss = 2 if "two" in txt or "2" in txt else 1
        note = f"⚠️ Local fallback routing ({e})"

    outputs = [note]

    x, y = st.session_state.x, st.session_state.y

    if action in ["fit", "both"]:
        xfit, yfit, ypred, params = fit_first_peak(x, y, n_gauss)
        df_fit = pd.DataFrame({"Energy": xfit, "Data": yfit, "Fit": ypred})
        st.session_state["fit_plot"] = df_fit
        outputs.append(f"Gaussian fit done with {n_gauss} Gaussian(s).")

    if action in ["fft", "both"]:
        freq, mag = compute_fft(x, y)
        st.session_state["fft_plot"] = pd.DataFrame(
            {"Frequency": freq, "Magnitude": mag}
        )
        outputs.append("Fourier transform completed.")

    if action == "none":
        outputs.append("I can do Gaussian fitting or Fourier transform.")

    st.session_state.chat.append(("assistant", "\n\n".join(outputs)))
    st.rerun()

# ============================================================
# Output plots
# ============================================================
st.divider()
st.subheader("Results")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Peak Fit")
    if "fit_plot" in st.session_state:
        st.line_chart(st.session_state.fit_plot, x="Energy")
    else:
        st.caption("No fit yet")

with c2:
    st.markdown("### Fourier Transform")
    if "fft_plot" in st.session_state:
        st.line_chart(st.session_state.fft_plot, x="Frequency", y="Magnitude")
    else:
        st.caption("No FFT yet")
