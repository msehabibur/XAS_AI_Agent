import os
import re
import random
import numpy as np
import pandas as pd
import streamlit as st
import openai
from ase.io import read
from st_py3mol import showmol
import py3mol
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# -----------------------------
# 1. CORE XAS MATH UTILITIES
# -----------------------------
def load_spectrum(path: str):
    try:
        df = pd.read_csv(path)
        # Simplified column inference for brevity
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        return x[m].astype(float), y[m].astype(float)
    except:
        arr = np.loadtxt(path)
        return arr[:, 0], arr[:, 1]

def compute_fft(x, y, n=2048):
    xu = np.linspace(np.min(x), np.max(x), n)
    yu = np.interp(xu, x, y)
    yu -= np.mean(yu)
    w = np.hanning(len(yu))
    Y = np.fft.rfft(yu * w)
    freq = np.fft.rfftfreq(len(yu), d=(xu[1] - xu[0]))
    return xu, yu, freq, np.abs(Y)

# -----------------------------
# 2. LLM AGENT INTERFACE
# -----------------------------
def ask_dr_xas(user_prompt, api_key, context_data):
    if not api_key:
        return "Please provide an OpenAI API Key in the sidebar to use the AI Assistant."
    
    client = openai.OpenAI(api_key=api_key)
    system_msg = (
        "You are Dr. XAS, an expert in X-ray Absorption Spectroscopy. "
        "Analyze the user's data (peaks, structural clusters, FFT) with scientific rigor. "
        "Context: " + str(context_data)
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error contacting GPT: {str(e)}"

# -----------------------------
# 3. UI LAYOUT & SIDEBAR
# -----------------------------
st.set_page_config(page_title="Dr. XAS AI Agent", layout="wide")
st.title("🧪 Dr. XAS: Advanced Spectroscopy & Cluster Analysis")

with st.sidebar:
    st.header("Settings & API")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Enter your sk-... key")
    
    st.divider()
    data_folder = st.text_input("Spectra folder", value="data")
    cif_folder = st.text_input("CIF folder", value="cif_files")
    
    if st.button("🔄 Scan & Load Random"):
        if os.path.isdir(data_folder):
            files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.csv', '.txt'))]
            if files:
                path = random.choice(files)
                st.session_state.x, st.session_state.y = load_spectrum(path)
                st.session_state.current_path = path

# -----------------------------
# 4. MAIN DASHBOARD
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Spectrum & FFT")
    if "x" in st.session_state:
        # Plot Raw
        df_plot = pd.DataFrame({"Energy": st.session_state.x, "Mu": st.session_state.y})
        st.line_chart(df_plot, x="Energy", y="Mu", height=250)
        
        # Auto-compute FFT
        xu, yu, freq, mag = compute_fft(st.session_state.x, st.session_state.y)
        st.write("**Fourier Transform (R-space equivalent)**")
        fft_df = pd.DataFrame({"Freq": freq, "Magnitude": mag})
        st.line_chart(fft_df, x="Freq", y="Magnitude", height=250)
    else:
        st.info("Load a spectrum from the sidebar.")

with col2:
    st.subheader("Cluster Structure (CIF)")
    if os.path.exists(cif_folder):
        cifs = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]
        if cifs:
            atoms = read(os.path.join(cif_folder, cifs[0]))
            # 3D Render
            view = py3mol.view(width=400, height=400)
            view.addModel(atoms.get_positions().tolist(), 'xyz') # Simplified for demo
            view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.1}})
            showmol(view, height=400, width=400)
            st.write(f"**Formula:** {atoms.get_chemical_formula()}")

# -----------------------------
# 5. INTEGRATED CHAT
# -----------------------------
st.divider()
st.subheader("💬 Chat with Dr. XAS")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the peaks or the local coordination environment..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare context for GPT
    context = {
        "spectrum_file": st.session_state.get("current_path", "None"),
        "formula": atoms.get_chemical_formula() if 'atoms' in locals() else "Unknown"
    }
    
    with st.chat_message("assistant"):
        response = ask_dr_xas(prompt, openai_key, context)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
