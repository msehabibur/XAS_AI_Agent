import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from openai import OpenAI

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Dr. XAS AI Agent", layout="wide")

# Sidebar for Logo and API Key
with st.sidebar:
    st.image("drxas_logo_big.png", use_container_width=True)
    st.title("Dr. XAS Controls")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your key to enable AI features")
    st.divider()
    st.info("Tasks: \n1. Gaussian Fitting\n2. Fourier Transform")

# Initialize OpenAI Client
client = OpenAI(api_key=api_key) if api_key else None

# --- 2. DATA PROCESSING FUNCTIONS ---
def gaussian(x, amp, cen, wid):
    """Simple Gaussian function: A * exp(-(x-xc)^2 / (2*w^2))"""
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def perform_gaussian_fit(x, y):
    """Performs a fit on the first major peak."""
    peak_idx = np.argmax(y)  # Finds the highest point as the first major peak
    initial_guess = [y[peak_idx], x[peak_idx], 1.0] # [amp, center, width]
    popt, _ = curve_fit(gaussian, x, y, p0=initial_guess)
    return popt

def perform_fourier_transform(y):
    """Performs a basic Fast Fourier Transform (FFT)."""
    yf = np.fft.fft(y)
    return np.abs(yf[:len(yf)//2])

# --- 3. SESSION STATE & DATA LOADING ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mock data loader (Replace with your actual database reading logic)
@st.cache_data
def load_data():
    # Example: Generating synthetic XAS-like data
    x = np.linspace(7100, 7200, 300)
    # Background + Edge + Peak
    y = 0.5 + 0.5 * np.tanh((x - 7115)/5) + 1.2 * np.exp(-(x - 7125)**2 / 10)
    return pd.DataFrame({"Energy": x, "Intensity": y})

df = load_data()

# --- 4. CHAT INTERFACE & LOGIC ---
st.title("Dr. XAS AI Agent 🔬")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "fig" in message:
            st.plotly_chart(message["fig"])

# User Input
if prompt := st.chat_input("Ask Dr. XAS (e.g., 'Fit the peak' or 'Do a Fourier transform')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not client:
        st.warning("Please provide an OpenAI API Key in the sidebar.")
    else:
        # AI Logic: Determine if user wants a Fit or FFT
        response_text = ""
        fig = None
        
        # Simple keyword routing (could be replaced with a system prompt for intent classification)
        low_prompt = prompt.lower()
        
        with st.chat_message("assistant"):
            if "fit" in low_prompt or "gaussian" in low_prompt:
                popt = perform_gaussian_fit(df["Energy"].values, df["Intensity"].values)
                amp, cen, wid = popt
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Energy"], y=df["Intensity"], name="Original Data"))
                fig.add_trace(go.Scatter(x=df["Energy"], y=gaussian(df["Energy"], *popt), name="Gaussian Fit", line=dict(color='red', dash='dash')))
                fig.update_layout(title="Peak Fitting Results", xaxis_title="Energy (eV)", ydata_title="Intensity")
                st.plotly_chart(fig)
                
                response_text = f"I've completed the Gaussian fit on the first major peak. \n\n**Parameters:**\n- Position: {cen:.2f} eV\n- Amplitude: {amp:.2f}\n- Width: {wid:.2f}"
                st.markdown(response_text)

            elif "fourier" in low_prompt or "transform" in low_prompt:
                mag = perform_fourier_transform(df["Intensity"].values)
                fig = go.Figure(data=go.Scatter(y=mag, name="FFT Magnitude"))
                fig.update_layout(title="Fourier Transform (R-space)", xaxis_title="Index", yaxis_title="Magnitude")
                st.plotly_chart(fig)
                
                response_text = "I have performed the Fourier transform of the spectrum. You can see the periodicity/scattering path contributions in the R-space plot above."
                st.markdown(response_text)
            
            else:
                # Regular Chat fallback
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are Dr. XAS, an expert in X-ray absorption spectroscopy."}] + 
                             [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                )
                response_text = completion.choices[0].message.content
                st.markdown(response_text)

        # Save to memory
        msg_data = {"role": "assistant", "content": response_text}
        if fig: msg_data["fig"] = fig
        st.session_state.messages.append(msg_data)
