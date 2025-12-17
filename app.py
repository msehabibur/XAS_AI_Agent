import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from openai import OpenAI

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Dr. XAS AI Agent", layout="wide")

# Sidebar for Logo and API
with st.sidebar:
    try:
        st.image("drxas_logo_big.png", use_container_width=True)
    except:
        st.warning("Logo file 'drxas_logo_big.png' not found.")
    
    st.title("Control Panel")
    api_key = st.text_input("OpenAI API Key", type="password")
    client = OpenAI(api_key=api_key) if api_key else None
    
    st.divider()
    st.write("**Database Status:**")
    try:
        # Assuming your CSV has columns 'energy' and 'intensity'
        df = pd.read_csv("xas_data.csv")
        st.success("xas_data.csv loaded successfully!")
    except FileNotFoundError:
        st.error("xas_data.csv not found. Using dummy data for now.")
        x = np.linspace(7000, 7200, 500)
        y = np.exp(-(x - 7120)**2 / 40) + 0.5 * np.exp(-(x - 7150)**2 / 100) + 0.1 * np.random.normal(size=500)
        df = pd.DataFrame({"energy": x, "intensity": y})

# --- 2. MATH FUNCTIONS ---
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def perform_fit(x, y):
    # Initial guess: max height, x-coord of max height, and a standard width
    peak_idx = np.argmax(y)
    p0 = [max(y), x[peak_idx], 1.0]
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    return popt

# --- 3. CHAT INTERFACE & MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Dr. XAS AI Agent")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot" in message:
            st.pyplot(message["plot"])

# User prompt
if prompt := st.chat_input("How can I help with your XAS data?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not client:
        st.error("Please enter your OpenAI API Key in the sidebar to proceed.")
    else:
        with st.chat_message("assistant"):
            low_prompt = prompt.lower()
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # TASK: GAUSSIAN FITTING
            if "fit" in low_prompt or "gaussian" in low_prompt:
                x_val, y_val = df["energy"].values, df["intensity"].values
                popt = perform_fit(x_val, y_val)
                
                ax.plot(x_val, y_val, 'k-', label='Original Data', alpha=0.7)
                ax.plot(x_val, gaussian(x_val, *popt), 'r--', label='Gaussian Fit')
                ax.set_title("XAS Peak Fitting")
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("Intensity")
                ax.legend()
                st.pyplot(fig)
                
                res_text = (f"I've performed a Gaussian fit on the primary peak.\n\n"
                            f"**Parameters:**\n- Position: {popt[1]:.2f}\n- Amplitude: {popt[0]:.2f}\n- Width: {popt[2]:.2f}")
                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text, "plot": fig})

            # TASK: FOURIER TRANSFORM
            elif "fourier" in low_prompt or "transform" in low_prompt:
                y_val = df["intensity"].values
                fft_res = np.abs(np.fft.fft(y_val))[:len(y_val)//2]
                
                ax.plot(fft_res, color='blue')
                ax.set_title("Fourier Transform (R-space Magnitude)")
                ax.set_xlabel("k (approx)")
                ax.set_ylabel("|FT|")
                st.pyplot(fig)
                
                res_text = "Fourier Transform complete. The plot above shows the magnitude in R-space."
                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text, "plot": fig})

            # TASK: GENERAL CHAT (Memory included)
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                )
                res_text = response.choices[0].message.content
                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})
