from pathlib import Path

import numpy as np
import streamlit as st
from bachelorarbeit.plot.spectrum_and_image import image_to_spectrum, spectrum_to_image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

uploaded_file = st.file_uploader("Choose a Witec file", type=["mat"])
load_sample = st.button("Load sample data")

data = None
if uploaded_file is not None:
    try:
        data = loadmat(uploaded_file)
        data = [v for k, v in data.items() if k.startswith("Struct")][0]
    except Exception as e:
        st.error(f"Fehler beim Laden der hochgeladenen Datei: {e}")
        st.stop()
elif load_sample:
    try:
        root = Path(__file__).parent.parent
        DATA = root / "data/P4-Raman-Scan-150x150um.mat"
        data = loadmat(DATA)
        data = data["Struct_ScanPiezo003SpecData1"]
    except Exception as e:
        st.error(f"Fehler beim Laden der Beispieldatei: {e}")
        st.stop()
else:
    st.info("Bitte eine Datei hochladen oder 'Load sample data' drücken.")
    st.stop()
size = np.append(data["imagesize"][0, 0][0], data["data"][0, 0].shape[1])
wavenumber = np.array(data["axisscale"][0, 0][1, 0][0])
data = data["data"][0, 0]

with st.sidebar:
    sigma = st.number_input(
        "Gaussian filter sigma",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    sti_or_its = st.selectbox(
        label="Type of Spectrum vs Image Plot",
        options=[
            "Spectrum to Image",
            "Image to Spectrum",
        ],
    )
if sigma > 0:
    data = gaussian_filter1d(data, sigma=sigma, axis=1)

match sti_or_its:
    case "Spectrum to Image":
        st.title("Spectrum to Image Plot")
        spectrum_to_image(data, wavenumber, size)
    case "Image to Spectrum":
        st.title("Image to Spectrum Plot")
        image_to_spectrum(data, wavenumber, size)
    case _:
        st.error("Unknown option")
        st.stop()
