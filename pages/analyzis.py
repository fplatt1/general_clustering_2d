from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from bachelorarbeit.plot.fwhm import explore_fwhm
from bachelorarbeit.plot.spectrum_and_image import image_to_spectrum, spectrum_to_image
from bachelorarbeit.reduction import fwhm_analyzis, pca_analyzis, pca_transformed
from plotly.express.colors import sample_colorscale
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import HDBSCAN, OPTICS, KMeans
from sklearn.mixture import GaussianMixture

uploaded_file = st.file_uploader("Choose a Witec file", type=["mat"])
# Button to explicitly load the bundled sample data (avoid loading at import-time)
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
    prominence = st.number_input(
        "Peak prominence",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
data = gaussian_filter1d(data, sigma=sigma, axis=1)

ignore = np.array([500.6, 512.0])
ignore_idx = np.abs(np.subtract(wavenumber[:, None], ignore[None, :]))
ignore_idx = np.argmin(ignore_idx, axis=0)
st.write(ignore_idx, wavenumber[ignore_idx])
fwhm_data = fwhm_analyzis(data, prominence, ignore=ignore_idx)
# fwhm_data = fwhm_analyzis(data, prominence, ignore=[134, 246])

if st.checkbox("Explore FWHM data", value=False):
    col1, col2 = st.columns(2)
    peak = col1.number_input(
        "Peak",
        min_value=0,
        max_value=fwhm_data.shape[1] - 1,
        value=0,
    )
    features = [
        "Localtion",
        "Height",
        # "FWHM",
    ]
    feature = col2.selectbox(
        label="Feature",
        options=np.arange(len(features)),
        format_func=lambda x: features[x],
    )
    peak_location = int(np.mean(fwhm_data[:, peak, 0]))
    st.write("Peak indices: ", *np.mean(fwhm_data[:, :, 0], axis=0).tolist())
    st.write(
        "Peak values : ", *wavenumber[np.mean(fwhm_data[:, :, 0], axis=0, dtype=int)]
    )
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=wavenumber))
    # fig.add_vline(x=peak_location, line_width=1, line_dash="dash")
    # st.plotly_chart(fig)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavenumber, y=np.mean(data, axis=0)))
    fig.add_vline(x=wavenumber[peak_location], line_width=1, line_dash="dash")
    fig.update_layout(xaxis=dict(title="Wavenumber", ticksuffix="cm⁻¹"))
    st.plotly_chart(fig)
    explore_fwhm(fwhm_data, size, peak, int(feature))

# fwhm_data = fwhm_data[:, :, 0]
fwhm_data = fwhm_data.reshape(fwhm_data.shape[0], -1)
with st.sidebar:
    s = np.min((size[-1], fwhm_data.shape[-1]))
    pca_n = st.number_input(
        "Number of PCA components",
        min_value=1,
        max_value=s,
        value=s // 2,
    )
    pca_n = int(pca_n)
pca_data = pca_analyzis(fwhm_data, pca_n)

with st.expander("Ratios"):
    # ratio = pca_var(data, n)
    ratio = pca_data.explained_variance_ratio_

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ratio))
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.cumsum(ratio)))
    fig.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig)

transformed = pca_transformed(data, pca_n)
transformed = np.reshape(transformed, (size[0], size[1], pca_n))

if st.checkbox("Show PCA channel", value=False):
    channel = st.number_input(
        "Channel",
        min_value=0,
        max_value=pca_n - 1,
        value=0,
    )
    target = transformed[:, :, channel]
    target = np.clip(
        a=target,
        a_min=np.percentile(target, 1),
        a_max=np.percentile(target, 99),
    )
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=np.rot90(target), colorscale="Viridis"))
    fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig)

with st.sidebar:
    algorithm = st.selectbox(
        "Clustering algorithm",
        options=[None, "KMeans", "GMM", "HDBSCAN", "OPTICS"],
    )
    # n_clusters = 1
    # if algorithm in ["KMeans", "GMM"]:
    n_clusters = st.number_input(
        "Number of clusters",
        min_value=2,
        max_value=20,
        value=4,
    )
with st.spinner("Clustering..."):
    match algorithm:
        case "OPTICS":
            clusters = OPTICS(
                min_cluster_size=n_clusters,
            ).fit(transformed.reshape(-1, pca_n))
            labels = clusters.labels_
        case "HDBSCAN":
            clusters = HDBSCAN(
                min_cluster_size=n_clusters,
            ).fit(transformed.reshape(-1, pca_n))
            labels = clusters.labels_
        case "KMeans":
            clusters = KMeans(
                n_clusters=n_clusters,
                random_state=0,
                n_init="auto",
            ).fit(transformed.reshape(-1, pca_n))
            labels = clusters.labels_
        case "GMM":
            clusters = GaussianMixture(
                n_components=n_clusters,
                random_state=0,
            ).fit(transformed.reshape(-1, pca_n))
            labels = clusters.predict(transformed.reshape(-1, pca_n))
        case _:
            st.error("Unknown algorithm")
            st.stop()
labels = np.array(labels)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        z=np.rot90(np.reshape(np.array(labels), size[0:2])),
        colorscale="Viridis",
    )
)
fig.update_layout(height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
st.plotly_chart(fig)

# col1, col2 = st.columns(2)
# y = col1.number_input("X", min_value=0, max_value=size[0] - 1, value=size[0] // 2)
# x = col2.number_input("Y", min_value=0, max_value=size[1] - 1, value=size[1] // 2)

fig = go.Figure()
# fig.add_trace(
#     go.Scatter(
#         x=wavenumber,
#         y=data.reshape(size[0], size[1], -1)[x, y, :],
#         mode="lines",
#         name=f"({x}, {y})",
#     )
# )
for i in range(n_clusters):
    mask = labels == i
    centroid = np.mean(data.reshape(-1, size[-1])[mask, :], axis=0)
    peaks, _ = find_peaks(centroid, prominence=prominence)
    fig.add_trace(
        go.Scatter(
            x=wavenumber,
            y=centroid,
            mode="lines",
            name=f"Centroid {i}",
            line=dict(color=sample_colorscale("Viridis", i / (n_clusters - 1))[0]),
        )
    )
fig.update_layout(
    xaxis=dict(
        title="Wavenumber",
        ticksuffix="cm⁻¹",
    ),
    yaxis=dict(type="log"),
)
st.plotly_chart(fig)

three_vs_two = st.toggle("3D vs 2D PCA plot", value=True)
dim_z = 2
with st.form("my_form"):
    col1, col2, col3 = st.columns(3)
    dim_x = col1.number_input("PCA X axis", min_value=0, max_value=pca_n - 1, value=0)
    dim_y = col2.number_input("PCA Y axis", min_value=0, max_value=pca_n - 1, value=1)
    if not three_vs_two:
        dim_z = col3.number_input(
            "PCA Z axis",
            min_value=0,
            max_value=pca_n - 1,
            value=2,
        )
    submitted = st.form_submit_button("Submit")

t = transformed.reshape(-1, pca_n)
# random_sampling = 10000
random_sampling = int(np.min((t.shape[0] * 0.1, 10_000)))
st.write(random_sampling)
random_sampling = np.random.choice(
    t.shape[0],
    size=random_sampling,
    replace=False,
)
fig = go.Figure()
if not three_vs_two:
    fig.add_trace(
        go.Scatter3d(
            x=t[random_sampling, dim_x].reshape(-1),
            y=t[random_sampling, dim_y].reshape(-1),
            z=t[random_sampling, dim_z].reshape(-1),
            mode="markers",
            marker=dict(
                size=2,
                color=labels[random_sampling],
                colorscale="Viridis",
                opacity=0.8,
            ),
            text=labels[random_sampling],
        )
    )
else:
    fig.add_trace(
        go.Scatter(
            x=t[random_sampling, dim_x].reshape(-1),
            y=t[random_sampling, dim_y].reshape(-1),
            mode="markers",
            marker=dict(
                size=3,
                color=labels[random_sampling],
                colorscale="Viridis",
                opacity=0.8,
            ),
            text=labels[random_sampling],
        )
    )
fig.update_layout(
    height=1000,
)
st.plotly_chart(fig)
