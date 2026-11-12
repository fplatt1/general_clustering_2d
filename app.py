import logging
import numpy as np
import streamlit as st
from plotly import graph_objs as go
import plotly.express as px

from funktionen_streamlit import run_pca_som_analysis

LOG_FILE = "log.txt"
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
)


@st.cache_data
def analyze(file_bytes, map_height: int, map_width: int):
    # Führt die reine, universelle PCA + SOM Analyse aus
    return run_pca_som_analysis(file_bytes, map_height=map_height, map_width=map_width)


def plot_cluster_map(cluster_map, unique_labels):
    # 1. Dimensionen & Basis-Setup
    num_rows, num_cols = cluster_map.shape
    aspect_ratio = num_cols / num_rows
    base_height = 700
    calculated_width = int(base_height * aspect_ratio) + 120

    # 2. Diskrete Farben aus Viridis generieren
    unique_sorted = sorted(unique_labels)
    n_clusters = len(unique_sorted)
    
    viridis_samples = np.linspace(0, 1, n_clusters)
    current_colors = px.colors.sample_colorscale("Viridis", viridis_samples)
    
    discrete_colorscale = []
    step = 1 / n_clusters
    for i, color in enumerate(current_colors):
        discrete_colorscale.append([i * step, color])       
        discrete_colorscale.append([(i + 1) * step, color]) 

    # Mapping der original Labels auf 0 bis n-1 für die Darstellung
    label_to_idx = {label: i for i, label in enumerate(unique_sorted)}
    mapped_cluster_map = np.vectorize(label_to_idx.get)(cluster_map)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(mapped_cluster_map), 
            colorscale=discrete_colorscale,
            zmin=0,
            zmax=n_clusters, 
            colorbar=dict(
                title="Cluster ID",
                title_side="top",
                title_font=dict(size=28),
                tickfont=dict(size=20),
                tickmode='array',
                tickvals=[x + 0.5 for x in range(n_clusters)], 
                # Legend labels (Background vs. Neuron)
                ticktext=["Background" if l == -1 else f"Neuron {int(l)}" for l in unique_sorted],  
                xpad=0,
                thickness=40,
                len=1.0
            )
        )
    )
    
    # Layout anpassen
    fig.update_layout(
        height=base_height,
        width=calculated_width, 
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(
            range=[0, num_cols - 1],
            constrain="domain",
            showticklabels=False, ticks="", showgrid=False, zeroline=False
        ),
        yaxis=dict(
            scaleanchor="x", scaleratio=1,
            range=[0, num_rows - 1],
            constrain="domain",
            showticklabels=False, ticks="", showgrid=False, zeroline=False
        )
    )
    return fig


def plot_mean_spectra(mean_spectra, plot_labels, label_values, y_limit, unique_labels_order):
    fig = go.Figure()

    # Exakt gleiche Farb-Generierung wie in der Heatmap für Konsistenz
    sorted_labels = sorted(unique_labels_order)
    n_clusters = len(sorted_labels)

    viridis_samples = np.linspace(0, 1, n_clusters)
    current_colors = px.colors.sample_colorscale("Viridis", viridis_samples)
    label_to_color = {label: color for label, color in zip(sorted_labels, current_colors)}

    for spectrum, label_text, label_value in zip(mean_spectra, plot_labels, label_values):
        # Gleiche Farben wie in der Heatmap, direkt über den Label-Wert gekoppelt
        line_color = label_to_color.get(label_value, "black")
        padded_label = label_text + "      " 
        
        fig.add_trace(
            go.Scatter(
                x=spectrum.spectral_axis,
                y=spectrum.spectral_data,
                name=padded_label,
                line=dict(width=3, color=line_color) 
            )
        )
    
    fig.update_layout(
        xaxis=dict(
            title="Raman Shift (cm⁻¹)",
            title_font=dict(size=20),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Intensity (a.u.)",
            range=[0, y_limit],
            title_font=dict(size=20),
            tickfont=dict(size=16)
        ),
        legend=dict(
            font=dict(size=20),
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=20, t=40, b=360),
        width=1000,
        height=700
    )
    return fig

# --- Streamlit Hauptanwendung ---

st.set_page_config(layout="wide")
st.title("🔬 Universal Raman Map Analysis (PCA + SOM)")

# Sidebar for SOM parameters
st.sidebar.header("SOM Settings")
map_height = st.sidebar.slider("SOM grid height (m)", min_value=1, max_value=5, value=3)
map_width = st.sidebar.slider("SOM grid width (n)", min_value=1, max_value=5, value=3)

# Main area for file upload
st.header("1. Upload Raman Map")
uploaded_file = st.file_uploader("Select a .mat file (WITec)", type=["mat"])

if uploaded_file is None:
    st.info("Please upload a `.mat` file to start the automated analysis.")
    st.stop()
else:
    st.success(f"File '{uploaded_file.name}' loaded successfully.")

# Lese die Bytes der Datei
try:
    file_bytes = uploaded_file.getvalue()
except Exception as e:
    logging.exception("Error reading the uploaded file")
    st.error(f"Error reading file: {e}")
    st.stop()

# Führe die Analyse automatisch bei Datei-Upload oder Parameter-Änderung aus
try:
    with st.spinner("Computing PCA and training Self-Organizing Map..."):
        results = analyze(file_bytes, map_height, map_width)
except Exception as e:
    logging.exception("Error during analysis")
    results = {"success": False, "error": str(e)}
    st.error(f"Analysis failed: {e}")

# --- Show results ---
st.header("2. Results")

# Show the analysis log in a collapsible section
with st.expander("Show analysis log"):
    try:
        with open(LOG_FILE) as f:
            logs = f.read()
        st.text_area("Log", logs, height=300)
    except Exception:
        st.write("No log available.")

# --- MAIN RESULTS LOGIC ---
if results and results["success"]:
    st.success("Analysis completed successfully!")

    # 1. 3D PCA feature space
    st.subheader("3D PCA Feature Space")
    if results.get("pca_fig") is not None:
        st.plotly_chart(results["pca_fig"], use_container_width=True)
    else:
        st.warning("No 3D PCA plot available.")

    # 2. Cluster map (heatmap)
    st.subheader("Spatial Cluster Map")
    if results.get("cluster_map") is not None:
        fig_map = plot_cluster_map(
            results["cluster_map"], results["unique_labels"]
        )
        st.plotly_chart(fig_map)
    else:
        st.warning("No cluster map available.")

    # 3. Mean spectra
    st.subheader("Mean Spectra of Neurons")
    if results.get("mean_spectra") is not None:
        fig_spectra = plot_mean_spectra(
            results["mean_spectra"],
            results["plot_labels"],
            results["unique_labels"],
            results["y_limit"],
            results["unique_labels"],
        )
        st.plotly_chart(fig_spectra)
    else:
        st.warning("No spectra available.")

elif results and not results.get("success", False):
    st.error(f"Analysis failed. Error: {results.get('error', 'Unknown error')}")
