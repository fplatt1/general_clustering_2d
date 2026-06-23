import logging
import numpy as np
import streamlit as st
from plotly import graph_objs as go
import plotly.express as px
import re

# Ersetze den alten try-except Block komplett durch diese Zeile:
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
                # Schöne Beschriftung der Legende (Hintergrund vs. Neuron)
                ticktext=["Hintergrund" if l == -1 else f"Neuron {int(l)}" for l in unique_sorted],  
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


def plot_mean_spectra(mean_spectra, plot_labels, y_limit, unique_labels_order):
    fig = go.Figure()

    # Exakt gleiche Farb-Generierung wie in der Heatmap für Konsistenz
    sorted_labels = sorted(unique_labels_order)
    n_clusters = len(sorted_labels)
    
    viridis_samples = np.linspace(0, 1, n_clusters)
    current_colors = px.colors.sample_colorscale("Viridis", viridis_samples)
    label_to_color = {label: color for label, color in zip(sorted_labels, current_colors)}

    for spectrum, label_text in zip(mean_spectra, plot_labels):
        try:
            # Filtert, ob es sich um den Hintergrund oder ein Neuron handelt
            if "Hintergrund" in label_text:
                found_id = -1
            else:
                match = re.search(r"Neuron\s*(-?\d+)", label_text)
                found_id = int(match.group(1)) if match else None
                
            if found_id is not None:
                line_color = label_to_color.get(found_id, "black")
            else:
                line_color = "black"
                
        except (AttributeError, ValueError):
            line_color = "black" 

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
st.title("🔬 Universelle Raman-Karten Analyse (Reine PCA + SOM)")

# Sidebar für SOM-Parameter
st.sidebar.header("SOM-Einstellungen")
map_height = st.sidebar.slider("SOM Gitter-Höhe (m)", min_value=1, max_value=5, value=3)
map_width = st.sidebar.slider("SOM Gitter-Breite (n)", min_value=1, max_value=5, value=3)

# Hauptbereich für Datei-Upload
st.header("1. Raman-Karte hochladen")
uploaded_file = st.file_uploader("Wähle eine .mat-Datei (Witec)", type=["mat"])

if uploaded_file is None:
    st.info("Bitte eine `.mat`-Datei hochladen, um die automatisierte Analyse zu starten.")
    st.stop()
else:
    st.success(f"Datei '{uploaded_file.name}' erfolgreich geladen.")

# Lese die Bytes der Datei
try:
    file_bytes = uploaded_file.getvalue()
except Exception as e:
    logging.exception("Fehler beim Lesen der hochgeladenen Datei")
    st.error(f"Fehler beim Lesen der Datei: {e}")
    st.stop()

# Führe die Analyse automatisch bei Datei-Upload oder Parameter-Änderung aus
try:
    with st.spinner("Berechne automatische PCA und trainiere Self-Organizing Map..."):
        results = analyze(file_bytes, map_height, map_width)
except Exception as e:
    logging.exception("Fehler während der Analyse")
    results = {"success": False, "error": str(e)}
    st.error(f"Analyse fehlgeschlagen: {e}")

# --- Ergebnisse anzeigen ---
st.header("2. Ergebnisse")

# Zeige das Analyse-Log in einem ausklappbaren Bereich
with st.expander("Analyse-Log anzeigen (Terminal-Ausgabe)"):
    try:
        with open(LOG_FILE) as f:
            logs = f.read()
        st.text_area("Log", logs, height=300)
    except Exception:
        st.write("Kein Log-Protokoll verfügbar.")

# --- HAUPT-ERGEBNIS LOGIK ---
if results and results["success"]:
    st.success("Analyse erfolgreich abgeschlossen!")

    # 1. Cluster-Karte (Heatmap)
    st.subheader("Ortsaufgelöste Cluster-Karte")
    if results.get("cluster_map") is not None:
        fig_map = plot_cluster_map(
            results["cluster_map"], results["unique_labels"]
        )
        st.plotly_chart(fig_map)
    else:
        st.warning("Keine Cluster-Karte verfügbar.")

    # 2. Mittlere Spektren
    st.subheader("Mittlere Spektren der Neuronen")
    if results.get("mean_spectra") is not None:
        fig_spectra = plot_mean_spectra(
            results["mean_spectra"], results["plot_labels"], results["y_limit"], results["unique_labels"]
        )
        st.plotly_chart(fig_spectra)
    else:
        st.warning("Keine Spektren verfügbar.")
        
    st.info("💡 Hinweis: Die interaktive 3D-Visualisierung des PCA-Merkmalsraums wurde weiter oben nativ ausgegeben.")

elif results and not results.get("success", False):
    st.error(f"Analyse fehlgeschlagen. Fehler: {results.get('error', 'Unbekannter Fehler')}")