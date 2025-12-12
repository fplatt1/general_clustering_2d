import logging

import numpy as np
import streamlit as st
from plotly import graph_objs as go
import plotly.express as px
import re

try:
    from funktionen_streamlit import run_pca_dbscan_analysis, run_feature_engineering_k_mean_analysis, run_feature_engineering_som_analysis
except ImportError:
    st.error(
        "Fehler: Die Datei 'funktionen_streamlit.py' konnte nicht gefunden werden. Stelle sicher, dass sie im selben Verzeichnis wie 'app_cluster.py' liegt."
    )
    st.stop()

LOG_FILE = "log.txt"
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
)


@st.cache_data
def analyze(file_bytes, analysis_method: str):
    match analysis_method:
        case "K-Means":
            results = run_feature_engineering_k_mean_analysis(file_bytes)
        case "DBSCAN":
            results = run_pca_dbscan_analysis(file_bytes)
        case "SOM":
            results = run_feature_engineering_som_analysis(file_bytes)
        case _:
            raise Exception("Unbekannte Analyse-Methode ausgewählt.")
    return results


def plot_cluster_map(cluster_map, unique_labels):
    # 1. Dimensionen & Basis-Setup
    num_rows, num_cols = cluster_map.shape
    aspect_ratio = num_cols / num_rows
    base_height = 700
    calculated_width = int(base_height * aspect_ratio) + 120

    # 2. Diskrete Farben aus Viridis generieren
    unique_sorted = sorted(unique_labels)
    n_clusters = len(unique_sorted)
    
    # Wir holen uns n_clusters Farben, gleichmäßig verteilt über die Viridis-Skala (0.0 bis 1.0)
    # Das sorgt für den "Verlauf"-Look, aber mit diskreten Stufen.
    viridis_samples = np.linspace(0, 1, n_clusters)
    current_colors = px.colors.sample_colorscale("Viridis", viridis_samples)
    
    # Erstelle die "klötzchenhafte" colorscale für Plotly Heatmaps
    discrete_colorscale = []
    step = 1 / n_clusters
    for i, color in enumerate(current_colors):
        discrete_colorscale.append([i * step, color])       # Start des Blocks
        discrete_colorscale.append([(i + 1) * step, color]) # Ende des Blocks

    # Mapping der original Labels auf 0 bis n-1 für die Heatmap-Darstellung
    label_to_idx = {label: i for i, label in enumerate(unique_sorted)}
    mapped_cluster_map = np.vectorize(label_to_idx.get)(cluster_map)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(mapped_cluster_map), 
            colorscale=discrete_colorscale,
            zmin=0,
            zmax=n_clusters, # Wichtig für die korrekten Block-Grenzen
            colorbar=dict(
                title="Cluster ID",
                title_side="top",
                title_font=dict(size=28),
                tickfont=dict(size=20),
                
                tickmode='array',
                # Ticks genau in die Mitte der Farbblöcke setzen:
                tickvals=[x + 0.5 for x in range(n_clusters)], 
                ticktext=[str(l) for l in unique_sorted],  # noqa: E741
                
                xpad=0,
                thickness=40,
                len=1.0
            )
        )
    )
    
    # 3. Layout (unverändert)
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

    # 1. Exakt gleiche Farb-Generierung wie in der Heatmap
    sorted_labels = sorted(unique_labels_order)
    n_clusters = len(sorted_labels)
    
    # Farben aus Viridis ziehen (0.0 bis 1.0)
    viridis_samples = np.linspace(0, 1, n_clusters)
    current_colors = px.colors.sample_colorscale("Viridis", viridis_samples)

    # Mapping erstellen: Label ID -> Viridis Farbe
    # Wir mappen hier Strings auf Farben, z.B. -1 -> Farbe, 0 -> Farbe
    label_to_color = {label: color for label, color in zip(sorted_labels, current_colors)}

    for spectrum, label_text in zip(mean_spectra, plot_labels):
        # ID aus dem Label-Text parsen
        # Wir suchen jetzt nach "Cluster" ODER "Neuron"
        try:
            # (?:...) gruppiert ohne zu speichern (Cluster oder Neuron)
            # \s* erlaubt Leerzeichen
            # (-?\d+) fängt die Zahl (auch negativ)
            match = re.search(r"(?:Cluster|Neuron)\s*(-?\d+)", label_text)
            
            if match:
                found_id = int(match.group(1))
                # Farbe aus dem Mapping holen, fallback auf schwarz
                line_color = label_to_color.get(found_id, "black")
            else:
                # Fallback, falls das Format ganz anders ist
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
    
    # Layout (unverändert)
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

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Raman-Karten Analyse (PCA + Clustering)")

# Sidebar für Optionen
st.sidebar.header("Analyse-Einstellungen")
analysis_method = st.sidebar.selectbox(
    label="Wähle die Clustering-Methode:",
    options=(None, "K-Means", "DBSCAN", "SOM"),
    index=0,
)

if analysis_method is None:
    st.info("Bitte wähle eine Analyse-Methode in der Seitenleiste aus.")
    st.stop()

# Hauptbereich für Datei-Upload
st.header("1. Raman-Karte hochladen")
uploaded_file = st.file_uploader("Wähle eine .mat-Datei (Witec)", type=["mat"])

if uploaded_file is None:
    st.error("Datei konnte nicht geladen werden!")
    st.stop()
else:
    st.success(f"Datei '{uploaded_file.name}' geladen.")

# Lese die Bytes der Datei (sicherer Umgang, Fehler abfangen)
try:
    file_bytes = uploaded_file.getvalue()
except Exception as e:
    logging.exception("Fehler beim Lesen der hochgeladenen Datei")
    st.error(f"Fehler beim Lesen der Datei: {e}")
    st.stop()

# Führe die Analyse aus und fange Ausnahmen, damit der Server nicht neu startet
try:
    results = analyze(file_bytes, analysis_method)
except Exception as e:
    logging.exception("Fehler während der Analyse")
    results = {"success": False, "error": str(e)}
    st.error(f"Analyse fehlgeschlagen: {e}")

# --- Ergebnisse anzeigen ---
st.header("2. Ergebnisse")

# Zeige das Analyse-Log in einem ausklappbaren Bereich
with st.expander("Analyse-Log anzeigen (Terminal-Ausgabe)"):
    with open(LOG_FILE) as f:
        logs = f.read()
    st.text_area("Log", logs, height=300)

# --- HAUPT-ERGEBNIS LOGIK ---
if results and results["success"]:
    st.success("Analyse erfolgreich abgeschlossen!")

    # 1. Feature-Maps
    if results.get("feature_names") and results.get("feature_maps") is not None:
        st.subheader("Feature-Maps (vor PCA)")
        feature_names = results["feature_names"]
        feature_maps = results["feature_maps"]
        
        # Auswahl der Feature
        chosen = st.selectbox("Wähle ein Feature zur Anzeige:", options=feature_names)
        idx = feature_names.index(chosen)
        feature_map = feature_maps[:, :, idx]
        
        # -----------------------------------------
        def plot_feature_map(feature_map, title, cmap="Viridis"):
            # 1. Dimensionen holen
            num_rows, num_cols = feature_map.shape
            
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(z=np.flipud(feature_map), colorscale=cmap, colorbar=dict(title=chosen))
            )
            fig.update_layout(
                width=800, 
                height=800, 
                title=title,
                # X-Achse begrenzen
                xaxis=dict(
                    range=[0, num_cols - 1],
                    constrain="domain" # Verhindert Whitespace
                ),
                # Y-Achse an X koppeln und begrenzen
                yaxis=dict(
                    scaleanchor="x", 
                    scaleratio=1,
                    range=[0, num_rows - 1],
                    constrain="domain"
                )
            )
            return fig
        # -----------------------------------------

        st.plotly_chart(plot_feature_map(feature_map, f"Feature: {chosen}"), use_container_width=False)

    # 2. Cluster-Karte (Wird immer angezeigt, wenn success=True)
    st.subheader("Cluster-Karte")
    if results.get("cluster_map") is not None:
        fig_map = plot_cluster_map(
            results["cluster_map"], results["unique_labels"]
        )
        st.plotly_chart(fig_map)
    else:
        st.warning("Keine Cluster-Karte verfügbar.")

    # 3. Mittlere Spektren (Wird immer angezeigt, wenn success=True)
    st.subheader("Mittlere Spektren")
    if results.get("mean_spectra") is not None:
        fig_spectra = plot_mean_spectra(
            results["mean_spectra"], results["plot_labels"], results["y_limit"], results["unique_labels"]
        )
        st.plotly_chart(fig_spectra)
    else:
        st.warning("Keine Spektren verfügbar.")

# Fehlerbehandlung: Greift nur, wenn results["success"] False ist oder results leer ist
elif results and not results.get("success", False):
    st.error(f"Analyse fehlgeschlagen. Fehler: {results.get('error', 'Unbekannter Fehler')}")