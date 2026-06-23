import logging
import os
import tempfile

import numpy as np
import ramanspy as rp
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn_som.som import SOM
import plotly.express as px
import pandas as pd

logger = logging.getLogger(__name__)

def plot_3d_cluster_space(pca_scores, labels, algorithm_name, opacity=0.6):
    """
    Visualisiert die ersten drei Hauptkomponenten (PCs) im 3D-Raum.
    """
    df = pd.DataFrame(pca_scores[:, :3], columns=['PC1', 'PC2', 'PC3'])
    
    # Label -1 als Hintergrund deklarieren
    labels_str = labels.astype(str)
    labels_str = np.where(labels_str == '-1', 'Hintergrund (-1)', labels_str)
        
    df['Cluster'] = labels_str
    df = df.sort_values('Cluster')

    fig = px.scatter_3d(
        df, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Cluster',
        title=f'3D-Merkmalsraum: {algorithm_name}',
        labels={'PC1': 'Hauptkomponente 1', 'PC2': 'Hauptkomponente 2', 'PC3': 'Hauptkomponente 3'},
        opacity=opacity,
        color_discrete_sequence=px.colors.qualitative.G10 
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
        ),
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig.update_traces(marker=dict(size=3))

    try:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    return fig


def Laden_Vorverarbeitung(Dateipfad):
    """
    Lädt die Witec-Datei und wendet eine generische Vorverarbeitung an.
    Kein Cropping auf spezifische Bereiche mehr.
    """
    logger.info("Starte generische Vorverarbeitung...")
    raman_image = rp.load.witec(Dateipfad, laser_excitation=488.047) 
    
    pipeline = rp.preprocessing.Pipeline(
        [
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.Gaussian(),
            rp.preprocessing.baseline.ASLS(),
        ]
    )

    raman_image = pipeline.apply(raman_image)

    h, w, _ = raman_image.spectral_data.shape
    
    # Flatten der Karte für Machine Learning: (h*w, anzahl_wellenzahlen)
    flat_spectra = raman_image.spectral_data.reshape((h * w, -1))
    
    # Filtere tote Pixel/Rauschen heraus (Spektren mit extrem wenig Signal)
    valid_mask_1d = np.sum(np.abs(flat_spectra), axis=1) > 1e-6

    if np.sum(valid_mask_1d) == 0:
        raise ValueError("Keine gültigen Spektren nach der Vorverarbeitung gefunden.")
        
    logger.info(f"Anzahl der gültigen Spektren gefunden: {np.sum(valid_mask_1d)}")

    return valid_mask_1d, raman_image, h, w


def run_pca_som_analysis(file_bytes, map_height=3, map_width=3):
    """
    Führt eine reine PCA auf den Spektren durch und nutzt SOM für das Clustering.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # --- 1. DATEN LADE & VORVERARBEITEN ---
        valid_mask_1d, raman_image, h, w = Laden_Vorverarbeitung(temp_file_path)
        
        # Isoliere nur die gültigen Spektren für das Training
        flat_spectra = raman_image.spectral_data.reshape((h * w, -1))
        valid_spectra = flat_spectra[valid_mask_1d]

        # --- 2. DATEN SKALIEREN ---
        logger.info("Skaliere Spektren (StandardScaler)...")
        scaler = StandardScaler()
        scaled_spectra = scaler.fit_transform(valid_spectra)

        # --- 3. REINE PCA AUF SPEKTREN ---
        logger.info("Starte PCA auf den vorverarbeiteten Spektraldaten...")
        pca = SklearnPCA(n_components=0.95, svd_solver='full', random_state=42)
        scores_np = pca.fit_transform(scaled_spectra)
        
        optimal_pcs_gefunden = pca.n_components_
        logger.info(f"PCA abgeschlossen. {optimal_pcs_gefunden} PCs erklären 95% der Varianz.")

        # --- 4. SOM CLUSTERING ---
        logger.info(f"Starte SOM-Training ({map_height}x{map_width})...")
        
        # Initialisiere SOM mit den Dimensionen der optimalen PCs
        som = SOM(m=map_height, n=map_width, dim=optimal_pcs_gefunden, lr=0.5, random_state=42)
        
        # Trainiere SOM und sage Cluster vorher
        som.fit(scores_np, epochs=5) 
        cluster_labels = som.predict(scores_np)
        
        logger.info("SOM-Training abgeschlossen.")

        # Visualisiere den PCA-Raum
        plot_3d_cluster_space(scores_np, cluster_labels, "Reine PCA + SOM")

        # --- 5. ERGEBNISSE ZUSAMMENFÜHREN ---
        final_cluster_map_1d = np.full(h*w, np.nan)
        BACKGROUND_LABEL = -1 
        
        # Weise Hintergrund und gefundene Cluster zu
        final_cluster_map_1d[~valid_mask_1d] = BACKGROUND_LABEL
        final_cluster_map_1d[valid_mask_1d] = cluster_labels
        
        final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

        # --- 6. IDENTIFIZIERUNG & MITTLERE SPEKTREN ---
        unique_final_labels = sorted([label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)])
        
        mean_spectra = []
        finale_plot_labels = []

        for label in unique_final_labels:
            cluster_mask_1d = (final_cluster_map_1d == label)
            cluster_mask_2d = cluster_mask_1d.reshape((h, w))
            
            if np.any(cluster_mask_2d):
                # Berechne das durchschnittliche Spektrum für dieses Cluster
                mean_spectra.append(raman_image[cluster_mask_2d].mean)
                
                # Generische Benennung
                if label == BACKGROUND_LABEL:
                    finale_plot_labels.append("Hintergrund / Rauschen")
                else:
                    finale_plot_labels.append(f"Neuron {int(label)}")
        
        # Einheitliches Y-Limit für den Plot berechnen
        global_max_intensity = 0
        for spectrum in mean_spectra:
             if spectrum.spectral_data.size > 0:
                current_max = np.max(spectrum.spectral_data)
                if current_max > global_max_intensity: 
                    global_max_intensity = current_max
        plot_ylim = global_max_intensity * 1.1

        return {
            "success": True,
            "cluster_map": final_cluster_map_2d,
            "unique_labels": unique_final_labels,
            "mean_spectra": mean_spectra,
            "plot_labels": finale_plot_labels,
            "y_limit": plot_ylim,
            "map_title": f"Cluster-Karte (SOM {map_height}x{map_width})"
        }

    except Exception as e:
        logger.error(f"FEHLER bei der PCA/SOM Analyse: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path): 
            os.remove(temp_file_path)