import logging
import os
import tempfile

import numpy as np
import ramanspy as rp
from kneed import KneeLocator
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
from sklearn_som.som import SOM
import multiprocessing
from joblib.externals.loky.process_executor import TerminatedWorkerError
import plotly.express as px
import pandas as pd

logger = logging.getLogger(__name__)

D_PEAK_FENSTER = (1300, 1400)
G_PEAK_FENSTER = (1550, 1620)
TWOD_PEAK_FENSTER = (2600, 2800)
G_PEAK_REF = 1580
TWOD_PEAK_REF = 2700
PMMA_CH_FENSTER = (2800, 3100)
PMMA_CO_FENSTER = (1720, 1750)
G_PEAK_PROMINENZ_SCHWELLE = 0.01
G_BASELINE_FENSTER_LINKS = (1450, 1550)
G_BASELINE_FENSTER_RECHTS = (1620, 1720)

def plot_3d_cluster_space(pca_scores, labels, algorithm_name, opacity=0.6):
    # 1. Daten in einen DataFrame umwandeln für Plotly
    # Wir nehmen nur die ersten 3 PCs, da wir nur 3D plotten können
    df = pd.DataFrame(pca_scores[:, :3], columns=['PC1', 'PC2', 'PC3'])
    
    # Cluster-Labels als String konvertieren, damit sie als diskrete Farben (Kategorien) behandelt werden
    # Für DBSCAN: Label -1 explizit als "Rauschen" benennen
    labels_str = labels.astype(str)
    if algorithm_name.upper() == "DBSCAN":
        labels_str = np.where(labels_str == '-1', 'Noise (-1)', labels_str)
        
    df['Cluster'] = labels_str
    
    # 2. Sortieren, damit die Legende ordentlich ist
    df = df.sort_values('Cluster')

    # 3. Der 3D Plot
    fig = px.scatter_3d(
        df, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Cluster',
        title=f'3D-Merkmalsraum Visualisierung: {algorithm_name}',
        labels={'PC1': 'Hauptkomponente 1 (Varianz)', 'PC2': 'Hauptkomponente 2', 'PC3': 'Hauptkomponente 3'},
        opacity=opacity,
        # Diskrete Farbskala nutzen
        color_discrete_sequence=px.colors.qualitative.G10 
    )
    # 4. Layout anpassen für bessere Sichtbarkeit
    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
        ),
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=0, r=0, b=0, t=30)  # Ränder minimieren
    )
    
    # Marker etwas kleiner machen für bessere Sichtbarkeit bei vielen Punkten
    fig.update_traces(marker=dict(size=3))

    # In Streamlit-Umgebung: direkt mit Streamlit darstellen.
    try:
        import streamlit as st

        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Fallback: gib das Figure-Objekt zurück (Aufrufer kann entscheiden).
        pass

    return fig

def lorentzian(x, amplitude, center, width, offset):
    return offset + (amplitude * (width**2 / ((x - center)**2 + width**2)))

def Laden_Vorverarbeitung(Dateipfad):
    logger.info("Starte Vorverarbeitung...")
    # Lädt die Raman-Karte vom temporären Dateipfad
    raman_image = rp.load.witec(Dateipfad, laser_excitation=488.047)  # type: ignore
    pipeline = rp.preprocessing.Pipeline(
        [
            rp.preprocessing.despike.WhitakerHayes(),
            rp.preprocessing.denoise.Gaussian(),
            rp.preprocessing.baseline.ASLS(),
        ]
    )

    raman_image = pipeline.apply(raman_image)

    cropper_si = rp.preprocessing.misc.Cropper(region=(400, 700))
    karte_silizium = cropper_si.apply(raman_image)

    cropper_graphen = rp.preprocessing.misc.Cropper(region=(1200, 3500))
    karte_graphen = cropper_graphen.apply(raman_image)
    logger.info(type(karte_graphen))

    if karte_graphen.spectral_data.shape[-1] == 0:  # type: ignore
        raise ValueError(
            "Der Graphen-Bereich (1200-3500 cm⁻¹) enthält keine Datenpunkte."
        )

    si_referenz_intensitaet = karte_silizium.spectral_data.max()    # type: ignore
    logger.info(
        f"Interner Si-Standard (I_Si0) gefunden: {si_referenz_intensitaet:.2f} a.u."
    )

    if si_referenz_intensitaet > 0:
        karte_silizium.spectral_data /= si_referenz_intensitaet # type: ignore
        karte_graphen.spectral_data /= si_referenz_intensitaet  # type: ignore

    h, w, _ = karte_graphen.spectral_data.shape     # type: ignore
    flat_spectra = karte_graphen.spectral_data.reshape((h * w, -1)) # type: ignore
    valid_mask_1d = np.sum(np.abs(flat_spectra), axis=1) > 1e-6

    if np.sum(valid_mask_1d) == 0:
        raise ValueError("Keine gültigen Spektren nach der Vorverarbeitung gefunden.")
    logger.info(f"Anzahl der gültigen Spektren gefunden: {np.sum(valid_mask_1d)}")

    return valid_mask_1d, karte_silizium, karte_graphen, h, w

def extrahiere_features_robust(spectrum, spectral_axis):
    try:
        # Normiere Zugriff auf y- und x-Werte
        if hasattr(spectrum, "spectral_data"):
            y = np.asarray(spectrum.spectral_data, dtype=float)
        else:
            y = np.asarray(spectrum, dtype=float)

        x = np.asarray(spectral_axis, dtype=float)

        if y.size != x.size or y.size == 0:
            return [np.nan] * 6

        # Hilfsfunktionen
        def safe_mask(window):
            return (x >= window[0]) & (x <= window[1])

        def estimate_peak_pos_and_height(xseg, yseg, baseline):
            if yseg.size == 0:
                return (np.nan, np.nan)
            idx = np.nanargmax(yseg)
            return (xseg[idx], yseg[idx] - baseline)

        def estimate_fwhm(xseg, yseg):
            if yseg.size == 0:
                return np.nan
            ymax = np.nanmax(yseg)
            if np.isnan(ymax) or ymax <= 0:
                return np.nan
            half = ymax / 2.0
            inds = np.where(yseg >= half)[0]
            if inds.size < 2:
                return np.nan
            # Linear interpolation to estimate boundaries
            left = inds[0]
            right = inds[-1]
            # interpolate left edge
            if left == 0:
                left_x = xseg[left]
            else:
                x1, x0 = xseg[left - 1], xseg[left]
                y1, y0 = yseg[left - 1], yseg[left]
                if y0 == y1:
                    left_x = xseg[left]
                else:
                    left_x = x1 + (half - y1) * (x0 - x1) / (y0 - y1)
            # interpolate right edge
            if right == len(yseg) - 1:
                right_x = xseg[right]
            else:
                x0, x1 = xseg[right], xseg[right + 1]
                y0, y1 = yseg[right], yseg[right + 1]
                if y1 == y0:
                    right_x = xseg[right]
                else:
                    right_x = x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            return abs(right_x - left_x)

        # Baseline: mittlerer Wert der linken und rechten Baseline-Fenster
        baseline_left_mask = safe_mask(G_BASELINE_FENSTER_LINKS)
        baseline_right_mask = safe_mask(G_BASELINE_FENSTER_RECHTS)
        left_mean = np.nanmean(y[baseline_left_mask]) if baseline_left_mask.any() else 0.0
        right_mean = np.nanmean(y[baseline_right_mask]) if baseline_right_mask.any() else 0.0
        baseline = (left_mean + right_mean) / 2.0

        # G-Peak
        g_mask = safe_mask(G_PEAK_FENSTER)
        if not g_mask.any():
            return [np.nan] * 6
        xg, yg = x[g_mask], y[g_mask]
        pos_g, g_amp = estimate_peak_pos_and_height(xg, yg, baseline)
        if np.isnan(g_amp) or g_amp < G_PEAK_PROMINENZ_SCHWELLE:
            return [np.nan] * 6

        # 2D-Peak
        mask_2d = safe_mask(TWOD_PEAK_FENSTER)
        x2d, y2d = x[mask_2d], y[mask_2d]
        pos_2d, twod_amp = estimate_peak_pos_and_height(x2d, y2d, baseline)
        fwhm_2d = estimate_fwhm(x2d, y2d)

        # D-Peak
        d_mask = safe_mask(D_PEAK_FENSTER)
        yd = y[d_mask] if d_mask.any() else np.array([])
        d_amp = np.nanmax(yd) - baseline if yd.size > 0 else 0.0

        # PMMA bands
        pmma_ch_mask = safe_mask(PMMA_CH_FENSTER)
        pmma_co_mask = safe_mask(PMMA_CO_FENSTER)
        pmma_ch = np.nanmean(y[pmma_ch_mask] - baseline) if pmma_ch_mask.any() else 0.0
        pmma_co = np.nanmean(y[pmma_co_mask] - baseline) if pmma_co_mask.any() else 0.0

        # Ratios
        ig = g_amp if g_amp > 0 else np.nan
        id_ig_ratio = (d_amp / ig) if not np.isnan(ig) and ig != 0 else np.nan
        i2d_ig_ratio = (twod_amp / ig) if not np.isnan(ig) and ig != 0 else np.nan
        pmma_ratio = ((pmma_ch + pmma_co) / ig) if not np.isnan(ig) and ig != 0 else np.nan

        return [id_ig_ratio, fwhm_2d, i2d_ig_ratio, pos_g, pos_2d, pmma_ratio]

    except Exception:
        return [np.nan] * 6

def finde_optimales_k(daten, k_max=10):
    logger.info("Suche nach optimaler Clusteranzahl (K) mittels Silhouetten-Analyse...")

    num_samples = daten.shape[0]
    sample_size = min(10000, num_samples // 10)

    # Stelle sicher, dass sample_size größer als k_max ist, sonst schlägt die Analyse fehl
    if sample_size <= k_max:
        logger.info(
            f"  WARNUNG: Stichprobengröße ({sample_size}) ist zu klein. Verwende alle {num_samples} Proben."
        )
        sample_size = num_samples
        daten_sample = daten
    else:
        sample_indices = np.random.choice(num_samples, size=sample_size, replace=False)
        daten_sample = daten[sample_indices]
        logger.info(f"  Analysiere eine Stichprobe von {sample_size} Spektren...")

    silhouette_scores = []
    k_range = range(2, k_max + 1)

    for k in k_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans_model.fit_predict(daten_sample)

        score = silhouette_score(daten_sample, cluster_labels)
        silhouette_scores.append(score)
        logger.info(f"  Silhouetten-Score für K={k}: {score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"-> Optimales K gefunden: {optimal_k} (höchster Silhouetten-Score)")
    return optimal_k

def filtere_graphen_spektren(karte_graphen, valid_mask_1d, h, w):
    logger.info("Filtere Spektren: Trenne Graphen von Substrat...")

    graphen_mask_1d = np.copy(valid_mask_1d)

    G_PEAK_FENSTER = (1550, 1620)
    G_BASELINE_FENSTER_LINKS = (1450, 1550)
    G_BASELINE_FENSTER_RECHTS = (1620, 1720)
    G_PEAK_PROMINENZ_SCHWELLE = 0.01

    valid_indices = np.where(valid_mask_1d)[0]
    y_coords, x_coords = np.unravel_index(valid_indices, (h, w))

    for i in range(len(valid_indices)):
        original_index = valid_indices[i]
        spectrum = karte_graphen[y_coords[i], x_coords[i]]

        g_peak_mask = (spectrum.spectral_axis >= G_PEAK_FENSTER[0]) & (
            spectrum.spectral_axis <= G_PEAK_FENSTER[1]
        )
        baseline_mask_links = (
            spectrum.spectral_axis >= G_BASELINE_FENSTER_LINKS[0]
        ) & (spectrum.spectral_axis <= G_BASELINE_FENSTER_LINKS[1])
        baseline_mask_rechts = (
            spectrum.spectral_axis >= G_BASELINE_FENSTER_RECHTS[0]
        ) & (spectrum.spectral_axis <= G_BASELINE_FENSTER_RECHTS[1])

        g_peak_intensity_max = (
            spectrum.spectral_data[g_peak_mask].max() if g_peak_mask.any() else 0
        )
        baseline_links_mean = (
            np.mean(spectrum.spectral_data[baseline_mask_links])
            if baseline_mask_links.any()
            else g_peak_intensity_max
        )
        baseline_rechts_mean = (
            np.mean(spectrum.spectral_data[baseline_mask_rechts])
            if baseline_mask_rechts.any()
            else g_peak_intensity_max
        )

        lokale_baseline = (baseline_links_mean + baseline_rechts_mean) / 2
        g_peak_prominenz = g_peak_intensity_max - lokale_baseline

        if g_peak_prominenz < G_PEAK_PROMINENZ_SCHWELLE:
            graphen_mask_1d[original_index] = False

    substrat_mask_1d = valid_mask_1d & ~graphen_mask_1d

    logger.info(
        f"  {np.sum(graphen_mask_1d)} Graphen-Spektren und {np.sum(substrat_mask_1d)} Substrat-Spektren gefunden."
    )

    return graphen_mask_1d, substrat_mask_1d

def K_Mean(valid_mask_1d, scores, h, w):
    logger.info("Starte K-Means-Clustering...")

    pca_scores_for_clustering = np.array(scores).T

    # Bestimme optimales K mittels Silhouetten-Analyse
    chosen_k = finde_optimales_k(pca_scores_for_clustering, k_max=10)
    # Grenzen: mindestens 2, höchstens 10 (wie in finde_optimales_k)
    chosen_k = max(2, min(chosen_k, 10))

    logger.info(f"Silhouette-optimales K={chosen_k}; verwende K={chosen_k} für finalen K-Means.")
    kmeans_model = KMeans(n_clusters=chosen_k, random_state=42, n_init="auto")
    cluster_labels = kmeans_model.fit_predict(pca_scores_for_clustering)
    return cluster_labels

def identifiziere_cluster(
    mean_spectra_graphen,
    mean_spectra_silizium,
    gefundene_cluster_ids,
    substrat_label=None,
):
    logger.info(
        "Starte finale, hierarchische Identifizierung (inkl. PMMA/Strain/Doping-Check)..."
    )

    # --- Hilfsfunktion für den Fit ---
    def lorentzian(x, amplitude, center, width, offset):
        return offset + (amplitude * (width**2 / ((x - center) ** 2 + width**2)))

    # --- Fenster, Schwellenwerte und Referenzpositionen ---
    SI_PEAK_FENSTER = (500, 540)
    D_PEAK_FENSTER = (1300, 1400)
    G_PEAK_FENSTER = (1550, 1620)
    TWOD_PEAK_FENSTER = (2600, 2800)
    G_BASELINE_FENSTER_LINKS = (1450, 1550)
    G_BASELINE_FENSTER_RECHTS = (1620, 1720)
    G_PEAK_PROMINENZ_SCHWELLE = 0.01
    FWHM_GRENZE_SLG = 39
    FWHM_GRENZE_BLG = 50
    I2D_IG_GRENZE_SLG = 1.3
    ID_IG_GRENZE_NIEDRIG = 0.3
    ID_IG_GRENZE_HOCH = 1.0

    G_PEAK_REF = 1580
    TWOD_PEAK_REF = 2700
    SHIFT_THRESHOLD = 10

    PMMA_CH_FENSTER = (2800, 3100)
    PMMA_CO_FENSTER = (1720, 1750)
    PMMA_SCHWELLE = 2.0

    cluster_identitaeten = {}

    for i, spectrum_graphen in enumerate(mean_spectra_graphen):
        if substrat_label is not None and gefundene_cluster_ids[i] == substrat_label:
            cluster_identitaeten[f"Cluster {i}"] = "Substrat"
            continue

        g_peak_mask = (spectrum_graphen.spectral_axis >= G_PEAK_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= G_PEAK_FENSTER[1]
        )
        pmma_ch_mask = (spectrum_graphen.spectral_axis >= PMMA_CH_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= PMMA_CH_FENSTER[1]
        )
        pmma_co_mask = (spectrum_graphen.spectral_axis >= PMMA_CO_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= PMMA_CO_FENSTER[1]
        )

        g_region_intensity = (
            np.mean(spectrum_graphen.spectral_data[g_peak_mask])
            if g_peak_mask.any()
            else 1e-9
        )
        pmma_ch_intensity = (
            np.mean(spectrum_graphen.spectral_data[pmma_ch_mask])
            if pmma_ch_mask.any()
            else 0
        )
        pmma_co_intensity = (
            np.mean(spectrum_graphen.spectral_data[pmma_co_mask])
            if pmma_co_mask.any()
            else 0
        )

        pmma_ratio = (pmma_ch_intensity + pmma_co_intensity) / g_region_intensity
        logger.info(
            f"  Cluster {gefundene_cluster_ids[i]}: PMMA/G-Verhältnis = {pmma_ratio:.2f}"
        )

        if pmma_ratio > PMMA_SCHWELLE:
            cluster_identitaeten[f"Cluster {i}"] = "PMMA-Rückstand"
            continue

        g_peak_intensity_max = (
            spectrum_graphen.spectral_data[g_peak_mask].max()
            if g_peak_mask.any()
            else 0
        )
        baseline_mask_links = (
            spectrum_graphen.spectral_axis >= G_BASELINE_FENSTER_LINKS[0]
        ) & (spectrum_graphen.spectral_axis <= G_BASELINE_FENSTER_LINKS[1])
        baseline_mask_rechts = (
            spectrum_graphen.spectral_axis >= G_BASELINE_FENSTER_RECHTS[0]
        ) & (spectrum_graphen.spectral_axis <= G_BASELINE_FENSTER_RECHTS[1])
        baseline_links_mean = (
            np.mean(spectrum_graphen.spectral_data[baseline_mask_links])
            if baseline_mask_links.any()
            else g_peak_intensity_max
        )
        baseline_rechts_mean = (
            np.mean(spectrum_graphen.spectral_data[baseline_mask_rechts])
            if baseline_mask_rechts.any()
            else g_peak_intensity_max
        )
        lokale_baseline = (baseline_links_mean + baseline_rechts_mean) / 2
        g_peak_prominenz = g_peak_intensity_max - lokale_baseline

        if g_peak_prominenz < G_PEAK_PROMINENZ_SCHWELLE:
            cluster_identitaeten[f"Cluster {i}"] = "Substrat"
            continue

        pos_g = G_PEAK_REF
        if g_peak_mask.any():
            x_g, y_g = (
                spectrum_graphen.spectral_axis[g_peak_mask],
                spectrum_graphen.spectral_data[g_peak_mask],
            )
            try:
                p0_g = [np.max(y_g) - np.min(y_g), x_g[np.argmax(y_g)], 15, np.min(y_g)]
                params_g, _ = curve_fit(lorentzian, x_g, y_g, p0=p0_g)
                pos_g = params_g[1]
            except RuntimeError:
                logger.info(
                    f"  WARNUNG: G-Peak-Fit für Cluster-Index {i} fehlgeschlagen."
                )

        fwhm_2d = 999
        pos_2d = TWOD_PEAK_REF
        x_peak, y_peak = spectrum_graphen.spectral_axis, spectrum_graphen.spectral_data
        mask_2d = (x_peak >= TWOD_PEAK_FENSTER[0]) & (x_peak <= TWOD_PEAK_FENSTER[1])
        if mask_2d.any():
            x_2d, y_2d = x_peak[mask_2d], y_peak[mask_2d]
            try:
                p0_2d = [
                    np.max(y_2d) - np.min(y_2d),
                    x_2d[np.argmax(y_2d)],
                    20,
                    np.min(y_2d),
                ]
                params_2d, _ = curve_fit(lorentzian, x_2d, y_2d, p0=p0_2d)
                pos_2d = params_2d[1]
                fwhm_2d = 2 * abs(params_2d[2])
            except RuntimeError:
                logger.info(
                    f"  WARNUNG: 2D-Peak-Fit für Cluster-Index {i} fehlgeschlagen."
                )
        logger.info(
            f"  -> Cluster {gefundene_cluster_ids[i]}: FWHM(2D)={fwhm_2d:.2f} cm-1, Pos(G)={pos_g:.2f} cm-1, Pos(2D)={pos_2d:.2f} cm-1"
        )

        twod_peak_intensity = (
            spectrum_graphen.spectral_data[mask_2d].max() if mask_2d.any() else 0
        )
        i2d_ig_ratio = (
            twod_peak_intensity / g_peak_intensity_max
            if g_peak_intensity_max > 0
            else 0
        )
        d_peak_mask = (spectrum_graphen.spectral_axis >= D_PEAK_FENSTER[0]) & (
            spectrum_graphen.spectral_axis <= D_PEAK_FENSTER[1]
        )
        d_peak_intensity = (
            spectrum_graphen.spectral_data[d_peak_mask].max()
            if d_peak_mask.any()
            else 0
        )
        id_ig_ratio = (
            d_peak_intensity / g_peak_intensity_max if g_peak_intensity_max > 0 else 0
        )

        schicht_label = ""

        is_fwhm_slg = fwhm_2d < FWHM_GRENZE_SLG
        is_ratio_slg = i2d_ig_ratio > I2D_IG_GRENZE_SLG

        if is_fwhm_slg and is_ratio_slg:
            schicht_label = "Monolage Graphen"

        elif fwhm_2d < FWHM_GRENZE_BLG:
            schicht_label = "Zweischichtiges Graphen"

        else:
            spectrum_silizium = mean_spectra_silizium[i]
            si_mask = (spectrum_silizium.spectral_axis >= SI_PEAK_FENSTER[0]) & (
                spectrum_silizium.spectral_axis <= SI_PEAK_FENSTER[1]
            )
            si_verhaeltnis = (
                min(spectrum_silizium.spectral_data[si_mask].max(), 1.0)
                if si_mask.any()
                else 0
            )

            if si_verhaeltnis > 0.75:
                schicht_label = "Zweischichtiges Graphen"
            else:
                schicht_label = "Viele Lagen Graphen"

        qualitaet_label = ""
        if id_ig_ratio >= ID_IG_GRENZE_HOCH:
            qualitaet_label = " (stark defekt)"
        elif id_ig_ratio >= ID_IG_GRENZE_NIEDRIG:
            qualitaet_label = " (defekt)"
        else:
            qualitaet_label = " (hohe Qualität)"

        strain_doping_label = ""
        delta_pos_g = pos_g - G_PEAK_REF
        delta_pos_2d = pos_2d - TWOD_PEAK_REF

        if abs(delta_pos_g) > 1:
            shift_ratio = delta_pos_2d / delta_pos_g
        else:
            shift_ratio = 0

        if delta_pos_g > SHIFT_THRESHOLD or delta_pos_2d > SHIFT_THRESHOLD:
            if shift_ratio > 2.0:
                strain_doping_label = ", kompressiv verspannt"
            else:
                strain_doping_label = ", p-dotiert"
        elif delta_pos_g < -SHIFT_THRESHOLD or delta_pos_2d < -SHIFT_THRESHOLD:
            strain_doping_label = ", tensil verspannt"
        elif delta_pos_g > SHIFT_THRESHOLD or delta_pos_2d < -SHIFT_THRESHOLD:
            strain_doping_label = ", n-dotiert"

        cluster_identitaeten[f"Cluster {i}"] = (
            schicht_label + qualitaet_label + strain_doping_label
        )

    logger.info("Identifizierung abgeschlossen:", cluster_identitaeten)
    return cluster_identitaeten

def DBSCAN_Clustering(eps, min_samples, valid_mask_1d, scaled_scores, h, w):
    logger.info(
        f"Führe DBSCAN-Clustering mit eps={eps:.4f} und min_samples={min_samples} durch..."
    )

    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan_model.fit_predict(scaled_scores)
    return cluster_labels

def finde_besten_eps(scores, min_samples):
    logger.info("Bestimme optimalen eps-Wert automatisch...")

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(scores)
    distances, indices = neighbors_fit.kneighbors(scores)

    distances = np.sort(distances, axis=0)
    distances = distances[:, min_samples - 1]

    kneedle = KneeLocator(
        range(len(distances)), distances, S=1.0, curve="convex", direction="increasing"
    )

    optimal_eps = kneedle.knee_y
    logger.info(f"-> Optimaler eps-Wert gefunden: {optimal_eps:.4f}")
    return optimal_eps

def run_feature_engineering_k_mean_analysis(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # --- 1. DATENLADEN & FILTERN ---
        logger.info("Starte Vorverarbeitung...")
        valid_mask_1d, karte_silizium, karte_graphen, h, w = \
            Laden_Vorverarbeitung(temp_file_path)
        
        logger.info("Filtere Spektren: Trenne Graphen von Substrat...")
        graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(
            karte_graphen, valid_mask_1d, h, w)
        logger.info(f"  {np.sum(graphen_mask_1d)} Graphen-Spektren und {np.sum(substrat_mask_1d)} Substrat-Spektren gefunden.")

        # --- 2. FEATURE ENGINEERING ---
        logger.info("Starte Feature Engineering (Parallel-Extraktion)...")
        
        # Bereite die Daten für die parallele Verarbeitung vor
        graphen_spektren_daten = karte_graphen.spectral_data[graphen_mask_1d.reshape((h, w))] # type: ignore
        spectral_axis = karte_graphen.spectral_axis # type: ignore
        
        # Begrenze die parallelen Jobs, um OOM/Worker-Abstürze zu vermeiden.
        cpu_count = multiprocessing.cpu_count()
        # Verwende maximal 4 Jobs oder alle verfügbaren Kerne, je nachdem, was kleiner ist.
        num_cores = max(1, min(4, cpu_count))
        logger.info(f"Nutze {num_cores} CPU-Kerne für {len(graphen_spektren_daten)} Spektren (begrenzte Parallelisierung zur Stabilität)...")

        # Paralleler Aufruf von 'extrahiere_features_robust' in Batches,
        # um Peak-Memory zu begrenzen und stabiler in der EXE zu laufen.
        try:
            feature_list = []
            batch_size = 2000
            total = len(graphen_spektren_daten)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                logger.info(f"  Verarbeite Batch {start}-{end} von {total} Spektren...")
                batch = graphen_spektren_daten[start:end]
                batch_features = Parallel(n_jobs=num_cores, backend="threading")(  # use threads in EXE
                    delayed(extrahiere_features_robust)(spectrum, spectral_axis)
                    for spectrum in batch
                )
                feature_list.extend(batch_features)
        except TerminatedWorkerError as e:
            logger.error(
                "Ein Worker-Prozess ist abgestürzt. Fallback auf sequentielle Verarbeitung. Fehler: %s",
                e,
            )
            # Retry sequentiell, um mehr Informationen zu sammeln und OOM zu vermeiden
            feature_list = [extrahiere_features_robust(spectrum, spectral_axis) for spectrum in graphen_spektren_daten]
        except Exception as e:
            logger.exception("Unbekannter Fehler während der Parallel-Extraktion; versuche sequentiell. Fehler: %s", e)
            feature_list = [extrahiere_features_robust(spectrum, spectral_axis) for spectrum in graphen_spektren_daten]
        
        feature_matrix = np.array(feature_list)

        # Definiere Feature-Namen (immer 6 Features wie von extrahiere_features_robust)
        feature_names = [
            "I(D)/I(G)",
            "FWHM(2D)",
            "I(2D)/I(G)",
            "Pos(G)",
            "Pos(2D)",
            "PMMA_ratio",
        ]

        # Erzeuge 2D-Feature-Maps (h, w, n_features) und fülle die Graphen-Positionen
        try:
            feature_maps_2d = np.full((h, w, len(feature_names)), np.nan, dtype=float)
            # Indizes der gültigen Graphen-Positionen (flach)
            valid_indices = np.where(graphen_mask_1d)[0]
            if feature_matrix.ndim == 2 and feature_matrix.shape[0] == len(valid_indices):
                y_coords, x_coords = np.unravel_index(valid_indices, (h, w))
                for i in range(len(valid_indices)):
                    feature_maps_2d[y_coords[i], x_coords[i], :] = feature_matrix[i]
            else:
                logger.warning("Feature-Matrix und Graphen-Positionen stimmen nicht überein; Feature-Maps bleiben mit NaN gefüllt.")
        except Exception:
            feature_maps_2d = None
            logger.exception("Fehler beim Erstellen der Feature-Maps; setze auf None.")

        logger.info("Feature-Extraktion abgeschlossen.")

        # --- 3. DATEN BEREINIGEN & SKALIEREN ---
        # Ersetze 'np.nan' (von fehlgeschlagenen Fits) mit dem Mittelwert der Spalte
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        feature_matrix_imputed = imputer.fit_transform(feature_matrix)
        
        # Skaliere die Features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix_imputed)

        # --- 4. PCA AUF FEATURES ---
        # Diese PCA ist jetzt sauber (6 Dimensionen -> 3-4)
        logger.info("Starte PCA auf extrahierten Features...")
        # n_components=0.95 (95% Varianz) und svd_solver='full' (da D < N)
        pca = SklearnPCA(n_components=0.95, svd_solver='full', random_state=42)
        scores_np = pca.fit_transform(scaled_features)
        
        optimal_pcs_gefunden = pca.n_components_
        logger.info(f"PCA auf Features abgeschlossen. {optimal_pcs_gefunden} PCs erklären 95% der Varianz.")
        
        # K-Means erwartet (Features, Samples), also transponieren
        final_scores = scores_np.T

        # --- 5. K-MEANS CLUSTERING ---
        logger.info("Starte K-Means-Clustering auf Feature-Scores...")
        graphen_cluster_labels = K_Mean(graphen_mask_1d, final_scores, h, w) 

        plot_3d_cluster_space(scores_np, graphen_cluster_labels, "K-Means")

        # --- 6. ERGEBNISSE BERECHNEN ---
        logger.info("Kombiniere Ergebnisse zu finaler Cluster-Karte...")
        final_cluster_map_1d = np.full(h*w, np.nan)
        SUBSTRAT_LABEL = 0
        final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
        final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels + 1 # +1, da 0 für Substrat reserviert ist
        
        final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

        logger.info("Berechne mittlere Spektren für jeden finalen Cluster...")
        unique_final_labels = sorted([label for label in np.unique(
            final_cluster_map_1d) if not np.isnan(label)])
        
        mean_spectra_graphen = []
        mean_spectra_silizium = []
        gefundene_cluster_ids = []

        for label in unique_final_labels:
            cluster_mask_1d = (final_cluster_map_1d == label)
            cluster_mask_2d = cluster_mask_1d.reshape((h, w))
            if np.any(cluster_mask_2d):
                mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
                mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean)
                gefundene_cluster_ids.append(int(label))
        
        logger.info("Starte finale, hierarchische Identifizierung (inkl. PMMA/Strain/Doping-Check)...")
        neue_labels_map = identifiziere_cluster(mean_spectra_graphen,
                                           mean_spectra_silizium, 
                                           gefundene_cluster_ids, 
                                           substrat_label=SUBSTRAT_LABEL)
        logger.info(f"Identifizierung abgeschlossen: {neue_labels_map}")
        
        finale_plot_labels = [f"Cluster {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}"
                              for i, original_id in enumerate(gefundene_cluster_ids)]

        # Y-Limit-Berechnung für einheitliche Darstellung
        global_max_intensity = 0
        for spectrum in mean_spectra_graphen:
            if spectrum.spectral_data.size > 0:
                current_max = np.max(spectrum.spectral_data)
                if current_max > global_max_intensity: 
                    global_max_intensity = current_max
        plot_ylim = global_max_intensity * 1.1
        logger.info(f"Setze einheitliches Y-Achsen-Limit auf: {plot_ylim:.4f} a.u.")

        return {
            "success": True,
            "cluster_map": final_cluster_map_2d,
            "unique_labels": unique_final_labels,
            "mean_spectra": mean_spectra_graphen,
            "plot_labels": finale_plot_labels,
            "y_limit": plot_ylim,
            "feature_maps": feature_maps_2d,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,  # raw extracted features (may contain NaNs)
            "feature_matrix_imputed": feature_matrix_imputed,
        }
        

    except Exception as e:
        logger.error(f"FEHLER während der Feature-Engineering-K-Means-Analyse: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path): # type: ignore
            os.remove(temp_file_path)   # type: ignore
            logger.info(f"Temporäre Datei gelöscht: {temp_file_path}")  # type: ignore

def run_pca_dbscan_analysis(file_bytes):
    try:
        # Erstelle eine temporäre Datei, um die Bytes zu speichern
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        logger.info(f"Temporäre Datei erstellt: {temp_file_path}")

        # Schritt 1: Lade und verarbeite die Raman-Karte.
        valid_mask_1d, karte_silizium, karte_graphen, h, w = Laden_Vorverarbeitung(
            temp_file_path
        )

        # Schritt 2: Trenne Graphen-Spektren von Substrat-Spektren
        graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(
            karte_graphen, valid_mask_1d, h, w
        )

        # --- Feature-Extraktion wie in run_feature_engineering_k_mean_analysis ---
        logger.info("Starte Feature-Extraktion für DBSCAN-Analyse...")
        graphen_spektren_daten = karte_graphen.spectral_data[graphen_mask_1d.reshape((h, w))]  # type: ignore
        spectral_axis = karte_graphen.spectral_axis  # type: ignore
        cpu_count = multiprocessing.cpu_count()
        num_cores = max(1, min(4, cpu_count))
        logger.info(f"Nutze {num_cores} CPU-Kerne (Threads) für {len(graphen_spektren_daten)} Spektren...")

        feature_list = Parallel(n_jobs=num_cores, backend="threading")(  # use threads to avoid process spawn issues in EXE
            delayed(extrahiere_features_robust)(spectrum, spectral_axis)
            for spectrum in graphen_spektren_daten
        )

        feature_matrix = np.array(feature_list)

        # Definiere Feature-Namen und Feature-Maps
        feature_names = [
            "I(D)/I(G)",
            "FWHM(2D)",
            "I(2D)/I(G)",
            "Pos(G)",
            "Pos(2D)",
            "PMMA_ratio",
        ]

        try:
            feature_maps_2d = np.full((h, w, len(feature_names)), np.nan, dtype=float)
            valid_indices = np.where(graphen_mask_1d)[0]
            if feature_matrix.ndim == 2 and feature_matrix.shape[0] == len(valid_indices):
                y_coords, x_coords = np.unravel_index(valid_indices, (h, w))
                for i in range(len(valid_indices)):
                    feature_maps_2d[y_coords[i], x_coords[i], :] = feature_matrix[i]
            else:
                logger.warning("Feature-Matrix und Graphen-Positionen stimmen nicht überein; Feature-Maps bleiben NaN.")
        except Exception:
            feature_maps_2d = None
            logger.exception("Fehler beim Erstellen der Feature-Maps; setze auf None.")

        logger.info("Feature-Extraktion abgeschlossen.")

        # --- Imputation & Skalierung ---
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        feature_matrix_imputed = imputer.fit_transform(feature_matrix)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix_imputed)

        logger.info("Starte PCA auf extrahierten Features...")
        pca = SklearnPCA(n_components=0.95, svd_solver='full', random_state=42)
        scores_np = pca.fit_transform(scaled_features)
        
        optimal_pcs_gefunden = pca.n_components_
        logger.info(f"PCA auf Features abgeschlossen. {optimal_pcs_gefunden} PCs erklären 95% der Varianz.")


        D = optimal_pcs_gefunden
        min_samples_auto = 2 * D
        logger.info(f"Bestimme min_samples automatisch (2*D): {min_samples_auto}")

        eps_auto = finde_besten_eps(scores_np, min_samples_auto)

        graphen_cluster_labels = DBSCAN_Clustering(
            eps_auto, min_samples_auto, graphen_mask_1d, scores_np, h, w
        )

        # --- Finale Cluster-Karte ---
        logger.info("Kombiniere Ergebnisse zu finaler Cluster-Karte...")
        final_cluster_map_1d = np.full(h * w, np.nan)
        SUBSTRAT_LABEL = -2
        final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
        final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels
        final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

        # --- Mittlere Spektren pro Cluster ---
        logger.info("Berechne mittlere Spektren für jeden finalen Cluster...")
        unique_final_labels = sorted([label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)])

        mean_spectra_graphen = []
        mean_spectra_silizium = []
        gefundene_cluster_ids = []

        for label in unique_final_labels:
            cluster_mask_1d = final_cluster_map_1d == label
            cluster_mask_2d = cluster_mask_1d.reshape((h, w))
            if np.any(cluster_mask_2d):
                mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
                mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean)
                gefundene_cluster_ids.append(int(label))

        # --- Cluster-Identifikation ---
        neue_labels_map = identifiziere_cluster(
            mean_spectra_graphen, mean_spectra_silizium, gefundene_cluster_ids, substrat_label=SUBSTRAT_LABEL
        )
        finale_plot_labels = [f"Cluster {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}" for i, original_id in enumerate(gefundene_cluster_ids)]

        # --- Plot-Parameter ---
        global_max_intensity = 0
        for spectrum in mean_spectra_graphen:
            if spectrum.spectral_data.size > 0:
                current_max = np.max(spectrum.spectral_data)
                if current_max > global_max_intensity:
                    global_max_intensity = current_max
        plot_ylim = global_max_intensity * 1.1
        logger.info(f"Setze einheitliches Y-Achsen-Limit auf: {plot_ylim:.4f} a.u.")

        # --- Rückgabe an Streamlit ---
        return {
            "success": True,
            "cluster_map": final_cluster_map_2d,
            "unique_labels": unique_final_labels,
            "mean_spectra": mean_spectra_graphen,
            "plot_labels": finale_plot_labels,
            "y_limit": plot_ylim,
            "map_title": "Finale Cluster-Karte (DBSCAN auf Features)",
            "feature_maps": feature_maps_2d,
            "feature_names": feature_names,
            "feature_matrix": feature_matrix,
            "feature_matrix_imputed": feature_matrix_imputed,
        }

    except Exception as e:
        logger.info(f"FEHLER während der DBSCAN-Analyse: {e}")
        import traceback

        logger.info(traceback.format_exc())
        return {"success": False, "error": str(e)}

    finally:
        # Bereinige die temporäre Datei
        if "temp_file_path" in locals() and os.path.exists(temp_file_path): # type: ignore
            os.remove(temp_file_path)   # type: ignore
            logger.info(f"Temporäre Datei gelöscht: {temp_file_path}")  # type: ignore

def run_feature_engineering_som_analysis(file_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # --- 1. DATENLADEN & FILTERN ---
        logger.info("Starte Vorverarbeitung...")
        valid_mask_1d, karte_silizium, karte_graphen, h, w = Laden_Vorverarbeitung(temp_file_path)
        
        logger.info("Filtere Spektren: Trenne Graphen von Substrat...")
        graphen_mask_1d, substrat_mask_1d = filtere_graphen_spektren(karte_graphen, valid_mask_1d, h, w)

        # --- 2. FEATURE ENGINEERING ---
        logger.info("Starte Feature Engineering (Parallel)...")
        graphen_spektren_daten = karte_graphen.spectral_data[graphen_mask_1d.reshape((h, w))]   # type: ignore
        spectral_axis = karte_graphen.spectral_axis # type: ignore
        
        cpu_count = multiprocessing.cpu_count()
        num_cores = max(1, min(4, cpu_count))
        feature_list = Parallel(n_jobs=num_cores, backend="threading")(  # use threads in EXE
            delayed(extrahiere_features_robust)(spectrum, spectral_axis) 
            for spectrum in graphen_spektren_daten
        )
        feature_matrix = np.array(feature_list)
        
        # Definiere Feature-Namen
        feature_names = [
            "I(D)/I(G)",
            "FWHM(2D)",
            "I(2D)/I(G)",
            "Pos(G)",
            "Pos(2D)",
            "PMMA_ratio",
        ]
        
        # Erzeuge 2D-Feature-Maps für die Visualisierung
        feature_maps_2d = None
        try:
            feature_maps_2d = np.full((h, w, len(feature_names)), np.nan, dtype=float)
            valid_indices = np.where(graphen_mask_1d)[0]
            if feature_matrix.ndim == 2 and feature_matrix.shape[0] == len(valid_indices):
                y_coords, x_coords = np.unravel_index(valid_indices, (h, w))
                for i in range(len(valid_indices)):
                    feature_maps_2d[y_coords[i], x_coords[i], :] = feature_matrix[i]
        except Exception:
            logger.warning("Feature-Maps konnten nicht erstellt werden.")


        # --- 3. BEREINIGEN & SKALIEREN ---
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        feature_matrix_imputed = imputer.fit_transform(feature_matrix)
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix_imputed)

        # --- 4. PCA AUF FEATURES ---
        logger.info("Starte PCA auf Features...")
        pca = SklearnPCA(n_components=0.95, svd_solver='full', random_state=42)
        scores_np = pca.fit_transform(scaled_features)
        
        optimal_pcs_gefunden = pca.n_components_
        logger.info(f"PCA fertig. {optimal_pcs_gefunden} PCs genutzt.")

        # --- 5. SOM CLUSTERING ---
        logger.info("Starte SOM-Training...")
        
        # EINSTELLUNGEN:
        # Ein 3x3 Gitter ergibt 9 mögliche Cluster (Neuronen).
        # Benachbarte Nummern sind sich physikalisch ähnlich!
        map_height = 3 # m
        map_width = 3  # n
        
        # Initialisiere SOM
        # dim: Anzahl der Dimensionen im Inputraum (hier: optimale PCs)
        graphen_som = SOM(m=map_height, n=map_width, dim=optimal_pcs_gefunden, lr=0.5, random_state=42) # type: ignore
        
        # Trainiere SOM (fit) und sage Cluster vorher (predict)
        # epochs=5 sorgt für stabilere Karten
        graphen_som.fit(scores_np, epochs=5) 
        
        # Predict gibt jedem Punkt eine Zahl von 0 bis (m*n - 1)
        graphen_cluster_labels = graphen_som.predict(scores_np)
        
        logger.info("SOM-Training abgeschlossen.")

        # --- 6. ERGEBNISSE ZUSAMMENFÜHREN ---
        final_cluster_map_1d = np.full(h*w, np.nan)
        SUBSTRAT_LABEL = -1 # Wir nehmen -1 für Substrat
        
        final_cluster_map_1d[substrat_mask_1d] = SUBSTRAT_LABEL
        final_cluster_map_1d[graphen_mask_1d] = graphen_cluster_labels
        final_cluster_map_2d = final_cluster_map_1d.reshape((h, w))

        # --- 7. IDENTIFIZIERUNG ---
        unique_final_labels = sorted([label for label in np.unique(final_cluster_map_1d) if not np.isnan(label)])
        
        mean_spectra_graphen = []
        mean_spectra_silizium = []
        gefundene_cluster_ids = []

        for label in unique_final_labels:
            cluster_mask_1d = (final_cluster_map_1d == label)
            cluster_mask_2d = cluster_mask_1d.reshape((h, w))
            if np.any(cluster_mask_2d):
                mean_spectra_graphen.append(karte_graphen[cluster_mask_2d].mean)
                mean_spectra_silizium.append(karte_silizium[cluster_mask_2d].mean)
                gefundene_cluster_ids.append(int(label))
        
        neue_labels_map = identifiziere_cluster(mean_spectra_graphen,
                                           mean_spectra_silizium, 
                                           gefundene_cluster_ids, 
                                           substrat_label=SUBSTRAT_LABEL)
        
        finale_plot_labels = [f"Neuron {original_id}: {neue_labels_map.get(f'Cluster {i}', 'Unbekannt')}"
                              for i, original_id in enumerate(gefundene_cluster_ids)]
        
        # Y-Limit für Plot
        global_max_intensity = 0
        for spectrum in mean_spectra_graphen:
             if spectrum.spectral_data.size > 0:
                current_max = np.max(spectrum.spectral_data)
                if current_max > global_max_intensity: 
                    global_max_intensity = current_max
        plot_ylim = global_max_intensity * 1.1

        return {
            "success": True,
            "cluster_map": final_cluster_map_2d,
            "unique_labels": unique_final_labels,
            "mean_spectra": mean_spectra_graphen,
            "plot_labels": finale_plot_labels,
            "y_limit": plot_ylim,
            "map_title": f"Finale Cluster-Karte (SOM {map_height}x{map_width})",
            "feature_maps": feature_maps_2d,
            "feature_names": feature_names
        }

    except Exception as e:
        logger.error(f"FEHLER bei SOM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path): # type: ignore
            os.remove(temp_file_path) # type: ignore