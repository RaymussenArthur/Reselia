"""
RESILIA — Resilience & Infrastructure Logistics Analyzer
Streamlit Dashboard v1.0 — Streamlit Cloud deployment
"""

import copy
import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import folium
import streamlit as st
from streamlit_folium import st_folium
import osmnx as ox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RESILIA — Urban Risk Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow:wght@300;400;500;600;700;800&family=Barlow+Condensed:wght@400;600;700;800&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
  }
  .stApp {
    background: #040810;
    color: #b8ccd8;
  }

  /* ── Noise texture overlay ── */
  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #020608;
    border-right: 1px solid #0c1520;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #3d5a70 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
  }
  section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #0a4d8c 0%, #0d6aa8 100%);
    color: #e0eaf4;
    border: none;
    border-radius: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 12px 0;
    transition: all 0.2s;
    box-shadow: 0 2px 12px rgba(13, 95, 166, 0.3);
  }
  section[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(135deg, #0d6aa8 0%, #1180c8 100%);
    box-shadow: 0 4px 20px rgba(13, 95, 166, 0.5);
    transform: translateY(-1px);
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #070d14;
    border: 1px solid #0c1828;
    border-radius: 2px;
    padding: 18px 22px 16px;
    position: relative;
    overflow: hidden;
  }
  div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0d5fa6 0%, #0d5fa620 100%);
  }
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #4a7a9a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase;
    letter-spacing: 0.18em;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c8d8e8 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.24em;
    color: #2d7ab0;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 40px 0 20px;
  }
  .section-header::before {
    content: '';
    width: 24px;
    height: 1px;
    background: #1d5a8a;
    flex-shrink: 0;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #0c1828 0%, transparent 100%);
  }

  /* ── Risk badges ── */
  .badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    padding: 6px 20px;
    border-radius: 2px;
    text-transform: uppercase;
  }
  .badge-low      { background: #041208; color: #2da44e; border: 1px solid #133326; }
  .badge-moderate { background: #110d04; color: #d49a10; border: 1px solid #332508; }
  .badge-high     { background: #110404; color: #e8554e; border: 1px solid #330c0c; }
  .badge-critical { background: #1a0404; color: #ff7070; border: 1px solid #550e0e;
                    box-shadow: 0 0 16px rgba(255,80,80,0.15); }

  /* ── Expander ── */
  div[data-testid="stExpander"] {
    background: #070d14 !important;
    border: 1px solid #0c1828 !important;
    border-radius: 2px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #3d5a70 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* ── Alert ── */
  div[data-testid="stAlert"] {
    background: #110d04 !important;
    border: 1px solid #332508 !important;
    border-radius: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
  }

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #070d14;
    border-bottom: 1px solid #0c1828;
    gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #3d5a70;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 10px 20px;
    transition: all 0.15s;
  }
  .stTabs [aria-selected="true"] {
    background: transparent !important;
    border-bottom: 2px solid #0d5fa6 !important;
    color: #7aabcc !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: transparent;
    padding-top: 20px;
  }

  /* ── Divider ── */
  hr {
    border: none;
    border-top: 1px solid #0c1828;
    margin: 24px 0;
  }

  /* ── Caption ── */
  .stCaption {
    color: #3a6a8a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 0.08em;
  }

  /* ── Status widget ── */
  div[data-testid="stStatusWidget"] {
    background: #070d14 !important;
    border: 1px solid #0c1828 !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
  }

  /* ── Dataframe ── */
  .stDataFrame {
    border: 1px solid #0c1828 !important;
    border-radius: 2px !important;
  }

  /* ── Selectbox / radio ── */
  .stSelectbox > div > div,
  .stRadio > div {
    background: #070d14 !important;
    border-color: #0c1828 !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-track { background: #040810; }
  ::-webkit-scrollbar-thumb { background: #0d1e2e; border-radius: 2px; }

  /* ── Info panel card ── */
  .info-card {
    background: #070d14;
    border: 1px solid #0c1828;
    border-radius: 2px;
    padding: 18px 22px;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    line-height: 1.9;
    color: #2e4a5e;
  }
  .info-card .label {
    color: #1d3a50;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 8px;
    margin-bottom: 8px;
    display: block;
  }
  .info-card .value {
    color: #7aabcc;
    font-size: 11px;
  }
</style>
""", unsafe_allow_html=True)

# ── Plot style (matches notebook) ─────────────────────────────────────────────
DARK_BG   = "#0d1117"
DARK_SURF = "#0b1420"
DARK_LINE = "#0c1828"
C_BLUE    = "#0d5fa6"
C_RED     = "#e5534b"
C_TEAL    = "#2a9d8f"
C_YELLOW  = "#e9c46a"
C_TEXT    = "#c8d4e0"
C_MUTED   = "#7a9ab0"

plt.rcParams.update({
    "figure.facecolor"  : DARK_BG,
    "axes.facecolor"    : DARK_SURF,
    "axes.edgecolor"    : DARK_LINE,
    "axes.labelcolor"   : C_MUTED,
    "axes.titlecolor"   : C_TEXT,
    "xtick.color"       : C_MUTED,
    "ytick.color"       : C_MUTED,
    "text.color"        : C_TEXT,
    "grid.color"        : DARK_LINE,
    "legend.facecolor"  : DARK_BG,
    "legend.edgecolor"  : DARK_LINE,
    "font.family"       : "monospace",
    "axes.titleweight"  : "bold",
    "axes.titlesize"    : 11,
})

# ── Constants ─────────────────────────────────────────────────────────────────
FLOOD_THRESHOLD_M = 2.5

AREA_CONFIGS = {
    "Kemayoran"  : {"center": (-6.1600, 106.8600), "dist": 2200, "adm4": "31.71.03.1001"},
    "Penjaringan": {"center": (-6.1200, 106.8000), "dist": 2000, "adm4": "31.71.01.1001"},
    "Pluit"      : {"center": (-6.1100, 106.7900), "dist": 1800, "adm4": "31.71.01.1004"},
    "Cengkareng" : {"center": (-6.1500, 106.7400), "dist": 2000, "adm4": "31.73.01.1001"},
}

WEATHER_WEIGHTS = {
    "Cerah": 0.05, "Cerah Berawan": 0.10, "Berawan": 0.15,
    "Hujan Ringan": 0.45, "Hujan Sedang": 0.65,
    "Hujan Lebat": 0.85, "Hujan Petir": 0.95,
}

BADGE_CLASS = {
    "LOW": "badge-low", "MODERATE": "badge-moderate",
    "HIGH": "badge-high", "CRITICAL": "badge-critical"
}
TIER_COLOR  = {"LOW": "#2da44e", "MODERATE": "#d49a10", "HIGH": "#e8554e", "CRITICAL": "#ff7070"}
TIER_BG     = {"LOW": "#041208", "MODERATE": "#110d04", "HIGH": "#110404", "CRITICAL": "#1a0404"}
TIER_BORDER = {"LOW": "#133326", "MODERATE": "#332508", "HIGH": "#330c0c", "CRITICAL": "#550e0e"}

# Custom colormap matching notebook
CMAP_RESILIA = mcolors.LinearSegmentedColormap.from_list(
    "resilia", ["#0b1a2e", "#1e3a5f", "#4a7fa5", "#c0622a", "#e5534b"], N=256
)

FEAT_COLS = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
FEAT_COLS_DISPLAY = ['elevation'] + FEAT_COLS  # elevation kept for EDA/visualization only


# ── Core pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_network(area_name):
    cfg = AREA_CONFIGS[area_name]
    ox.settings.timeout = 120
    ox.settings.log_console = False
    G = ox.graph_from_point(cfg["center"], dist=cfg["dist"], network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G)
    return G, nodes, edges


def inject_elevation(G, area_name):
    cfg = AREA_CONFIGS[area_name]
    lat_south = cfg["center"][0] - cfg["dist"] / 111000
    for node, data in G.nodes(data=True):
        elev = round(max(0.0, 2.0 + (data["y"] - lat_south) * 150), 2)
        G.nodes[node]["elevation"]   = elev
        G.nodes[node]["flood_label"] = 1 if elev < FLOOD_THRESHOLD_M else 0
    return G


def build_ml_model(G):
    """
    Build RF model using 3 graph-topology features only (NO elevation).
    Elevation is the label source — including it causes data leakage/overfit.
    Uses predict_proba for continuous risk scores.
    """
    degree_c      = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, k=200, normalized=True, seed=42)
    closeness_c   = nx.closeness_centrality(G)

    records = [{
        "node_id"               : n,
        "elevation"             : d.get("elevation", 5.0),
        "degree_centrality"     : degree_c[n],
        "betweenness_centrality": betweenness_c[n],
        "closeness_centrality"  : closeness_c[n],
        "flood_label"           : d.get("flood_label", 0),
    } for n, d in G.nodes(data=True)]

    df = pd.DataFrame(records)
    X  = df[FEAT_COLS].values
    y  = df["flood_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred    = rf.predict(X_test)
    f1        = f1_score(y_test, y_pred, average='weighted')
    f1_mac    = f1_score(y_test, y_pred, average='macro')
    acc       = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(
        rf, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1_weighted', n_jobs=-1
    )
    cm = confusion_matrix(y_test, y_pred)

    # Continuous risk score (predict_proba)
    risk_scores      = rf.predict_proba(X)[:, 1]
    df['risk_score'] = risk_scores
    df['ml_pred']    = rf.predict(X)

    # Inject back into graph
    for i, row in df.iterrows():
        G.nodes[row['node_id']]['vulnerability'] = "High" if df.at[i, 'ml_pred'] == 1 else "Low"
        G.nodes[row['node_id']]['risk_score']    = df.at[i, 'risk_score']

    return G, df, rf, f1, f1_mac, cv_scores, acc, cm, y_test, y_pred


@st.cache_data(show_spinner=False, ttl=900)
def fetch_bmkg(adm4):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        block = r.json().get("data", [])
        if block:
            cuaca = block[0].get("cuaca", [])
            desc  = (cuaca[0][0] if cuaca and cuaca[0] else {}).get("weather_desc", "Unknown")
        else:
            desc = "Unknown"
        w = WEATHER_WEIGHTS.get(desc, 0.85 if "Hujan" in desc else 0.10)
        return desc, w, True
    except Exception:
        return "Heavy Rain (Fallback)", 0.85, False


def compute_risk(G, stressor_weight):
    vulnerable = [n for n, d in G.nodes(data=True) if d.get("vulnerability") == "High"]
    n_total    = G.number_of_nodes()
    sfp        = stressor_weight * (len(vulnerable) / n_total) * 100
    tier = ("CRITICAL" if sfp >= 15 else "HIGH" if sfp >= 5
            else "MODERATE" if sfp >= 1 else "LOW")
    return vulnerable, n_total, sfp, tier


def compute_resilience(G, vulnerable_nodes):
    """Network resilience analysis — matches notebook Cell 18."""
    G_und = G.to_undirected()
    baseline_comps = nx.number_connected_components(G_und)
    baseline_lcc   = len(max(nx.connected_components(G_und), key=len))

    G_flooded = copy.deepcopy(G_und)
    G_flooded.remove_nodes_from(vulnerable_nodes)
    post_comps     = nx.number_connected_components(G_flooded)
    post_lcc_nodes = max(nx.connected_components(G_flooded), key=len)
    post_lcc       = len(post_lcc_nodes)

    connectivity_loss = (baseline_lcc - post_lcc) / baseline_lcc * 100
    new_iso_clusters  = post_comps - baseline_comps

    return dict(
        baseline_lcc=baseline_lcc,
        post_lcc=post_lcc,
        connectivity_loss=connectivity_loss,
        new_iso_clusters=new_iso_clusters,
        post_lcc_nodes=post_lcc_nodes,
    )


def rank_chokepoints(df):
    """Critical chokepoint ranking — matches notebook Cell 20."""
    critical_df = df[df['ml_pred'] == 1].copy()
    critical_df = critical_df.sort_values('betweenness_centrality', ascending=False).reset_index(drop=True)
    critical_df.index += 1
    return critical_df


def build_folium_map(G, edges, vulnerable, area, weather, sfp, tier, f1, n_critical):
    cfg = AREA_CONFIGS[area]
    m   = folium.Map(location=list(cfg["center"]), zoom_start=14, tiles="CartoDB dark_matter")

    for _, row in edges.iterrows():
        folium.PolyLine(
            [(lat, lon) for lon, lat in row.geometry.coords],
            color="#1d3a52", weight=1.2, opacity=0.55
        ).add_to(m)

    for n in G.nodes():
        d     = G.nodes[n]
        score = d.get('risk_score', 0.0)
        rgba  = CMAP_RESILIA(score)
        hex_c = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        radius  = 2 + score * 4
        opacity = 0.25 + score * 0.65

        folium.CircleMarker(
            location=(d["y"], d["x"]),
            radius=radius,
            color="none",
            fill=True, fill_color=hex_c, fill_opacity=opacity,
            tooltip=(
                f"Node {n}<br>"
                f"Elev: {d.get('elevation', '?')} m<br>"
                f"Risk Score: {d.get('risk_score', 0):.3f}<br>"
                f"Class: {d.get('vulnerability', '?')}"
            )
        ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#040810;padding:18px 22px;border-radius:2px;
                border:1px solid #0c1828;border-top:2px solid #0d5fa6;
                font-family:'Space Mono',monospace;font-size:10px;
                color:#7aabcc;box-shadow:0 8px 32px rgba(0,0,0,0.8);min-width:190px;">
      <div style="color:#c8d8e8;font-weight:700;letter-spacing:0.14em;
                  text-transform:uppercase;font-size:9px;margin-bottom:12px;">
        RESILIA · Risk Output
      </div>
      <div style="margin-bottom:6px;">
        <span style="color:#2e4a5e;text-transform:uppercase;font-size:8px;letter-spacing:0.14em;">
          High-risk nodes
        </span><br>
        <span style="color:#e5534b;font-weight:700;font-size:16px;">{n_critical:,}</span>
      </div>
      <div style="border-top:1px solid #0c1828;margin:10px 0;"></div>
      RF F1 &nbsp;{f1:.4f}<br>
      Weather &nbsp;{weather}<br>
      SFP &nbsp;<b style="color:{TIER_COLOR[tier]};">{sfp:.2f}% · {tier}</b>
      <div style="margin-top:14px;margin-bottom:4px;font-size:8px;color:#1d3a50;
                  text-transform:uppercase;letter-spacing:0.14em;">Risk Score</div>
      <div style="height:8px;width:150px;border-radius:2px;
                  background:linear-gradient(to right,#0b1a2e,#1e3a5f,#4a7fa5,#c0622a,#e5534b);
                  margin-bottom:4px;"></div>
      <div style="display:flex;justify-content:space-between;font-size:8px;
                  width:150px;color:#2e4a5e;">
        <span>0.0 Low</span><span>1.0 High</span>
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 32px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:800;
                  color:#c8d8e8;letter-spacing:-0.02em;line-height:1;
                  text-transform:uppercase;">
        RESILIA
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:8px;
                  color:#3a6a8a;margin-top:6px;text-transform:uppercase;
                  letter-spacing:0.2em;">
        Urban Flood Risk Engine v2.0
      </div>
      <div style="margin-top:10px;height:1px;background:linear-gradient(90deg,#0d5fa6,transparent);"></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    selected_area = st.selectbox("Study Area", list(AREA_CONFIGS.keys()), index=0)

    st.markdown('<div class="section-header">Execute</div>', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis Pipeline", use_container_width=True, type="primary")
    st.caption("Network cached 1 hr  ·  Weather refreshes 15 min")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:9px;
                color:#3a6a8a;line-height:2.2;">
      <span style="color:#5a8aaa;text-transform:uppercase;letter-spacing:0.14em;
                   font-size:8px;">Data Sources</span><br>
      OSM via osmnx (ODbL)<br>
      BMKG Public API<br>
      DEM: simulated (PoC)<br><br>
      <span style="color:#5a8aaa;text-transform:uppercase;letter-spacing:0.14em;
                   font-size:8px;">ML Model</span><br>
      Random Forest (Phase 1)<br>
      Phase 2: GraphSAGE / GAT<br>
      Azure ML deployment
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 4px;">
  <div style="display:flex;align-items:baseline;gap:14px;">
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:42px;font-weight:800;
                color:#c8d8e8;letter-spacing:-0.03em;line-height:1;text-transform:uppercase;">
      RESILIA
    </div>
    <div style="font-family:'Barlow',sans-serif;font-weight:300;color:#2e5a7a;font-size:16px;
                letter-spacing:0.04em;">
      Urban Infrastructure Risk Engine
    </div>
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:9px;
              color:#3a6a8a;margin-top:8px;letter-spacing:0.06em;">
    Flood vulnerability assessment &nbsp;·&nbsp; OpenStreetMap · Random Forest · BMKG telemetry
    &nbsp;·&nbsp; Network resilience simulation
  </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#0d5fa6 0%,#0d5fa640 30%,transparent 70%);
            margin:18px 0 28px;"></div>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    with st.status("Running RESILIA pipeline...", expanded=True) as status:
        try:
            st.write("① Fetching road network from OpenStreetMap...")
            G, nodes, edges = fetch_network(selected_area)
            st.write(f"   ✓ {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")

            st.write("② Injecting terrain elevation model (DEM)...")
            G = inject_elevation(G, selected_area)
            st.write("   ✓ Elevation scores assigned, flood labels generated")

            st.write("③ Engineering graph features + training Random Forest...")
            G, df, rf, f1, f1_mac, cv_scores, acc, cm, y_test, y_pred = build_ml_model(G)
            st.write(f"   ✓ F1={f1:.4f} · CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            st.write("④ Fetching BMKG live weather telemetry...")
            adm4 = AREA_CONFIGS[selected_area]["adm4"]
            weather, stressor_w, live = fetch_bmkg(adm4)
            st.write(f"   ✓ {weather} · stressor={stressor_w} ({'live' if live else 'fallback'})")

            st.write("⑤ Computing Systemic Failure Probability...")
            vulnerable, n_total, sfp, tier = compute_risk(G, stressor_w)
            st.write(f"   ✓ SFP={sfp:.2f}% · Tier: {tier}")

            st.write("⑥ Simulating network resilience under flood scenario...")
            resilience = compute_resilience(G, vulnerable)
            st.write(f"   ✓ Connectivity loss: {resilience['connectivity_loss']:.1f}%")

            st.write("⑦ Ranking critical chokepoints by betweenness centrality...")
            critical_df = rank_chokepoints(df)
            st.write(f"   ✓ Top {min(15, len(critical_df))} chokepoints identified")

            st.session_state.results = dict(
                G=G, nodes=nodes, edges=edges, df=df,
                rf=rf, f1=f1, f1_mac=f1_mac, acc=acc,
                cv_scores=cv_scores, cm=cm, y_test=y_test, y_pred=y_pred,
                weather=weather, stressor_w=stressor_w, live=live,
                vulnerable=vulnerable, n_total=n_total,
                sfp=sfp, tier=tier, area=selected_area,
                resilience=resilience, critical_df=critical_df,
            )
            status.update(label="Pipeline complete — all 7 modules executed", state="complete")
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.error(str(e))


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    r = st.session_state.results

    # Auto-collapse sidebar via JS click on the collapse button
    st.markdown("""
    <script>
    (function() {
      function tryCollapse() {
        // Try multiple selectors for different Streamlit versions
        var btn = document.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                  document.querySelector('[data-testid="collapsedControl"]') ||
                  document.querySelector('button[kind="header"][aria-label*="sidebar"]') ||
                  document.querySelector('.stSidebar button[kind="header"]');
        if (btn) {
          var sidebar = document.querySelector('[data-testid="stSidebar"]');
          if (sidebar && sidebar.getBoundingClientRect().width > 50) {
            btn.click();
          }
        } else {
          setTimeout(tryCollapse, 300);
        }
      }
      setTimeout(tryCollapse, 600);
    })();
    </script>
    """, unsafe_allow_html=True)

    if not r["live"]:
        st.warning("BMKG API unavailable — stressor weight defaulted to 0.85 (Heavy Rain fallback).")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Risk Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Study Area",      r["area"][:12] + ("..." if len(r["area"]) > 12 else ""),
              delta=r["area"] if len(r["area"]) > 12 else None)
    c2.metric("Total Nodes",     f"{r['n_total']:,}")
    c3.metric("High-Risk Nodes", f"{len(r['vulnerable']):,}",
              delta=f"{len(r['vulnerable'])/r['n_total']*100:.1f}% exposed",
              delta_color="inverse")
    c4.metric("SFP",             f"{r['sfp']:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RF F1 (weighted)",    f"{r['f1']:.3f}")
    c6.metric("CV F1 mean",          f"{r['cv_scores'].mean():.3f}",
              delta=f"std {r['cv_scores'].std():.3f}")
    c7.metric("Connectivity Loss",   f"{r['resilience']['connectivity_loss']:.1f}%")
    c8.metric("Weather",             r["weather"][:20])

    # Risk badge
    st.markdown("<div style='margin:18px 0 8px;'>", unsafe_allow_html=True)
    _, badge_col, _ = st.columns([1, 2, 5])
    with badge_col:
        st.markdown(
            f'<span class="badge {BADGE_CLASS[r["tier"]]}">'
            f'Risk Tier: {r["tier"]}</span>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Inline Spatial Map (always visible) ───────────────────────────────────
    st.markdown('<div class="section-header">Spatial Vulnerability Map</div>', unsafe_allow_html=True)

    fmap = build_folium_map(
        r["G"], r["edges"], r["vulnerable"],
        r["area"], r["weather"], r["sfp"], r["tier"], r["f1"],
        len(r["vulnerable"])
    )
    st_folium(fmap, width="100%", height=520, returned_objects=[])
    st.caption("Node color & size encodes RF predict_proba risk score (0–1) · Hover for node details")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Evaluation",
        "EDA — Feature Analysis",
        "Network Resilience",
        "Critical Chokepoints",
    ])

    # ── Tab 1: Model Evaluation ───────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Model Evaluation — Random Forest Baseline</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",          f"{r['acc']:.4f}")
        m2.metric("F1 (weighted)",      f"{r['f1']:.4f}")
        m3.metric("F1 (macro)",         f"{r['f1_mac']:.4f}")
        m4.metric("CV F1 Mean (5-fold)", f"{r['cv_scores'].mean():.4f}")

        fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig2.suptitle("Random Forest — Evaluation Suite", fontsize=12, fontweight="bold")

        # Confusion Matrix
        ConfusionMatrixDisplay(r['cm'], display_labels=['Low Risk', 'High Risk']).plot(
            ax=axes[0], colorbar=False, cmap='Blues'
        )
        axes[0].set_title("Confusion Matrix")
        axes[0].set_facecolor(DARK_SURF)

        # Feature Importance (4 features from notebook)
        imp_df = pd.DataFrame({
            "feature"   : FEAT_COLS,
            "importance": r["rf"].feature_importances_
        }).sort_values("importance")
        bar_colors = [C_RED if i == imp_df["importance"].idxmax() else C_BLUE for i in imp_df.index]
        axes[1].barh(imp_df["feature"], imp_df["importance"],
                     color=bar_colors, edgecolor=DARK_BG, height=0.5)
        axes[1].set_title("Feature Importance (MDI)")
        axes[1].set_xlabel("Mean Decrease in Impurity")

        # CV Scores
        axes[2].bar(range(1, 6), r["cv_scores"], color=C_BLUE, edgecolor=DARK_BG, width=0.55)
        axes[2].axhline(r["cv_scores"].mean(), color=C_RED, linewidth=1.6,
                        linestyle="--", label=f"Mean = {r['cv_scores'].mean():.4f}")
        axes[2].set_title("5-Fold CV F1-Score (weighted)")
        axes[2].set_xlabel("Fold")
        axes[2].set_ylabel("F1")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xticks(range(1, 6))
        axes[2].set_xticklabels([f"Fold {i}" for i in range(1, 6)])
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

        with st.expander("Classification Report"):
            report = classification_report(r['y_test'], r['y_pred'],
                                           target_names=['Low Risk', 'High Risk'])
            st.code(report, language=None)

    # ── Tab 2: EDA ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Exploratory Data Analysis — Feature Matrix</div>',
                    unsafe_allow_html=True)

        df = r["df"]
        df['label_str'] = df['flood_label'].map({0: 'Low Risk', 1: 'High Risk'})

        fig3, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig3.suptitle("EDA — RESILIA Feature Matrix", fontsize=13, fontweight="bold", y=1.01)

        # Correlation heatmap (pure matplotlib)
        corr = df[FEAT_COLS_DISPLAY + ['flood_label']].corr()
        corr_vals = corr.values
        im = axes[0, 0].imshow(corr_vals, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        clabels = list(corr.columns)
        axes[0, 0].set_xticks(range(len(clabels)))
        axes[0, 0].set_yticks(range(len(clabels)))
        axes[0, 0].set_xticklabels(clabels, rotation=45, ha="right", fontsize=7)
        axes[0, 0].set_yticklabels(clabels, fontsize=7)
        for i in range(len(clabels)):
            for j in range(len(clabels)):
                axes[0, 0].text(j, i, f"{corr_vals[i, j]:.2f}",
                                ha="center", va="center", fontsize=7, color=C_TEXT)
        axes[0, 0].set_title("Feature Correlation Matrix")

        # Class balance
        counts = df['flood_label'].value_counts().sort_index()
        bars = axes[0, 1].bar(
            ['Low Risk', 'High Risk'],
            [counts.get(0, 0), counts.get(1, 0)],
            color=[C_TEAL, C_RED], edgecolor=DARK_BG, width=0.5
        )
        axes[0, 1].set_title("Class Balance")
        axes[0, 1].set_ylabel("Node Count")
        for bar, v in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                            v + 10, f"{v:,}", ha='center', fontsize=10, fontweight='bold', color=C_TEXT)

        # Elevation by class (pure matplotlib boxplot)
        low_elev  = df[df['flood_label'] == 0]['elevation'].values
        high_elev = df[df['flood_label'] == 1]['elevation'].values
        bp = axes[0, 2].boxplot([low_elev, high_elev], labels=['Low Risk', 'High Risk'],
                                patch_artist=True, widths=0.45,
                                medianprops=dict(color=C_TEXT, linewidth=1.5))
        bp['boxes'][0].set_facecolor(C_TEAL)
        bp['boxes'][1].set_facecolor(C_RED)
        for element in ['whiskers', 'caps', 'fliers']:
            for item in bp[element]:
                item.set_color(C_MUTED)
        axes[0, 2].axhline(FLOOD_THRESHOLD_M, color=C_RED, linestyle='--',
                           linewidth=1.2, label=f'Threshold ({FLOOD_THRESHOLD_M} m)')
        axes[0, 2].set_title("Elevation by Class")
        axes[0, 2].set_xlabel("")
        axes[0, 2].set_ylabel("Elevation (m)")
        axes[0, 2].legend(fontsize=9)

        # Centrality distributions
        for idx, (feat, color) in enumerate([
            ('degree_centrality',      C_BLUE),
            ('betweenness_centrality', C_YELLOW),
            ('closeness_centrality',   "#8ecae6"),
        ]):
            axes[1, idx].hist(df[feat], bins=40, color=color,
                              edgecolor=DARK_BG, linewidth=0.3, alpha=0.85)
            axes[1, idx].set_title(feat.replace('_', ' ').title())
            axes[1, idx].set_ylabel("Count")

        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    # ── Tab 3: Network Resilience ─────────────────────────────────────────────
    with tab3:
        res = r["resilience"]
        st.markdown('<div class="section-header">Network Resilience — Flood Impact Simulation</div>',
                    unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Baseline LCC",       f"{res['baseline_lcc']:,}")
        r2.metric("Post-Flood LCC",     f"{res['post_lcc']:,}",
                  delta=f"-{res['baseline_lcc'] - res['post_lcc']:,} nodes",
                  delta_color="inverse")
        r3.metric("Connectivity Loss",  f"{res['connectivity_loss']:.1f}%", delta_color="inverse")
        r4.metric("New Isolated Clusters", f"{res['new_iso_clusters']}")

        fig4, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig4.suptitle("Network Resilience — Flood Impact Analysis", fontsize=12, fontweight="bold")

        # LCC comparison
        cats   = ['Baseline LCC', 'Post-Flood LCC']
        vals   = [res['baseline_lcc'], res['post_lcc']]
        bcols  = [C_BLUE, C_RED]
        bars   = axes[0].bar(cats, vals, color=bcols, edgecolor=DARK_BG, width=0.45)
        axes[0].set_title("Largest Connected Component")
        axes[0].set_ylabel("Node Count")
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 10,
                         f"{v:,}", ha='center', fontsize=10, fontweight='bold', color=C_TEXT)

        # Node composition waterfall
        n_removed  = len(r["vulnerable"])
        n_isolated = r["n_total"] - n_removed - res['post_lcc']
        comp_labels = ['Total Nodes', 'High Risk\n(removed)', 'Remaining\nConnected', 'Isolated\nFragments']
        comp_values = [r["n_total"], n_removed, res['post_lcc'], max(0, n_isolated)]
        comp_colors = [C_BLUE, C_RED, C_TEAL, C_YELLOW]
        axes[1].bar(comp_labels, comp_values, color=comp_colors, edgecolor=DARK_BG, width=0.55)
        axes[1].set_title("Node Composition — Flood Scenario")
        axes[1].set_ylabel("Node Count")
        for i, (lbl, v) in enumerate(zip(comp_labels, comp_values)):
            axes[1].text(i, v + 10, f"{v:,}", ha='center', fontsize=9, color=C_TEXT)

        # Risk score distribution: high vs low (fixed bins for visibility)
        _df_r = r["df"]
        bins = np.linspace(0, 1, 25)
        axes[2].hist(_df_r[_df_r['ml_pred'] == 0]['risk_score'], bins=bins,
                     color=C_BLUE, alpha=0.7, label=f"Low Risk ({int((_df_r['ml_pred']==0).sum())})", edgecolor=DARK_BG)
        axes[2].hist(_df_r[_df_r['ml_pred'] == 1]['risk_score'], bins=bins,
                     color=C_RED, alpha=0.7, label=f"High Risk ({int((_df_r['ml_pred']==1).sum())})", edgecolor=DARK_BG)
        axes[2].axvline(0.5, color=C_YELLOW, linewidth=1.5, linestyle='--', label='Decision boundary')
        axes[2].set_title("Risk Score Distribution by Class")
        axes[2].set_xlabel("RF Risk Score  P(High Risk)")
        axes[2].set_ylabel("Node Count")
        axes[2].set_xlim(0, 1)
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

    # ── Tab 4: Chokepoints ────────────────────────────────────────────────────
    with tab4:
        cdf = r["critical_df"]
        st.markdown('<div class="section-header">Critical Chokepoint Ranking</div>',
                    unsafe_allow_html=True)
        st.caption("High-risk nodes ranked by betweenness centrality — logistics disruption potential")

        fig5, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig5.suptitle("Critical Chokepoint Analysis — Top 10 High-Risk Nodes",
                       fontsize=12, fontweight="bold")

        top10 = cdf.head(10)
        norm      = mcolors.Normalize(vmin=0, vmax=1)
        colors    = [plt.cm.plasma(norm(v)) for v in top10['risk_score'].values]

        # Betweenness
        axes[0].barh(range(len(top10)), top10['betweenness_centrality'],
                     color=colors, edgecolor=DARK_BG, height=0.6)
        axes[0].set_yticks(range(len(top10)))
        axes[0].set_yticklabels([f"#{i+1}" for i in range(len(top10))], fontsize=9)
        axes[0].set_title("Betweenness Centrality (Chokepoint Severity)")
        axes[0].set_xlabel("Betweenness Centrality")
        axes[0].invert_yaxis()
        sm = ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[0], shrink=0.7, pad=0.02)
        cbar.set_label('Risk Score', labelpad=8)

        # Risk scores
        axes[1].barh(range(len(top10)), top10['risk_score'],
                     color=colors, edgecolor=DARK_BG, height=0.6)
        axes[1].axvline(0.5, color=C_YELLOW, linewidth=1.2, linestyle='--',
                        label='Decision boundary (0.5)')
        axes[1].set_yticks(range(len(top10)))
        axes[1].set_yticklabels([f"#{i+1}" for i in range(len(top10))], fontsize=9)
        axes[1].set_xlim(0, 1)
        axes[1].set_title("RF Risk Score  P(High)")
        axes[1].set_xlabel("Risk Score")
        axes[1].invert_yaxis()
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)

        # Top 15 table
        st.markdown('<div class="section-header">Top 15 Nodes</div>', unsafe_allow_html=True)
        top15 = cdf[['node_id', 'elevation', 'betweenness_centrality',
                      'closeness_centrality', 'risk_score']].head(15).copy()
        top15.columns = ['Node ID', 'Elevation (m)', 'Betweenness', 'Closeness', 'Risk Score']
        top15 = top15.round(5)
        top15.index = range(1, len(top15) + 1)
        st.dataframe(top15, use_container_width=True)

    # ── Policy Recommendation ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Policy Recommendation</div>', unsafe_allow_html=True)
    c   = TIER_COLOR[r["tier"]]
    bg  = TIER_BG[r["tier"]]
    bdr = TIER_BORDER[r["tier"]]

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {bdr};border-left:3px solid {c};
                border-radius:2px;padding:24px 28px;
                font-family:'Space Mono',monospace;">
      <div style="display:flex;align-items:baseline;gap:18px;margin-bottom:16px;flex-wrap:wrap;">
        <span style="color:{c};font-weight:700;font-size:10px;
                     text-transform:uppercase;letter-spacing:0.14em;">
          {r['tier']} RISK
        </span>
        <span style="color:#5a8aaa;font-size:9px;letter-spacing:0.06em;">
          SFP {r['sfp']:.2f}% &nbsp;·&nbsp; {r['weather']} &nbsp;·&nbsp;
          stressor={r['stressor_w']:.2f} &nbsp;·&nbsp;
          connectivity loss={r['resilience']['connectivity_loss']:.1f}%
        </span>
      </div>
      <div style="color:#8ab0c8;font-size:13px;line-height:1.9;font-family:'Barlow',sans-serif;font-weight:400;">
        Prioritize emergency drainage remediation across
        <b style="color:#c8d8e8;">{len(r['vulnerable']):,} RF-identified flood-prone nodes</b>
        in {r['area']}.
        Reroute logistics corridors away from the
        <b style="color:#c8d8e8;">{len(r['resilience'].get('post_lcc_nodes', []))}-node</b>
        post-flood connected component.
        Deploy real-time sensor network at the
        <b style="color:#c8d8e8;">top 15 chokepoint intersections</b>
        identified by betweenness centrality analysis.
        Phase 2 upgrade: replace Random Forest baseline with GraphSAGE/GAT on
        Azure ML for improved spatial generalization.
      </div>
    </div>""", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="padding: 120px 0 100px;text-align:center;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;
                  text-transform:uppercase;letter-spacing:0.28em;color:#2d6a9f;
                  margin-bottom:24px;">
        System Idle — Awaiting Input
      </div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:40px;font-weight:800;
                  color:#4a7fa8;line-height:1.15;max-width:520px;margin:0 auto 20px;
                  text-transform:uppercase;letter-spacing:-0.01em;">
        Select a study area and run the analysis pipeline.
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;
                  color:#2d5a7a;letter-spacing:0.08em;">
        OSM &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; BMKG &nbsp;·&nbsp; Network Resilience
      </div>
    </div>""", unsafe_allow_html=True)