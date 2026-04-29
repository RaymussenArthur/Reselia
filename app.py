"""
RESILIA — Resilience & Infrastructure Logistics Analyzer
Streamlit Dashboard v1.0 — Streamlit Cloud deployment
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import streamlit as st
from streamlit_folium import st_folium
import osmnx as ox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report

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
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .stApp {
    background-color: #070b0f;
    color: #c8d4e0;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background-color: #0a0f14;
    border-right: 1px solid #131c26;
  }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label {
    color: #5a7080 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #5a7080 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  /* Sidebar button */
  section[data-testid="stSidebar"] .stButton button {
    background: #0d5fa6;
    color: #e8f0f8;
    border: none;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 10px 0;
    transition: background 0.15s;
  }
  section[data-testid="stSidebar"] .stButton button:hover {
    background: #1170bc;
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #0b1018;
    border: 1px solid #131c26;
    border-radius: 6px;
    padding: 16px 20px 14px;
    position: relative;
    overflow: hidden;
  }
  div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0d5fa6, transparent);
  }
  div[data-testid="metric-container"] label,
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #3f5a72 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c8d4e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px !important;
    font-weight: 500;
    letter-spacing: -0.01em;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #2d6a9f;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 18px;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #131c26;
  }

  /* ── Risk badges ── */
  .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 5px 16px;
    border-radius: 3px;
    text-transform: uppercase;
  }
  .badge-low      { background: #071a0e; color: #2da44e; border: 1px solid #1a4d2e; }
  .badge-moderate { background: #19130a; color: #c38a06; border: 1px solid #3d2e0e; }
  .badge-high     { background: #1a0808; color: #e5534b; border: 1px solid #4d1414; }
  .badge-critical { background: #260808; color: #ff6b6b; border: 1px solid #6e1a1a; }

  /* ── Status box / expander ── */
  div[data-testid="stExpander"] {
    background: #0b1018 !important;
    border: 1px solid #131c26 !important;
    border-radius: 6px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #5a7080 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em;
  }

  /* ── Warning/info ── */
  div[data-testid="stAlert"] {
    background: #19130a !important;
    border: 1px solid #3d2e0e !important;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  /* ── Divider ── */
  hr {
    border: none;
    border-top: 1px solid #131c26;
    margin: 20px 0;
  }

  /* ── Status widget ── */
  div[data-testid="stStatusWidget"] {
    background: #0b1018 !important;
    border: 1px solid #131c26 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
  }
  div[data-testid="stStatusWidget"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #7a9ab0 !important;
  }

  /* ── Caption ── */
  .stCaption {
    color: #2d4a5e !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.06em;
  }

  /* ── Selectbox / radio ── */
  .stSelectbox > div > div,
  .stRadio > div {
    background: #0b1018 !important;
    border-color: #131c26 !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #070b0f; }
  ::-webkit-scrollbar-thumb { background: #1d2a38; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

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

TIER_COLOR = {
    "LOW": "#2da44e", "MODERATE": "#c38a06",
    "HIGH": "#e5534b", "CRITICAL": "#ff6b6b"
}

TIER_BG = {
    "LOW": "#071a0e", "MODERATE": "#19130a",
    "HIGH": "#1a0808", "CRITICAL": "#260808"
}

TIER_BORDER = {
    "LOW": "#1a4d2e", "MODERATE": "#3d2e0e",
    "HIGH": "#4d1414", "CRITICAL": "#6e1a1a"
}


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
        G.nodes[node]["elevation"]  = elev
        G.nodes[node]["flood_label"] = 1 if elev < FLOOD_THRESHOLD_M else 0
    return G


def build_ml_model(G):
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
    FEAT = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']

    X, y = df[FEAT].values, df['flood_label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred    = rf.predict(X_test)
    f1        = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(
        rf, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1_weighted', n_jobs=-1
    )

    preds = rf.predict(X)
    df['ml_pred'] = preds
    for i, row in df.iterrows():
        G.nodes[row['node_id']]['vulnerability'] = "High" if preds[i] == 1 else "Low"

    return G, df, rf, FEAT, f1, cv_scores, accuracy_score(y_test, y_pred)


@st.cache_data(show_spinner=False, ttl=900)
def fetch_bmkg(adm4):
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        block = r.json().get("data", [])
        if block:
            cuaca = block[0].get("cuaca", [])
            desc  = (cuaca[0][0] if cuaca and cuaca[0] else {}).get("weather_desc", "Unknown")
        else:
            desc = "Unknown"
        w = WEATHER_WEIGHTS.get(desc, 0.85 if "Hujan" in desc else 0.10)
        return desc, w, True
    except:
        return "Heavy Rain (Fallback)", 0.85, False


def compute_risk(G, stressor_weight):
    vulnerable = [n for n, d in G.nodes(data=True) if d.get("vulnerability") == "High"]
    n_total    = G.number_of_nodes()
    sfp        = stressor_weight * (len(vulnerable) / n_total) * 100
    tier = ("CRITICAL" if sfp >= 15 else "HIGH" if sfp >= 5
            else "MODERATE" if sfp >= 1 else "LOW")
    return vulnerable, n_total, sfp, tier


def build_folium_map(G, edges, vulnerable, area, weather, sfp, tier, f1):
    cfg = AREA_CONFIGS[area]
    m   = folium.Map(
        location=list(cfg["center"]), zoom_start=14,
        tiles="CartoDB dark_matter"
    )
    for _, row in edges.iterrows():
        folium.PolyLine(
            [(lat, lon) for lon, lat in row.geometry.coords],
            color="#1d3a52", weight=1.2, opacity=0.6
        ).add_to(m)
    for nid in vulnerable:
        d = G.nodes[nid]
        folium.CircleMarker(
            location=(d["y"], d["x"]), radius=5,
            color="#e5534b", fill=True, fill_color="#e5534b", fill_opacity=0.85,
            tooltip=f"Node {nid} | Elev: {d['elevation']} m | HIGH RISK"
        ).add_to(m)
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:1000;
                background:#0a0f14;padding:16px 20px;border-radius:4px;
                border:1px solid #1d2a38;border-top:2px solid #0d5fa6;
                font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#7a9ab0;box-shadow:0 8px 24px rgba(0,0,0,0.6);">
      <div style="color:#c8d4e0;font-weight:600;letter-spacing:0.1em;
                  text-transform:uppercase;font-size:10px;margin-bottom:10px;">
        RESILIA · Risk Output
      </div>
      <div style="margin-bottom:4px;">
        <span style="color:#3f5a72;text-transform:uppercase;letter-spacing:0.1em;font-size:9px;">
          High-risk nodes
        </span><br>
        <span style="color:#e5534b;font-weight:600;font-size:14px;">{len(vulnerable):,}</span>
      </div>
      <div style="border-top:1px solid #131c26;margin:8px 0;"></div>
      RF F1 &nbsp;{f1:.4f}<br>
      Weather &nbsp;{weather}<br>
      SFP &nbsp;<b style="color:{TIER_COLOR[tier]};">{sfp:.2f}% · {tier}</b>
    </div>"""))
    return m


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 28px;">
      <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                  color:#c8d4e0;letter-spacing:-0.01em;line-height:1;">
        RESILIA
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#2d4a5e;margin-top:5px;text-transform:uppercase;
                  letter-spacing:0.16em;">
        Urban Flood Risk Engine v1.0
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    selected_area = st.selectbox("Study Area", list(AREA_CONFIGS.keys()), index=0)
    view_mode     = st.radio("Map View", ["Interactive (Folium)", "Static (Matplotlib)"], index=0)

    st.markdown('<div class="section-header">Actions</div>', unsafe_allow_html=True)
    run_btn = st.button("Run Analysis", use_container_width=True, type="primary")
    st.caption("Network cached 1 hr  ·  Weather refreshes 15 min")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#2d4a5e;line-height:2.0;">
      <span style="color:#3f5a72;text-transform:uppercase;letter-spacing:0.12em;
                   font-size:9px;">Data Sources</span><br>
      OSM via osmnx (ODbL)<br>
      BMKG Public API<br>
      DEM: simulated (PoC)<br><br>
      <span style="color:#3f5a72;text-transform:uppercase;letter-spacing:0.12em;
                   font-size:9px;">Model</span><br>
      Random Forest baseline<br>
      Phase 2: GraphSAGE / GAT
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 0;">
  <div style="font-family:'Syne',sans-serif;font-size:30px;font-weight:800;
              color:#c8d4e0;letter-spacing:-0.03em;line-height:1.1;">
    RESILIA
    <span style="font-weight:400;color:#2d6a9f;font-size:22px;
                 letter-spacing:-0.01em;"> — Urban Infrastructure Risk Engine</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
              color:#2d4a5e;margin-top:8px;letter-spacing:0.04em;">
    Real-time flood vulnerability assessment &nbsp;·&nbsp;
    OpenStreetMap + Random Forest + BMKG telemetry
  </div>
</div>
<hr style="border:none;border-top:1px solid #131c26;margin:20px 0 28px;">
""", unsafe_allow_html=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    with st.status("Running RESILIA pipeline...", expanded=True) as status:
        try:
            st.write("Fetching road network from OpenStreetMap...")
            G, nodes, edges = fetch_network(selected_area)
            st.write(f"   {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges loaded")

            st.write("Injecting terrain elevation model...")
            G = inject_elevation(G, selected_area)
            st.write("   Elevation scores assigned to all nodes")

            st.write("Training Random Forest classifier...")
            G, df, rf, feat_cols, f1, cv_scores, acc = build_ml_model(G)
            st.write(f"   F1={f1:.4f} · CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            st.write("Fetching BMKG live weather data...")
            adm4 = AREA_CONFIGS[selected_area]["adm4"]
            weather, stressor_w, live = fetch_bmkg(adm4)
            st.write(f"   {weather} · stressor weight={stressor_w} ({'live' if live else 'fallback'})")

            st.write("Computing Systemic Failure Probability...")
            vulnerable, n_total, sfp, tier = compute_risk(G, stressor_w)
            st.write(f"   SFP={sfp:.2f}% · Risk tier: {tier}")

            st.session_state.results = dict(
                G=G, nodes=nodes, edges=edges, df=df,
                rf=rf, feat_cols=feat_cols,
                f1=f1, acc=acc, cv_scores=cv_scores,
                weather=weather, stressor_w=stressor_w, live=live,
                vulnerable=vulnerable, n_total=n_total,
                sfp=sfp, tier=tier, area=selected_area,
            )
            status.update(label="Pipeline complete", state="complete")
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.error(str(e))


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    r = st.session_state.results

    # KPI row
    st.markdown('<div class="section-header">Key Risk Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Study Area",    r["area"])
    c2.metric("Total Nodes",   f"{r['n_total']:,}")
    c3.metric("High-Risk (RF)", f"{len(r['vulnerable']):,}",
              delta=f"{len(r['vulnerable'])/r['n_total']*100:.1f}% exposed",
              delta_color="inverse")
    c4.metric("RF F1-Score",   f"{r['f1']:.4f}")
    c5.metric("Weather",       r["weather"])
    c6.metric("SFP",           f"{r['sfp']:.2f}%")

    # Risk tier badge row
    st.markdown("<div style='margin-top:16px;margin-bottom:4px;'>", unsafe_allow_html=True)
    _, badge_col, _ = st.columns([1, 2, 4])
    with badge_col:
        st.markdown(
            f'<span class="badge {BADGE_CLASS[r["tier"]]}">'
            f'Risk Tier: {r["tier"]}</span>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if not r["live"]:
        st.warning("BMKG API unavailable — stressor weight defaulted to 0.85 (Heavy Rain fallback).")

    # Map
    st.markdown('<div class="section-header">Spatial Vulnerability Map</div>',
                unsafe_allow_html=True)

    if view_mode == "Interactive (Folium)":
        fmap = build_folium_map(
            r["G"], r["edges"], r["vulnerable"],
            r["area"], r["weather"], r["sfp"], r["tier"], r["f1"]
        )
        st_folium(fmap, width="100%", height=560, returned_objects=[])
    else:
        nodes_gdf = gpd.GeoDataFrame(
            {"v": [r["G"].nodes[n].get("vulnerability") for n in r["G"].nodes()]},
            geometry=r["nodes"].geometry, crs=r["nodes"].crs
        )
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#070b0f")
        ax.set_facecolor("#070b0f")
        r["edges"].plot(ax=ax, color="#131c26", linewidth=0.8, alpha=0.9, aspect=None)
        nodes_gdf[nodes_gdf["v"]=="Low"].plot(
            ax=ax, color="#0d5fa6", markersize=3, alpha=0.5, aspect=None)
        nodes_gdf[nodes_gdf["v"]=="High"].plot(
            ax=ax, color="#e5534b", markersize=9, alpha=0.9, aspect=None)
        ax.set_title(
            f"Flood Vulnerability — {r['weather']}  |  SFP {r['sfp']:.2f}%  [{r['tier']}]",
            color="#7a9ab0", fontsize=10, fontfamily="monospace", pad=14
        )
        ax.tick_params(colors="#2d4a5e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#131c26")
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor="#e5534b", label=f"High Risk ({len(r['vulnerable']):,})"),
                Patch(facecolor="#0d5fa6", label="Low Risk")
            ],
            facecolor="#0a0f14", edgecolor="#1d2a38",
            labelcolor="#7a9ab0", fontsize=9
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Model metrics expander
    with st.expander("Model Evaluation Details"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",       f"{r['acc']:.4f}")
        m2.metric("F1 (weighted)",  f"{r['f1']:.4f}")
        m3.metric("CV F1 (5-fold)", f"{r['cv_scores'].mean():.4f} ± {r['cv_scores'].std():.4f}")

        fig2, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#0b1018")
        for ax in axes:
            ax.set_facecolor("#0b1018")
            for spine in ax.spines.values():
                spine.set_edgecolor("#131c26")

        imp_df = pd.DataFrame({
            "feature"   : r["feat_cols"],
            "importance": r["rf"].feature_importances_
        }).sort_values("importance")

        axes[0].barh(imp_df["feature"], imp_df["importance"],
                     color="#0d5fa6", edgecolor="#070b0f", height=0.5)
        axes[0].set_title("Feature Importance", color="#7a9ab0",
                           fontsize=10, fontfamily="monospace", pad=10)
        axes[0].tick_params(colors="#3f5a72", labelsize=8)

        axes[1].bar(range(1, 6), r["cv_scores"], color="#0d5fa6",
                    edgecolor="#070b0f", width=0.55)
        axes[1].axhline(r["cv_scores"].mean(), color="#e5534b", linewidth=1.4,
                        linestyle="--", label=f"Mean {r['cv_scores'].mean():.4f}")
        axes[1].set_title("5-Fold CV F1", color="#7a9ab0",
                          fontsize=10, fontfamily="monospace", pad=10)
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(facecolor="#070b0f", edgecolor="#131c26",
                       labelcolor="#7a9ab0", fontsize=9)
        axes[1].tick_params(colors="#3f5a72", labelsize=8)
        axes[1].set_xticks(range(1, 6))
        axes[1].set_xticklabels([f"Fold {i}" for i in range(1, 6)])

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # Policy Recommendation
    st.markdown('<div class="section-header">Policy Recommendation</div>',
                unsafe_allow_html=True)
    c = TIER_COLOR[r["tier"]]
    bg = TIER_BG[r["tier"]]
    border = TIER_BORDER[r["tier"]]

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-left:3px solid {c};
                border-radius:4px;padding:20px 24px;
                font-family:'JetBrains Mono',monospace;">
      <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:14px;">
        <span style="color:{c};font-weight:700;font-size:11px;
                     text-transform:uppercase;letter-spacing:0.1em;">
          {r['tier']}
        </span>
        <span style="color:#3f5a72;font-size:10px;">
          SFP {r['sfp']:.2f}% &nbsp;·&nbsp; {r['weather']} &nbsp;·&nbsp;
          stressor={r['stressor_w']:.2f}
        </span>
      </div>
      <div style="color:#a0b8cc;font-size:12px;line-height:1.8;font-family:'DM Sans',sans-serif;">
        Prioritize emergency drainage remediation across
        <b style="color:#c8d4e0;">{len(r['vulnerable']):,} RF-identified flood-prone nodes</b>
        in {r['area']}.
        Reroute logistics corridors away from southern low-elevation clusters.
        Deploy real-time sensor network to monitor inundation onset in critical segments.
      </div>
    </div>""", unsafe_allow_html=True)

else:
    # Empty state — no emoji, editorial layout
    st.markdown("""
    <div style="padding: 100px 0 80px;text-align:center;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  text-transform:uppercase;letter-spacing:0.22em;color:#1d3a52;
                  margin-bottom:20px;">
        Awaiting Input
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:700;
                  color:#1d2a38;line-height:1.2;max-width:400px;margin:0 auto 16px;">
        Select a study area and run the analysis pipeline.
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                  color:#1d3a52;letter-spacing:0.04em;">
        OSM &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; BMKG telemetry
      </div>
    </div>""", unsafe_allow_html=True)