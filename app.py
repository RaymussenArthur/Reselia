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
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  div[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
  }
  div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px !important;
  }
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #58a6ff;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin: 24px 0 16px 0;
  }
  .badge { display:inline-block; font-family:'IBM Plex Mono',monospace;
    font-size:12px; font-weight:600; padding:4px 12px; border-radius:4px; }
  .badge-low      { background:#1f4d2e; color:#3fb950; border:1px solid #3fb950; }
  .badge-moderate { background:#3a2f1e; color:#d29922; border:1px solid #d29922; }
  .badge-high     { background:#4a1f1f; color:#f85149; border:1px solid #f85149; }
  .badge-critical { background:#6e1a1a; color:#ff6e6e; border:1px solid #ff6e6e; }
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
    "LOW": "#3fb950", "MODERATE": "#d29922",
    "HIGH": "#f85149", "CRITICAL": "#ff6e6e"
}

TIER_BG = {
    "LOW": "#1f4d2e", "MODERATE": "#3a2f1e",
    "HIGH": "#4a1f1f", "CRITICAL": "#6e1a1a"
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
    m   = folium.Map(location=list(cfg["center"]), zoom_start=14, tiles="CartoDB positron")
    for _, row in edges.iterrows():
        folium.PolyLine(
            [(lat, lon) for lon, lat in row.geometry.coords],
            color="#6c757d", weight=1.2, opacity=0.5
        ).add_to(m)
    for nid in vulnerable:
        d = G.nodes[nid]
        folium.CircleMarker(
            location=(d["y"], d["x"]), radius=5,
            color="#f85149", fill=True, fill_color="#f85149", fill_opacity=0.8,
            tooltip=f"Node {nid} | Elev: {d['elevation']} m | HIGH RISK"
        ).add_to(m)
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:1000;
                background:#161b22;padding:14px 18px;border-radius:8px;
                border:1px solid #30363d;font-family:monospace;font-size:12px;
                color:#e6edf3;box-shadow:0 4px 12px rgba(0,0,0,0.4);">
      <b>RESILIA Risk Output</b><br><br>
      <span style="color:#f85149;">&#9679;</span> High-risk nodes: <b>{len(vulnerable):,}</b><br>
      RF Model · F1={f1:.4f}<br>
      Weather: {weather}<br>
      SFP: <b>{sfp:.2f}%</b>
      <span style="color:{TIER_COLOR[tier]};font-weight:bold;"> {tier}</span>
    </div>"""))
    return m


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:24px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                  font-weight:600;color:#58a6ff;">🌊 RESILIA</div>
      <div style="font-size:11px;color:#8b949e;margin-top:2px;">
        Urban Flood Risk Engine v1.0
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    selected_area = st.selectbox("Study Area", list(AREA_CONFIGS.keys()), index=0)
    view_mode     = st.radio("Map View", ["Interactive (Folium)", "Static (Matplotlib)"], index=0)

    st.markdown('<div class="section-header">Actions</div>', unsafe_allow_html=True)
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")
    st.caption("Network cached 1 hr · Weather refreshes 15 min")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#6e7681;line-height:1.8;">
      <b style="color:#8b949e;">Data Sources</b><br>
      OSM via osmnx (ODbL)<br>
      BMKG Public API<br>
      DEM: simulated (PoC)<br><br>
      <b style="color:#8b949e;">Model</b><br>
      Random Forest baseline<br>
      Phase 2: GraphSAGE/GAT
    </div>""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:26px;
            font-weight:600;color:#e6edf3;letter-spacing:-0.02em;">
  RESILIA — Urban Infrastructure Risk Engine
</div>
<div style="font-size:13px;color:#8b949e;margin-top:4px;margin-bottom:20px;">
  Real-time flood vulnerability assessment · OpenStreetMap + Random Forest + BMKG telemetry
</div>
<hr style="border:none;border-top:1px solid #21262d;margin-bottom:24px;">
""", unsafe_allow_html=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    with st.status("Running RESILIA pipeline...", expanded=True) as status:
        try:
            st.write("📡 Fetching road network from OpenStreetMap...")
            G, nodes, edges = fetch_network(selected_area)
            st.write(f"   ✓ {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")

            st.write("🏔️  Injecting terrain elevation...")
            G = inject_elevation(G, selected_area)
            st.write("   ✓ Elevation scores assigned")

            st.write("🤖  Training Random Forest classifier...")
            G, df, rf, feat_cols, f1, cv_scores, acc = build_ml_model(G)
            st.write(f"   ✓ F1={f1:.4f} · CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            st.write("🌦️  Fetching BMKG live weather...")
            adm4 = AREA_CONFIGS[selected_area]["adm4"]
            weather, stressor_w, live = fetch_bmkg(adm4)
            st.write(f"   ✓ {weather} · weight={stressor_w} ({'live' if live else 'fallback'})")

            st.write("⚙️  Computing Systemic Failure Probability...")
            vulnerable, n_total, sfp, tier = compute_risk(G, stressor_w)
            st.write(f"   ✓ SFP={sfp:.2f}% [{tier}]")

            st.session_state.results = dict(
                G=G, nodes=nodes, edges=edges, df=df,
                rf=rf, feat_cols=feat_cols,
                f1=f1, acc=acc, cv_scores=cv_scores,
                weather=weather, stressor_w=stressor_w, live=live,
                vulnerable=vulnerable, n_total=n_total,
                sfp=sfp, tier=tier, area=selected_area,
            )
            status.update(label="Pipeline complete ✓", state="complete")
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
    c3.metric("High-Risk (RF)",f"{len(r['vulnerable']):,}",
              delta=f"{len(r['vulnerable'])/r['n_total']*100:.1f}% exposed",
              delta_color="inverse")
    c4.metric("RF F1-Score",   f"{r['f1']:.4f}")
    c5.metric("Weather",       r["weather"])
    c6.metric("SFP",           f"{r['sfp']:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    _, badge_col, _ = st.columns([1, 2, 4])
    with badge_col:
        st.markdown(
            f'<span class="badge {BADGE_CLASS[r["tier"]]}">'
            f'Risk Tier: {r["tier"]}</span>',
            unsafe_allow_html=True
        )

    if not r["live"]:
        st.warning("⚠️ BMKG API unavailable — stressor weight defaulted to 0.85 (Heavy Rain fallback).")

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
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        r["edges"].plot(ax=ax, color="#2d333b", linewidth=0.7, alpha=0.9, aspect=None)
        nodes_gdf[nodes_gdf["v"]=="Low"].plot(
            ax=ax, color="#3d8a6e", markersize=3, alpha=0.6, aspect=None)
        nodes_gdf[nodes_gdf["v"]=="High"].plot(
            ax=ax, color="#f85149", markersize=9, alpha=0.9, aspect=None)
        ax.set_title(
            f"Flood Vulnerability — {r['weather']} | SFP {r['sfp']:.2f}% [{r['tier']}]",
            color="#e6edf3", fontsize=11, fontweight="bold"
        )
        ax.tick_params(colors="#8b949e")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#f85149", label=f"High Risk ({len(r['vulnerable']):,})"),
                           Patch(facecolor="#3d8a6e", label="Low Risk")],
                  facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Model metrics expander
    with st.expander("🤖 Model Evaluation Details"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",       f"{r['acc']:.4f}")
        m2.metric("F1 (weighted)",  f"{r['f1']:.4f}")
        m3.metric("CV F1 (5-fold)", f"{r['cv_scores'].mean():.4f} ± {r['cv_scores'].std():.4f}")

        fig2, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#161b22")
        for ax in axes: ax.set_facecolor("#161b22")

        imp_df = pd.DataFrame({"feature": r["feat_cols"],
                               "importance": r["rf"].feature_importances_})\
                   .sort_values("importance")
        axes[0].barh(imp_df["feature"], imp_df["importance"],
                     color="#457b9d", edgecolor="#0d1117")
        axes[0].set_title("Feature Importance", color="#e6edf3", fontweight="bold")
        axes[0].tick_params(colors="#8b949e")

        axes[1].bar(range(1, 6), r["cv_scores"], color="#2a9d8f", edgecolor="#0d1117")
        axes[1].axhline(r["cv_scores"].mean(), color="#f85149", linewidth=1.8,
                        linestyle="--", label=f"Mean {r['cv_scores'].mean():.4f}")
        axes[1].set_title("5-Fold CV F1", color="#e6edf3", fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(facecolor="#0d1117", labelcolor="#e6edf3", fontsize=9)
        axes[1].tick_params(colors="#8b949e")

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # Recommendation
    st.markdown('<div class="section-header">Policy Recommendation</div>',
                unsafe_allow_html=True)
    c = TIER_COLOR[r["tier"]]; bg = TIER_BG[r["tier"]]
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {c};border-radius:8px;
                padding:16px 20px;font-family:'IBM Plex Mono',monospace;font-size:13px;">
      <span style="color:{c};font-weight:600;">[{r['tier']}] SFP {r['sfp']:.2f}%</span>
      &nbsp;·&nbsp;
      <span style="color:#e6edf3;">{r['weather']} · stressor={r['stressor_w']:.2f}</span>
      <br><br>
      <span style="color:#e6edf3;">
        Prioritize emergency drainage to <b>{len(r['vulnerable']):,} RF-identified
        flood-prone nodes</b> in {r['area']}.
        Reroute logistics corridors away from southern low-elevation clusters.
      </span>
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#6e7681;">
      <div style="font-size:48px;margin-bottom:16px;">🌊</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;">
        Select a study area and click
        <b style="color:#58a6ff;">Run Analysis</b> to begin.
      </div>
    </div>""", unsafe_allow_html=True)
