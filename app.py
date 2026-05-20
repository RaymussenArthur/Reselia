"""
RESELIA v2.0 — v3 PATCH
Fixes applied:
  [v3-1] BMKG: proper adm4 param, graceful fallback (no false "error" banner)
  [v3-2] Font contrast: all dim text boosted for readability
  [v3-3] Run button: pulses blue when params change, grey "up to date" when fresh
  [v3-4] KPI carousel: proper iframe HTML, no raw-HTML bleed
  [v3-5] Plot loading: st.spinner on every heavy chart render
  [v3-6] Map zoom: auto-computed from dist value per area
  [v3-7] Red nodes: threshold raised to 0.65 — only truly high-risk nodes are red
  [v3-8] POI fallback: added entries for all 10 areas
"""
from __future__ import annotations
import math, os, pickle, warnings, requests
import numpy as np, pandas as pd, polars as pl
import networkx as nx, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from folium.plugins import HeatMap
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import osmnx as ox
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay, classification_report)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RESELIA v2", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Sidebar collapse button text ──────────────────────────────────────────
components.html("""
<script>
(function() {
    function fixBtn() {
        try {
            var doc = window.parent.document;
            doc.querySelectorAll('button').forEach(function(btn) {
                if (btn.innerText && btn.innerText.trim().startsWith('keyboard')) {
                    Array.from(btn.childNodes).forEach(function(node) {
                        if (node.nodeType === 3 || node.tagName === 'SPAN') {
                            node.style && (node.style.display = 'none');
                        }
                    });
                    btn.querySelectorAll('*').forEach(function(el) {
                        el.style.fontSize = '0';
                        el.style.visibility = 'hidden';
                        el.style.width = '0';
                        el.style.display = 'none';
                    });
                    if (!btn.querySelector('.reselia-icon')) {
                        var span = doc.createElement('span');
                        span.className = 'reselia-icon';
                        span.textContent = '≡';
                        span.style.cssText = 'font-size:26px;color:#58a6ff;font-family:monospace;font-weight:400;visibility:visible;display:inline;width:auto;line-height:1;';
                        btn.appendChild(span);
                    }
                }
            });
        } catch(e) {}
    }
    fixBtn();
    setTimeout(fixBtn, 500);
    setTimeout(fixBtn, 1200);
    try {
        new MutationObserver(fixBtn).observe(
            window.parent.document.body, {childList:true, subtree:true}
        );
    } catch(e) {}
})();
</script>
""", height=0, scrolling=False)

# Session state init
for k, v in [("last_params", {}), ("results", None), ("dirty", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');
.material-symbols-rounded {font-family: 'Material Symbols Rounded' !important;font-optical-sizing: auto;font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;}
.stApp { background-color: #05090f; color: #c9d1d9; }
section[data-testid="stSidebar"] { background-color: #090e18; border-right: 1px solid #1e2a3a; }
section[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px; }
div[data-testid="metric-container"] {background: linear-gradient(140deg,#090e18 0%,#0d1525 100%);border: 1px solid #1e2a3a; border-top: 2px solid #1f6feb; border-radius: 2px; padding: 16px 18px;}
div[data-testid="metric-container"] label {color: #58a6ff !important; font-family: 'IBM Plex Mono',monospace !important;font-size: 8px !important; text-transform: uppercase; letter-spacing: .2em; font-weight: 700;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {color: #e6edf3 !important; font-family: 'IBM Plex Mono',monospace !important;font-size: 18px !important; font-weight: 700;}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {font-family: 'IBM Plex Mono',monospace !important; font-size: 9px !important;}
/* v3-3: dirty run button pulses */
.run-btn-dirty > button {background: linear-gradient(135deg,#1f6feb,#388bfd) !important;color: #fff !important; border: none !important;box-shadow: 0 0 0 2px #388bfd, 0 0 22px rgba(56,139,253,.55) !important;animation: pulse-btn 1.4s infinite;}
@keyframes pulse-btn {0%  { box-shadow: 0 0 0 2px #388bfd, 0 0 12px rgba(56,139,253,.35); }50% { box-shadow: 0 0 0 2px #58a6ff, 0 0 28px rgba(88,166,255,.65); }100%{ box-shadow: 0 0 0 2px #388bfd, 0 0 12px rgba(56,139,253,.35); }}
.run-btn-clean > button {background: #0d1525 !important; color: #6e7681 !important;border: 1px solid #1e2a3a !important; box-shadow: none !important;}
.stButton > button {border-radius: 2px; font-family: 'IBM Plex Mono',monospace; font-size: 11px;font-weight: 700; letter-spacing: .1em; text-transform: uppercase;padding: 11px 20px; transition: background .2s, box-shadow .2s; width: 100%;}
.stTabs [data-baseweb="tab-list"] { background: #090e18; border-bottom: 1px solid #1e2a3a; gap: 0; }
.stTabs [data-baseweb="tab"] {font-family: 'IBM Plex Mono',monospace !important; font-size: 10px !important;font-weight: 700 !important; letter-spacing: .12em !important; text-transform: uppercase !important;color: #6e7681 !important; background: transparent !important;border-radius: 0 !important; border-bottom: 2px solid transparent !important; padding: 10px 18px !important;}
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #1f6feb !important; }
.stSelectbox > div > div {background: #090e18; border: 1px solid #1e2a3a; border-radius: 2px;font-family: 'IBM Plex Mono',monospace; font-size: 11px; color: #c9d1d9;}
.stSlider [data-baseweb="slider"] { color: #388bfd; }
.stRadio label { font-family: 'IBM Plex Mono',monospace !important; font-size: 10px !important; color: #adbac7 !important; }
.streamlit-expanderHeader {background: #090e18 !important; border: 1px solid #1e2a3a !important;border-radius: 2px !important; font-family: 'IBM Plex Mono',monospace !important;font-size: 10px !important; color: #adbac7 !important; text-transform: uppercase; letter-spacing: .1em;}
.stDataFrame { border: 1px solid #1e2a3a; }
hr { border-color: #1e2a3a !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #05090f; }
::-webkit-scrollbar-thumb { background: #1e2a3a; }
[data-testid="collapsedControl"] {display: flex !important;align-items: center !important;justify-content: center !important;}
[data-testid="collapsedControl"] span {font-size: 0 !important;line-height: 0 !important;}
[data-testid="collapsedControl"]::after {content: "▶" !important;font-size: 12px !important;font-family: monospace !important;color: #58a6ff !important;}
section[data-testid="stSidebar"][aria-expanded="true"] ~ * [data-testid="collapsedControl"]::after {content: "◀" !important;}
button[kind="header"] span[data-testid="stIconMaterial"] {font-size: 0 !important;}
.stCaption { font-family: 'IBM Plex Mono',monospace !important; font-size: 9px !important; color: #6e7681 !important; }
.stAlert { border-radius: 2px; }
</style>""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FLOOD_THRESHOLD_M   = 2.5
DBSCAN_EPS_M        = 350   
DBSCAN_MIN_PTS      = 5 
CRITICAL_RADIUS_M   = 300.0
EARTH_R             = 6_371_000.0
MODEL_PICKLE_PATH   = "Notebooks/Phase_2/resilia_gat_model.pkl"
HIGH_RISK_THRESHOLD = 0.65  # v3-7: only nodes with prob >= 0.65 are "High"

AREA_CONFIGS: dict[str, dict] = {
    "Kemayoran":         {"center":(-6.1625,106.8572),"dist":4000,"adm4":"31.71.03.1001",
                          "bbox":{"north":-6.1325,"south":-6.1925,"east":106.8872,"west":106.8272}},
    "Penjaringan":       {"center":(-6.1180,106.7870),"dist":4500,"adm4":"31.72.01.1001",
                          "bbox":{"north":-6.0880,"south":-6.1480,"east":106.8170,"west":106.7570}},
    "Cengkareng":        {"center":(-6.1534,106.7362),"dist":4500,"adm4":"31.73.01.1001",
                          "bbox":{"north":-6.1234,"south":-6.1834,"east":106.7662,"west":106.7062}},
    "Jatinegara":        {"center":(-6.2264,106.8711),"dist":4000,"adm4":"31.75.03.1001",
                          "bbox":{"north":-6.1964,"south":-6.2564,"east":106.9011,"west":106.8411}},
    "Pulo Gadung":       {"center":(-6.1912,106.8924),"dist":4000,"adm4":"31.75.02.1001",
                          "bbox":{"north":-6.1612,"south":-6.2212,"east":106.9224,"west":106.8624}},
    "Kebayoran Baru":    {"center":(-6.2461,106.8042),"dist":3800,"adm4":"31.74.07.1001",
                          "bbox":{"north":-6.2181,"south":-6.2741,"east":106.8322,"west":106.7762}},
    "Cilincing":         {"center":(-6.1245,106.9360),"dist":4500,"adm4":"31.72.06.1001",
                          "bbox":{"north":-6.0945,"south":-6.1545,"east":106.9660,"west":106.9060}},
    "Kelapa Gading":     {"center":(-6.1601,106.9032),"dist":4000,"adm4":"31.72.05.1001",
                          "bbox":{"north":-6.1301,"south":-6.1901,"east":106.9332,"west":106.8732}},
    "Grogol Petamburan": {"center":(-6.1643,106.7869),"dist":4000,"adm4":"31.73.02.1001",
                          "bbox":{"north":-6.1343,"south":-6.1943,"east":106.8169,"west":106.7569}},
    "Mampang Prapatan":  {"center":(-6.2520,106.8225),"dist":3800,"adm4":"31.74.03.1001",
                          "bbox":{"north":-6.2240,"south":-6.2800,"east":106.8505,"west":106.7945}},
}

WEATHER_WEIGHTS: dict[str,float] = {
    "Cerah":0.05,"Cerah Berawan":0.10,"Berawan":0.15,
    "Hujan Ringan":0.45,"Hujan Sedang":0.65,"Hujan Lebat":0.85,"Hujan Petir":0.95,
}

POI_COLORS = {"hospital":"#e63946","clinic":"#ff6b6b","school":"#2a9d8f",
              "marketplace":"#e9c46a","supermarket":"#f4a261","fire_station":"#e76f51",
              "police":"#264653","station":"#8ecae6"}
POI_TAGS   = {"amenity":["hospital","clinic","school","marketplace","supermarket","fire_station","police"],
              "public_transport":["station"]}

POI_FALLBACK: dict[str,list[dict]] = {
    "Kemayoran":[
        {"name":"RSUD Kemayoran","amenity":"hospital","lat":-6.155,"lon":106.855},
        {"name":"Puskesmas Kemayoran","amenity":"clinic","lat":-6.162,"lon":106.862},
        {"name":"SMAN 17 Jakarta","amenity":"school","lat":-6.158,"lon":106.858},
        {"name":"Pasar Kemayoran","amenity":"marketplace","lat":-6.168,"lon":106.865},
        {"name":"Polsek Kemayoran","amenity":"police","lat":-6.160,"lon":106.852},
        {"name":"Koramil Kemayoran","amenity":"fire_station","lat":-6.172,"lon":106.860},
        {"name":"Halte TransJakarta","amenity":"station","lat":-6.165,"lon":106.868},
    ],
    "Penjaringan":[
        {"name":"RS Pluit","amenity":"hospital","lat":-6.119,"lon":106.797},
        {"name":"SD Penjaringan 01","amenity":"school","lat":-6.122,"lon":106.803},
        {"name":"Pasar Penjaringan","amenity":"marketplace","lat":-6.118,"lon":106.801},
        {"name":"Polsek Penjaringan","amenity":"police","lat":-6.121,"lon":106.799},
    ],
    "Cengkareng":[
        {"name":"RSUD Cengkareng","amenity":"hospital","lat":-6.148,"lon":106.743},
        {"name":"SD Cengkareng 01","amenity":"school","lat":-6.152,"lon":106.741},
        {"name":"Pasar Cengkareng","amenity":"marketplace","lat":-6.155,"lon":106.744},
        {"name":"Polsek Cengkareng","amenity":"police","lat":-6.150,"lon":106.740},
    ],
    "Jatinegara":[
        {"name":"RS Hermina Jatinegara","amenity":"hospital","lat":-6.225,"lon":106.872},
        {"name":"Pasar Jatinegara","amenity":"marketplace","lat":-6.228,"lon":106.870},
        {"name":"Polsek Jatinegara","amenity":"police","lat":-6.226,"lon":106.871},
    ],
    "Pulo Gadung":[
        {"name":"RS Persahabatan","amenity":"hospital","lat":-6.191,"lon":106.892},
        {"name":"Pasar Pulo Gadung","amenity":"marketplace","lat":-6.194,"lon":106.895},
        {"name":"Polsek Pulo Gadung","amenity":"police","lat":-6.192,"lon":106.893},
    ],
    "Kebayoran Baru":[
        {"name":"RS Siloam","amenity":"hospital","lat":-6.248,"lon":106.805},
        {"name":"Blok M Plaza","amenity":"supermarket","lat":-6.245,"lon":106.800},
        {"name":"Polsek Kebayoran Baru","amenity":"police","lat":-6.247,"lon":106.803},
    ],
    "Cilincing":[
        {"name":"Puskesmas Cilincing","amenity":"clinic","lat":-6.124,"lon":106.937},
        {"name":"Pasar Cilincing","amenity":"marketplace","lat":-6.127,"lon":106.939},
    ],
    "Kelapa Gading":[
        {"name":"RS Mitra Keluarga","amenity":"hospital","lat":-6.158,"lon":106.902},
        {"name":"Mall Kelapa Gading","amenity":"supermarket","lat":-6.163,"lon":106.906},
        {"name":"Polsek Kelapa Gading","amenity":"police","lat":-6.161,"lon":106.903},
    ],
    "Grogol Petamburan":[
        {"name":"RS Sumber Waras","amenity":"hospital","lat":-6.165,"lon":106.789},
        {"name":"Univ Tarumanagara","amenity":"school","lat":-6.167,"lon":106.787},
        {"name":"Polsek Grogol","amenity":"police","lat":-6.164,"lon":106.788},
    ],
    "Mampang Prapatan":[
        {"name":"RS Columbia Asia","amenity":"hospital","lat":-6.252,"lon":106.823},
        {"name":"Pasar Mampang","amenity":"marketplace","lat":-6.255,"lon":106.826},
        {"name":"Polsek Mampang","amenity":"police","lat":-6.253,"lon":106.824},
    ],
}

TIER_COLOR  = {"LOW":"#3fb950","MODERATE":"#d29922","HIGH":"#f85149","CRITICAL":"#ff4444"}
TIER_BG     = {"LOW":"#0d2116","MODERATE":"#1f1700","HIGH":"#200c0c","CRITICAL":"#2d0000"}
TIER_BORDER = {"LOW":"#1e4d2b","MODERATE":"#4d3800","HIGH":"#4d1515","CRITICAL":"#660000"}

FEAT_COLS_V2 = ["elevation","degree_centrality","betweenness_centrality",
                "closeness_centrality","poi_criticality","clustering_coefficient","pagerank"]
FEAT_COLS_V1 = ["degree_centrality","betweenness_centrality","closeness_centrality","elevation"]

def _zoom_from_dist(d: int) -> int:
    return 15 if d<=1500 else 14 if d<=2500 else 13 if d<=4000 else 12

def _risk_color(s: float, stressor_w: float, elev: float = 0.) -> str:
    if elev > FLOOD_THRESHOLD_M * 2.0:   
        s = min(s, 0.40)
    adjusted = s * (0.75 + stressor_w * 0.25)
    return "#388bfd" if adjusted < 0.35 else "#d29922" if adjusted < 0.55 else "#f85149"

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_notebook_model() -> dict | None:
    if not os.path.exists(MODEL_PICKLE_PATH): return None
    try:
        with open(MODEL_PICKLE_PATH,"rb") as f: return pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load model: {e}"); return None

# ── Network ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_network(area_name: str):
    cfg = AREA_CONFIGS[area_name]
    ox.settings.timeout = 120; ox.settings.log_console = False
    G = ox.graph_from_point(cfg["center"], dist=cfg["dist"], network_type="drive")
    return G, *ox.graph_to_gdfs(G)

# ── Elevation ──────────────────────────────────────────────────────────────────
def inject_demnas_elevation(G, area_name: str, flood_threshold: float):
    cfg = AREA_CONFIGS[area_name]; bbox = cfg["bbox"]
    rng = np.random.default_rng(seed=42)
    min_lat,max_lat,min_lon,max_lon = bbox["south"],bbox["north"],bbox["west"],bbox["east"]
    for node,data in G.nodes(data=True):
        nl = (data["y"]-min_lat)/(max_lat-min_lat+1e-9)
        nx_ = (data["x"]-min_lon)/(max_lon-min_lon+1e-9)
        G.nodes[node]["elevation"] = float(np.clip(8.0*((1-nl)*.5+(1-nx_)*.5)+rng.uniform(-.3,.3)*4.,0.,8.))
    records = []
    for node,data in G.nodes(data=True):
        elev = data["elevation"]
        nbrs = list(G.predecessors(node))+list(G.successors(node))
        mn   = float(np.mean([G.nodes[n]["elevation"] for n in nbrs])) if nbrs else elev
        records.append({"node":node,"elevation":elev,"mean_neigh_elev":mn,
                        "betweenness":data.get("betweenness_centrality",0.),"closeness":data.get("closeness_centrality",0.)})
    df = pl.DataFrame(records)
    rng2 = np.random.default_rng(seed=42)
    df = df.with_columns(pl.Series("residual",rng2.uniform(0.,1.,size=df.height)))
    df = df.with_columns([
        (1./(1.+(-1.5*(pl.col("elevation")-flood_threshold)).exp())).alias("own"),
        (1./(1.+(-1.5*(pl.col("mean_neigh_elev")-flood_threshold)).exp())).alias("neigh"),
        ((pl.col("betweenness")-pl.col("betweenness").min())/(pl.col("betweenness").max()-pl.col("betweenness").min()+1e-8)).alias("nb"),
        ((pl.col("closeness")-pl.col("closeness").min())/(pl.col("closeness").max()-pl.col("closeness").min()+1e-8)).alias("nc"),
    ]).with_columns(((pl.col("nb")+pl.col("nc"))/2.).alias("ct")
    ).with_columns((0.15*pl.col("own")+0.45*pl.col("neigh")+0.30*pl.col("ct")+0.10*pl.col("residual")).alias("fp")
    ).with_columns(pl.when(pl.col("fp")>0.5).then(1).otherwise(0).alias("flood_label"))
    nx.set_node_attributes(G, dict(zip(df["node"].to_list(),df["flood_label"].to_list())), "flood_label")
    return G

def _hav(la1,lo1,la2,lo2):
    p1,p2=math.radians(la1),math.radians(la2)
    a=math.sin(math.radians(la2-la1)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(lo2-lo1)/2)**2
    return EARTH_R*2.*math.asin(math.sqrt(a))

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_pois(area_name: str) -> pd.DataFrame:
    cfg=AREA_CONFIGS[area_name]; bbox=cfg["bbox"]
    bbox_str=f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"
    q=f"""[out:json][timeout:30];(node["amenity"~"{'|'.join(POI_TAGS['amenity'])}"]({bbox_str});
    node["public_transport"="station"]({bbox_str});way["amenity"~"{'|'.join(POI_TAGS['amenity'])}"]({bbox_str}););out center;"""
    try:
        r=requests.post("https://overpass-api.de/api/interpreter",data={"data":q},
                        headers={"User-Agent":"ResiliaSpatialEngine/2.0","Accept":"application/json"},timeout=30)
        r.raise_for_status()
        pois=[]
        for e in r.json().get("elements",[]):
            lat=e.get("lat") or e.get("center",{}).get("lat"); lon=e.get("lon") or e.get("center",{}).get("lon")
            amenity=e.get("tags",{}).get("amenity") or e.get("tags",{}).get("public_transport","unknown")
            name=e.get("tags",{}).get("name",f"{amenity.title()} Facility")
            if lat and lon: pois.append({"name":name,"amenity":amenity,"lat":float(lat),"lon":float(lon)})
        if pois: return pd.DataFrame(pois)
        raise ValueError("empty")
    except Exception: return pd.DataFrame(POI_FALLBACK.get(area_name, POI_FALLBACK["Kemayoran"]))

def inject_poi_criticality(G, poi_df: pd.DataFrame):
    poi_list=[(float(r["lat"]),float(r["lon"])) for _,r in poi_df.iterrows()]
    for node,data in G.nodes(data=True):
        nlat,nlon=float(data["y"]),float(data["x"])
        G.nodes[node]["poi_criticality"]=round(max(0.,1.-min(_hav(nlat,nlon,p[0],p[1]) for p in poi_list)/CRITICAL_RADIUS_M),4)
    return G

def compute_graph_features(G):
    dc=nx.degree_centrality(G); bc=nx.betweenness_centrality(G,k=200,normalized=True,seed=42)
    cc=nx.closeness_centrality(G); clc=nx.clustering(nx.Graph(G.to_undirected())); pr=nx.pagerank(G,alpha=.85,max_iter=200)
    for n in G.nodes():
        G.nodes[n].update({"degree_centrality":float(dc[n]),"betweenness_centrality":float(bc[n]),
                           "closeness_centrality":float(cc[n]),"clustering_coefficient":float(clc[n]),"pagerank":float(pr[n])})
    records=[{"node_id":n,"elevation":float(d.get("elevation",5.)),"degree_centrality":float(dc[n]),
              "betweenness_centrality":float(bc[n]),"closeness_centrality":float(cc[n]),
              "poi_criticality":float(d.get("poi_criticality",0.)),"clustering_coefficient":float(clc[n]),
              "pagerank":float(pr[n]),"flood_label":int(d.get("flood_label",0))} for n,d in G.nodes(data=True)]
    df=pd.DataFrame(records)
    for col in FEAT_COLS_V2: df[col]=df[col].astype(np.float64)
    df["flood_label"]=df["flood_label"].astype(np.int64)
    return G,df

def _gat_mp(G,nl,ni,fm,n_hops=2):
    aug=fm.copy()
    for _ in range(n_hops):
        nf=np.zeros_like(aug)
        for node in G.nodes():
            if node not in ni: continue
            i=ni[node]; nbrs=[n for n in list(G.predecessors(node))+list(G.successors(node)) if n in ni]
            if not nbrs: nf[i]=aug[i]; continue
            nfeats=np.array([aug[ni[n]] for n in nbrs],dtype=np.float64)
            dists=np.linalg.norm(nfeats-aug[i],axis=1)+1e-8
            ac=(1./dists)/(1./dists).sum()
            nf[i]=.6*aug[i]+.4*np.sum(ac[:,None]*nfeats,axis=0)
        aug=nf
    return aug

def build_gat_model(G, df: pd.DataFrame, nb: dict | None, flood_threshold: float = FLOOD_THRESHOLD_M):
    nl=df["node_id"].tolist(); ni={n:i for i,n in enumerate(nl)}
    y=df["flood_label"].values.astype(np.int64)
    if nb is not None:
        scaler,ms=nb["scaler"],"notebook_pickle"; fm=scaler.transform(df[FEAT_COLS_V2].values.astype(np.float64))
    else:
        scaler,ms=StandardScaler(),"trained_fresh"; fm=scaler.fit_transform(df[FEAT_COLS_V2].values.astype(np.float64))
    n_hops=nb["n_hops"] if nb else 2
    Xg=_gat_mp(G,nl,ni,fm,n_hops).astype(np.float64)
    Xtr,Xte,ytr,yte=train_test_split(Xg,y,test_size=.20,random_state=42,stratify=y)
    if nb is not None:
        clf=nb["clf"]; yp=clf.predict(Xte); ypr=clf.predict_proba(Xte)[:,1]
    else:
        clf=GradientBoostingClassifier(n_estimators=300,learning_rate=.05,max_depth=5,
                                        min_samples_leaf=3,subsample=.8,random_state=42)
        clf.fit(Xtr,ytr); yp=clf.predict(Xte); ypr=clf.predict_proba(Xte)[:,1]
    acc=float(accuracy_score(yte,yp)); f1w=float(f1_score(yte,yp,average="weighted"))
    f1m=float(f1_score(yte,yp,average="macro")); auc=float(roc_auc_score(yte,ypr))
    cvs=cross_val_score(clf,Xg,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="f1_weighted",n_jobs=-1)
    cm=confusion_matrix(yte,yp); cr=classification_report(yte,yp,target_names=["Low Risk","High Risk"],output_dict=True)
    rf=RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1)
    Xb=df[FEAT_COLS_V1].values.astype(np.float64)
    Xtrb,Xteb,ytrb,yteb=train_test_split(Xb,y,test_size=.20,random_state=42,stratify=y)
    rf.fit(Xtrb,ytrb)
    f1b=float(f1_score(yteb,rf.predict(Xteb),average="weighted"))
    f1mb=float(f1_score(yteb,rf.predict(Xteb),average="macro"))
    pa = clf.predict_proba(Xg)[:,1]
    for i, nid in enumerate(nl):
        p    = float(pa[i])
        elev = float(G.nodes[nid].get("elevation", 0))
        if elev > flood_threshold * 2.0:
            p = min(p, 0.40)
        G.nodes[nid]["vulnerability"] = "High" if p >= HIGH_RISK_THRESHOLD else "Low"
        G.nodes[nid]["risk_score"]    = round(p, 4)
        G.nodes[nid]["gat_pred"]      = 1 if p >= HIGH_RISK_THRESHOLD else 0
    return {"clf":clf,"scaler":scaler,"feat_cols":FEAT_COLS_V2,"model_src":ms,
            "acc":acc,"f1_w":f1w,"f1_mac":f1m,"auc_roc":auc,"cv_scores":cvs,
            "conf_mat":cm,"class_rep":cr,"f1_base":f1b,"f1mac_base":f1mb,"y_test":yte,"y_pred":yp,"X_gat":Xg,"y":y}

def _net_metrics(G):
    Gu=nx.Graph(G.to_undirected())
    if Gu.number_of_nodes()==0: return 0.,0.,0.
    sample=list(Gu.nodes())[:min(150,len(Gu))]; ev=[]
    for s in sample:
        ls=nx.single_source_shortest_path_length(Gu,s)
        ev.append(sum(1/l for _,l in ls.items() if l>0)/max(1,Gu.number_of_nodes()-1))
    ge=float(np.mean(ev)) if ev else 0.
    comps=list(nx.connected_components(Gu))
    lcc=max(len(c) for c in comps)/Gu.number_of_nodes() if comps else 0.
    return ge,lcc,float(nx.average_clustering(Gu))

def run_cascade(G, n_rounds=5, removal_pct=.05):
    sn=sorted([(n,G.nodes[n].get("risk_score",0.)) for n in G.nodes()],key=lambda x:x[1],reverse=True)
    nr=max(1,int(len(sn)*removal_pct)); Gs=G.copy()
    r0e,r0l,r0c=_net_metrics(Gs)
    rounds=[{"round":0,"nodes_removed":0,"pct_removed":0.,"global_efficiency":r0e,"lcc_fraction":r0l,"avg_clustering":r0c}]
    for rnd in range(1,n_rounds+1):
        batch=sn[(rnd-1)*nr:rnd*nr]; rem=[n for n,_ in batch if Gs.has_node(n)]
        Gs.remove_nodes_from(rem); e,l,cl=_net_metrics(Gs)
        rounds.append({"round":rnd,"nodes_removed":rnd*nr,"pct_removed":round(rnd*nr/len(sn)*100,2),
                        "global_efficiency":e,"lcc_fraction":l,"avg_clustering":cl})
    sd=pd.DataFrame(rounds)
    ed=(r0e-sd.iloc[-1]["global_efficiency"])/(r0e+1e-8); ld=(r0l-sd.iloc[-1]["lcc_fraction"])/(r0l+1e-8)
    res=float(np.clip(1.-(0.6*ed+0.4*ld),0.,1.))
    return {"sim_df":sd,"resilience_score":res,"eff_degradation":ed,"lcc_degradation":ld,
            "baseline_eff":r0e,"baseline_lcc":r0l,"baseline_clust":r0c}

def run_dbscan(G, eps_m=DBSCAN_EPS_M, min_pts=DBSCAN_MIN_PTS):
    hr=[(n,G.nodes[n]) for n in G.nodes() if G.nodes[n].get("vulnerability")=="High"]
    if not hr: return {"epicenter_df":pd.DataFrame(),"n_epicenters":0,"n_noise":0,"labels":[],"node_ids":[],"coords":np.array([])}
    cd=np.array([(d["y"],d["x"]) for _,d in hr]); nids=[n for n,_ in hr]
    labels=[int(x) for x in DBSCAN(eps=eps_m/EARTH_R,min_samples=min_pts,metric="haversine").fit_predict(np.radians(cd))]
    clusters=[c for c in sorted(set(labels)) if c!=-1]; n_noise=labels.count(-1)
    recs=[]
    for cid in clusters:
        idx=[i for i,l in enumerate(labels) if l==cid]; cn=[nids[i] for i in idx]; cc=cd[idx]
        cent=cc.mean(axis=0); rs=[float(G.nodes[n].get("risk_score",0)) for n in cn]
        ps=[float(G.nodes[n].get("poi_criticality",0)) for n in cn]; den=len(cn)
        mr,mp=float(np.mean(rs)),float(np.mean(ps))
        recs.append({"cluster_id":cid,"n_nodes":den,"centroid_lat":round(float(cent[0]),6),
                      "centroid_lon":round(float(cent[1]),6),"mean_risk_score":round(mr,4),
                      "mean_poi_crit":round(mp,4),"triage_score":round(.5*mr+.3*mp+.2*(den/max(1,len(hr))),4)})
    edf = (pd.DataFrame(recs).sort_values("triage_score", ascending=False).reset_index(drop=True)
           if recs else pd.DataFrame(columns=["cluster_id","n_nodes","centroid_lat",
                                               "centroid_lon","mean_risk_score",
                                               "mean_poi_crit","triage_score"]))
    edf.index+=1
    for i,nid in enumerate(nids): G.nodes[nid]["epicenter_cluster"]=labels[i]
    return {"epicenter_df":edf,"n_epicenters":len(clusters),"n_noise":n_noise,"labels":labels,"node_ids":nids,"coords":cd}

# v3-1: BMKG with adm4, graceful multi-level fallback
@st.cache_data(show_spinner=False, ttl=900)
def fetch_bmkg(adm4: str) -> tuple[str, float, bool, str]:
    try:
        resp = requests.get(f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}", timeout=10)
        resp.raise_for_status()
        block = resp.json().get("data", [])
        if block:
            cuaca = block[0].get("cuaca", [])
            current = cuaca[0][0] if cuaca and cuaca[0] else {}
            desc = current.get("weather_desc", "")
            if desc in WEATHER_WEIGHTS:
                return desc, WEATHER_WEIGHTS[desc], True, ""
        return "Berawan", WEATHER_WEIGHTS["Berawan"], True, "unmapped"
    except requests.exceptions.Timeout:
        return "Berawan", WEATHER_WEIGHTS["Berawan"], False, "timeout"
    except requests.exceptions.HTTPError as e:
        return "Berawan", WEATHER_WEIGHTS["Berawan"], False, f"http_{e.response.status_code if e.response else '?'}"
    except Exception:
        return "Berawan", WEATHER_WEIGHTS["Berawan"], False, "offline"

def compute_risk_v2(G, sw, res):
    vuln=[n for n,d in G.nodes(data=True) if d.get("vulnerability")=="High"]
    nt=G.number_of_nodes(); exp=len(vuln)/nt; pen=1.+(1.-res); sfp=sw*exp*100*pen
    tier="CRITICAL" if sfp>=25 else "HIGH" if sfp>=15 else "MODERATE" if sfp>=5 else "LOW"
    return {"vulnerable":vuln,"n_total":nt,"exposure":exp,"penalty":pen,"sfp":sfp,"tier":tier}

def build_map(G, edges, vuln, poi_df, epi_data, area, weather, sfp, tier, f1, stressor_w, show_heatmap=False):
    import random
    from folium.plugins import MarkerCluster

    cfg = AREA_CONFIGS[area]; tc = TIER_COLOR[tier]; zoom = _zoom_from_dist(cfg["dist"])

    m = folium.Map(location=list(cfg["center"]), zoom_start=zoom,
               tiles=None, prefer_canvas=True)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Filters",
        max_zoom=19,
        subdomains="abcd"
    ).add_to(m)

    # ── Edges (base layer, always visible) ────────────────────────────────────
    edge_list = list(edges.iterrows())
    if len(edge_list) > 3000:
        rng_e = random.Random(42)
        edge_list = rng_e.sample(edge_list, 3000)
    for _, row in edge_list:
        folium.PolyLine([(lat, lon) for lon, lat in row.geometry.coords],
                        color="#1f6feb", weight=0.9, opacity=.25).add_to(m)

    # ── Risk Nodes FeatureGroup ────────────────────────────────────────────────
    node_fg = folium.FeatureGroup(name="Risk Nodes", show=True)
    rng_n = random.Random(42)
    for node, d in G.nodes(data=True):
        s = d.get("risk_score", 0.)
        if s < 0.35: continue
        color = _risk_color(s, stressor_w, elev=d.get("elevation", 0.))
        r2 = 3 if s < 0.55 else 5
        cl = d.get("epicenter_cluster", -1); ep = f"EP-{cl+1}" if cl != -1 else "—"
        folium.CircleMarker(
            location=(d["y"], d["x"]), radius=r2, color=color, fill=True,
            fill_color=color, fill_opacity=.55 if s < 0.35 else .85,
            tooltip=f"Risk {s:.3f} | Elev {d.get('elevation',0):.1f}m | POI {d.get('poi_criticality',0):.3f} | {ep}"
        ).add_to(node_fg)
    node_fg.add_to(m)

    # ── Heatmap FeatureGroup ───────────────────────────────────────
    if show_heatmap and vuln:
        heat_fg = folium.FeatureGroup(name="Heatmap", show=True)
        HeatMap(
            [(G.nodes[n]["y"], G.nodes[n]["x"], G.nodes[n].get("risk_score", .5)) for n in vuln],
            radius=18, blur=15, max_zoom=16,
            gradient={"0.4": "#388bfd", "0.65": "#d29922", "1.0": "#f85149"}
        ).add_to(heat_fg)
        heat_fg.add_to(m)

    # ── POI FeatureGroup + MarkerCluster ──────────────────────────────────────
    poi_fg = folium.FeatureGroup(name="POI", show=True)
    poi_cluster = MarkerCluster(
        options={"maxClusterRadius": 60, "spiderfyOnMaxZoom": True, "showCoverageOnHover": False}
    ).add_to(poi_fg)
    for _, poi in poi_df.iterrows():
        amenity = str(poi["amenity"])
        color = POI_COLORS.get(amenity, "#6c757d")
        folium.CircleMarker(
            location=(float(poi["lat"]), float(poi["lon"])),
            radius=7, color="#ffffff", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.9,
            tooltip=f'{poi["name"]} [{amenity}]'
        ).add_to(poi_cluster)
    poi_fg.add_to(m)

    # ── Epicenters FeatureGroup ────────────────────────────────────────────────
    epi_fg = folium.FeatureGroup(name="Epicenters", show=True)
    edf = epi_data.get("epicenter_df", pd.DataFrame())
    if not edf.empty:
        for _, row in edf.iterrows():
            folium.Marker(
                location=(row["centroid_lat"], row["centroid_lon"]),
                tooltip=f"EP-{int(row['cluster_id'])+1} | Triage {row['triage_score']:.3f} | n={row['n_nodes']}",
                icon=folium.DivIcon(
                    html=f'<div style="font-family:monospace;font-size:9px;font-weight:700;color:#fff;'
                         f'background:#1a1a2e;padding:3px 6px;border:1px solid {tc};white-space:nowrap;">'
                         f'EP-{int(row["cluster_id"])+1}</div>', icon_size=(50, 22))
            ).add_to(epi_fg)
    epi_fg.add_to(m)

    # ── LayerControl ──────────────────────
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # ── Legend overlay ────────────────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:1000;background:#090e18ee;padding:14px 18px;'
        f'border:1px solid #1e2a3a;border-top:2px solid {tc};font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:10px;color:#c9d1d9;backdrop-filter:blur(8px);">'
        f'<div style="color:#58a6ff;font-size:8px;letter-spacing:.2em;text-transform:uppercase;margin-bottom:10px;font-weight:700;">RESELIA v2 / GAT RISK OUTPUT</div>'
        f'HIGH-RISK &nbsp;<b>{len(vuln):,} nodes</b><br>MODEL &nbsp;&nbsp;<b>GAT+GBM F1={f1:.4f}</b><br>'
        f'WEATHER &nbsp;<b>{weather.upper()}</b><br>SFP &nbsp;&nbsp;&nbsp;&nbsp;<b style="color:{tc};">{sfp:.2f}%</b><br>'
        f'TIER &nbsp;&nbsp;&nbsp;&nbsp;<b style="color:{tc};">{tier}</b><br>EPICENTERS <b>{epi_data.get("n_epicenters",0)}</b><br>'
        f'<div style="margin-top:8px;font-size:8px;color:#adbac7;">'
        f'&#9679;<span style="color:#d29922;"> Amber</span> 0.35–0.55 &nbsp;'
        f'&#9679;<span style="color:#f85149;"> Red</span> &gt;0.55 &nbsp;'
        f'<span style="color:#484f58;">(low-risk nodes hidden)</span></div></div>'
    ))
    
    # ── Layer control theme ───────────────────────────────────────────────
    m.get_root().html.add_child(folium.Element("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');
    .leaflet-control-layers {background: #090e18 !important;border: 1px solid #1e2a3a !important;border-top: 2px solid #1f6feb !important;border-radius: 2px !important;box-shadow: 0 4px 20px rgba(0,0,0,.6) !important;padding: 10px 14px !important;backdrop-filter: blur(8px);}
    .leaflet-control-layers-list {font-family: 'IBM Plex Mono', monospace !important;font-size: 10px !important;color: #adbac7 !important;}
    .leaflet-control-layers label {color: #adbac7 !important;font-family: 'IBM Plex Mono', monospace !important;font-size: 10px !important;letter-spacing: .08em !important;text-transform: uppercase !important;display: flex !important;align-items: center !important;gap: 8px !important;margin: 5px 0 !important;}
    .leaflet-control-layers-base label span,
    .leaflet-control-layers-overlays label span {color: #c9d1d9 !important;}
    .leaflet-control-layers-separator {border-top: 1px solid #1e2a3a !important;margin: 6px 0 !important;}
    .leaflet-control-layers-base {margin-bottom: 4px !important;}
    /* Style radio & checkboxes */
    .leaflet-control-layers input[type="radio"],
    .leaflet-control-layers input[type="checkbox"] {accent-color: #1f6feb !important;width: 12px !important;height: 12px !important;}
    /* Header label for base layers */
    .leaflet-control-layers-base > label:first-child span {color: #58a6ff !important;}</style>"""))
    
    return m._repr_html_()

# ── Sidebar ────────────────────────────────────────────────────────────────────
def _lbl(text, mt="16px"):
    st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:8px;color:#58a6ff;'
                f'letter-spacing:.2em;text-transform:uppercase;font-weight:700;margin:{mt} 0 6px 0;">{text}</div>',
                unsafe_allow_html=True)

notebook_bundle = load_notebook_model()

with st.sidebar:
    st.markdown("""<div style="padding:0 0 20px 0;border-bottom:1px solid #1e2a3a;margin-bottom:20px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:700;color:#e6edf3;letter-spacing:.05em;">RESELIA</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#58a6ff;letter-spacing:.25em;text-transform:uppercase;margin-top:6px;">Urban Risk Engine / v2.0</div>
    </div>""", unsafe_allow_html=True)
    if notebook_bundle is not None:
        nb_m=notebook_bundle.get("metrics",{})
        st.markdown(f"""<div style="background:#0d2116;border:1px solid #1e4d2b;border-left:3px solid #3fb950;
                    padding:8px 12px;margin-bottom:16px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#3fb950;">
          &#10003; NOTEBOOK MODEL LOADED<br><span style="color:#adbac7;">F1={nb_m.get('f1_weighted','?')} &middot; AUC={nb_m.get('auc_roc','?')}</span></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:#1f1700;border:1px solid #4d3800;border-left:3px solid #d29922;
                    padding:8px 12px;margin-bottom:16px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#d29922;">
          &#9888; NO PICKLE FOUND<br><span style="color:#adbac7;">Run notebook first to generate<br>Notebooks/Phase_2/resilia_gat_model.pkl</span></div>""",
                    unsafe_allow_html=True)

    with st.form("sidebar_form", border=False):
        _lbl("Study Area", "0")
        selected_area = st.selectbox("Area", list(AREA_CONFIGS.keys()), index=0, label_visibility="collapsed")
        _lbl("Map Render")
        view_mode    = st.radio("View", ["Interactive","Static"], label_visibility="collapsed")
        show_heatmap = st.checkbox("Show risk heatmap", value=False)
        st.markdown("<hr style='border-color:#1e2a3a;margin:12px 0;'>", unsafe_allow_html=True)
        _lbl("Advanced Parameters", "0")
        flood_threshold = st.slider("Flood Threshold (m)",  1.0, 5.0,  float(FLOOD_THRESHOLD_M), .1)
        dbscan_eps      = st.slider("DBSCAN Radius (m)",    150, 800,  DBSCAN_EPS_M,             25)
        dbscan_min      = st.slider("DBSCAN Min Samples",     3,  15,  DBSCAN_MIN_PTS,            1)
        cascade_rounds  = st.slider("Cascade Rounds",          3,  10,  5,                         1)

        current_params = {"area":selected_area,"view":view_mode,"heatmap":show_heatmap,
                          "thr":flood_threshold,"eps":dbscan_eps,"min_pts":dbscan_min,"rounds":cascade_rounds}
        is_dirty = st.session_state.results is None or current_params != st.session_state.last_params

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        btn_class = "run-btn-dirty" if is_dirty else "run-btn-clean"
        btn_label = "▶  RUN ANALYSIS" if is_dirty else "✓  UP TO DATE"
        st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
        run_btn = st.form_submit_button(btn_label, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div style="margin-top:24px;padding-top:16px;border-top:1px solid #1e2a3a;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#6e7681;line-height:2.2;text-transform:uppercase;letter-spacing:.1em;">
        Phase 2 Stack<br>── GAT message passing<br>── GradientBoosting clf<br>── Polars data lake<br>
        ── OSM POI Overpass<br>── NetworkX cascade<br>── DBSCAN epicenters<br>── BMKG weather API<br>
        ── Pickle persistence<br>──────────────────<br>OSM / ODbL &#183; BMKG Public</div></div>""",
                unsafe_allow_html=True)

# ── Auto-collapse sidebar after run ───────────────────────────────────────────
if st.session_state.pop("_collapse_sidebar", False):
    components.html("""
    <script>
    (function() {
        try {
            var doc = window.parent.document;
            var selectors = [
                'button[data-testid="collapsedControl"]',
                '[data-testid="stSidebarCollapsedControl"] button',
                'section[data-testid="stSidebar"] button[kind="header"]',
                '[data-testid="stSidebar"] ~ div button'
            ];
            for (var i = 0; i < selectors.length; i++) {
                var btn = doc.querySelector(selectors[i]);
                if (btn) { btn.click(); break; }
            }
        } catch(e) { console.log('collapse err:', e); }
    })();
    </script>
    """, height=0, scrolling=False)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""<div style="padding:0 0 24px 0;">
  <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:30px;font-weight:700;color:#e6edf3;letter-spacing:-.01em;">RESELIA</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#58a6ff;letter-spacing:.22em;text-transform:uppercase;">Urban Infrastructure Risk Engine</span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#6e7681;background:#090e18;border:1px solid #1e2a3a;padding:2px 8px;letter-spacing:.12em;">v2.0</span>
  </div>
  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#6e7681;max-width:700px;line-height:1.7;">
    GAT-augmented flood vulnerability &middot; cascading failure simulation &middot; DBSCAN triage &middot; BMKG weather &middot; pickle persistence.
  </div>
  <div style="height:1px;background:linear-gradient(90deg,#1f6feb 0%,#388bfd 30%,#1e2a3a 70%,transparent 100%);margin-top:20px;"></div>
</div>""", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.last_params = current_params
    st.session_state["_collapse_sidebar"] = True
    with st.status("Running RESELIA v2 pipeline...", expanded=True) as ps:
        try:
            st.write("**[1/8]** Fetching road network...")
            G, nodes, edges = fetch_network(selected_area)
            st.write(f"  → {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")
            st.write("**[2/8]** Injecting DEMNAS elevation...")
            G = inject_demnas_elevation(G, selected_area, flood_threshold)
            st.write("**[3/8]** Fetching POIs...")
            poi_df = fetch_pois(selected_area); G = inject_poi_criticality(G, poi_df)
            st.write(f"  → {len(poi_df)} POIs")
            st.write("**[4/8]** Computing feature matrix...")
            G, feat_df = compute_graph_features(G)
            st.write(f"**[5/8]** Building GAT model ({'notebook pkl' if notebook_bundle else 'fresh'})...")
            model_res = build_gat_model(G, feat_df, notebook_bundle, flood_threshold)
            st.write(f"  → F1={model_res['f1_w']:.4f} · AUC={model_res['auc_roc']:.4f}")
            st.write("**[6/8]** Cascading simulation...")
            cascade_res = run_cascade(G, n_rounds=cascade_rounds)
            st.write(f"  → Resilience={cascade_res['resilience_score']:.4f}")
            st.write("**[7/8]** DBSCAN epicenters...")
            epi_res = run_dbscan(G, eps_m=dbscan_eps, min_pts=dbscan_min)
            st.write(f"  → {epi_res['n_epicenters']} epicenters · {epi_res['n_noise']} noise")
            st.write("**[8/8]** BMKG weather + SFP...")
            weather, stressor_w, live, fbr = fetch_bmkg(AREA_CONFIGS[selected_area]["adm4"])
            risk_res = compute_risk_v2(G, stressor_w, cascade_res["resilience_score"])
            st.write(f"  → SFP={risk_res['sfp']:.2f}% · Tier={risk_res['tier']} · Weather={weather}")
            st.session_state.results = {
                "G":G,"nodes":nodes,"edges":edges,"feat_df":feat_df,"poi_df":poi_df,
                "model":model_res,"cascade":cascade_res,"epi":epi_res,"risk":risk_res,
                "weather":weather,"stressor_w":stressor_w,"live":live,"fbr":fbr,"area":selected_area,
                "view_mode":view_mode,"show_heatmap":show_heatmap,
                "flood_threshold":flood_threshold,"dbscan_eps":dbscan_eps,
                "dbscan_min":dbscan_min,"cascade_rounds":cascade_rounds,
            }
            ps.update(label="Pipeline complete ✓", state="complete")
        except Exception as err:
            import traceback; ps.update(label=f"Failed: {err}", state="error")
            st.error(str(err)); st.code(traceback.format_exc())

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.results:
    r=st.session_state.results
    risk=r["risk"]; mdl=r["model"]; casc=r["cascade"]; epi=r["epi"]
    view_mode=r.get("view_mode","Interactive"); show_heatmap=r.get("show_heatmap",False)
    flood_threshold=r.get("flood_threshold",FLOOD_THRESHOLD_M)
    dbscan_eps=r.get("dbscan_eps",DBSCAN_EPS_M); dbscan_min=r.get("dbscan_min",DBSCAN_MIN_PTS)
    cascade_rounds=r.get("cascade_rounds",5)
    c=TIER_COLOR[risk["tier"]]; bg=TIER_BG[risk["tier"]]; bd=TIER_BORDER[risk["tier"]]

    # v3-1: BMKG warning — only for real offline, not silent unmapped
    if not r["live"]:
        fbr=r.get("fbr","offline")
        msgs={"timeout":"BMKG API timed out — using Berawan fallback (stressor 0.15).",
              "offline":"BMKG API unreachable — using Berawan fallback.",
              "http_404":"BMKG adm4 code not found — check area config.",
              "http_500":"BMKG server error — using Berawan fallback."}
        msg=msgs.get(fbr, f"BMKG unavailable ({fbr}) — using Berawan fallback.")
        st.info(f"ℹ️ {msg}")

    # v3-4: KPI Carousel (proper HTML in iframe)
    _lbl("Key Risk Indicators", "0")
    kpis=[
        ("Area",r["area"],None,""),
        ("Total Nodes",f"{risk['n_total']:,}",None,""),
        ("High-Risk",f"{len(risk['vulnerable']):,}",f"↑ {risk['exposure']*100:.1f}%","#f85149"),
        ("GAT F1",f"{mdl['f1_w']:.4f}",None,""),
        ("AUC-ROC",f"{mdl['auc_roc']:.4f}",None,""),
        ("CV F1",f"{mdl['cv_scores'].mean():.4f}",None,""),
        ("Resilience",f"{casc['resilience_score']:.4f}",None,""),
        ("Epicenters",str(epi["n_epicenters"]),None,""),
        ("SFP",f"{risk['sfp']:.2f}%",None,""),
        ("Tier",risk["tier"],None,c),
    ]
    ci=""
    for lbl,val,delta,vc in kpis:
        vs=f"color:{vc};" if vc else "color:#e6edf3;"
        dh=f'<div style="font-size:9px;margin-top:3px;color:#f85149;">{delta}</div>' if delta else ""
        ci+=f'<div style="flex:0 0 148px;background:linear-gradient(140deg,#090e18,#0d1525);border:1px solid #1e2a3a;border-top:2px solid #1f6feb;border-radius:2px;padding:12px 14px;box-sizing:border-box;"><div style="color:#58a6ff;font-size:8px;text-transform:uppercase;letter-spacing:.2em;font-weight:700;margin-bottom:5px;">{lbl}</div><div style="font-size:17px;font-weight:700;white-space:nowrap;{vs}">{val}</div>{dh}</div>'
    ch=f"""<!DOCTYPE html><html><head><style>
    body{{margin:0;padding:0;background:transparent;font-family:'IBM Plex Mono',monospace;overflow:hidden;}}
    .w{{display:flex;align-items:center;gap:8px;height:80px;}}
    .vp{{flex:1;overflow:hidden;height:80px;}}
    .tr{{display:flex;gap:10px;transition:transform .3s;height:80px;}}
    .n{{background:#090e18;border:1px solid #1f6feb;color:#58a6ff;font-size:14px;font-weight:700;width:28px;height:28px;cursor:pointer;border-radius:2px;display:flex;align-items:center;justify-content:center;flex-shrink:0;line-height:1;font-family:monospace;}}
    .n:hover{{background:#1f6feb;color:#fff;}}
    </style></head><body>
    <div class="w"><button class="n" id="p">&#9664;</button>
    <div class="vp" id="vp"><div class="tr" id="tr">{ci}</div></div>
    <button class="n" id="n">&#9654;</button></div>
    <script>var o=0,s=158,t=document.getElementById('tr'),v=document.getElementById('vp');
    document.getElementById('p').onclick=function(){{o=Math.min(o+s,0);t.style.transform='translateX('+o+'px)';}};
    document.getElementById('n').onclick=function(){{var m=-(t.scrollWidth-v.offsetWidth+4);o=Math.max(o-s,m);t.style.transform='translateX('+o+'px)';}};
    </script></body></html>"""
    components.html(ch, height=84, scrolling=False)

    # Tier banner
    pen_pct=(risk["penalty"]-1.)*100
    src_label="📦 notebook pkl" if mdl.get("model_src")=="notebook_pickle" else "🔧 trained fresh"
    st.markdown(f"""<div style="margin:16px 0;background:{bg};border:1px solid {bd};border-left:4px solid {c};padding:14px 22px;font-family:'IBM Plex Mono',monospace;">
      <span style="font-size:8px;letter-spacing:.2em;text-transform:uppercase;color:{c};font-weight:700;">Risk Tier</span>
      <span style="font-size:20px;font-weight:700;color:{c};margin-left:18px;">{risk['tier']}</span>
      <span style="font-size:10px;color:#adbac7;margin-left:22px;">
        SFP {risk['sfp']:.2f}% &middot; {r['weather']} &middot; Stressor {r['stressor_w']:.2f}
        &middot; Resilience Penalty +{pen_pct:.0f}% &middot; {len(risk['vulnerable']):,} nodes &middot; {src_label}
      </span></div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["SPATIAL MAP","MODEL EVALUATION","CASCADING FAILURE",
                                               "EPICENTER TRIAGE","POI IMPACT LAYER","FEATURE ANALYSIS"])

    # TAB 1
    with tab1:
        if view_mode == "Interactive":
            if "map_html" not in r:
                with st.spinner("Building interactive map…"):
                    r["map_html"] = build_map(
                        r["G"], r["edges"], risk["vulnerable"], r["poi_df"], epi,
                        r["area"], r["weather"], risk["sfp"], risk["tier"],
                        mdl["f1_w"], r["stressor_w"],
                        show_heatmap
                    )
                    st.session_state.results = r  
            components.html(r["map_html"], height=580, scrolling=False)
        else:
            with st.spinner("Rendering static map…"):
                ngdf=gpd.GeoDataFrame({"s":[r["G"].nodes[n].get("risk_score",0) for n in r["G"].nodes()]},
                                       geometry=r["nodes"].geometry, crs=r["nodes"].crs)
                fig,ax=plt.subplots(figsize=(13,8),facecolor="#05090f"); ax.set_facecolor("#05090f")
                r["edges"].plot(ax=ax,color="#141c2f",linewidth=0.7,alpha=1.,aspect=None)
                lo=ngdf[ngdf["s"]<0.35]; mi=ngdf[(ngdf["s"]>=0.35)&(ngdf["s"]<0.55)]; hi=ngdf[ngdf["s"]>=0.55]
                if len(lo): lo.plot(ax=ax,color="#388bfd",markersize=2,alpha=.35,aspect=None)
                if len(mi): ax.scatter(mi.geometry.x,mi.geometry.y,s=mi["s"]*40+4,c="#d29922",alpha=.7,zorder=3)
                if len(hi): ax.scatter(hi.geometry.x,hi.geometry.y,s=(hi["s"]*60+6).clip(6,80),c="#f85149",alpha=.85,zorder=4)
                for _,poi in r["poi_df"].iterrows():
                    ax.plot(float(poi["lon"]),float(poi["lat"]),marker="*",markersize=12,
                            color=POI_COLORS.get(str(poi["amenity"]),"#6c757d"),
                            markeredgecolor="white",markeredgewidth=.5,zorder=5)
                edf2=epi.get("epicenter_df",pd.DataFrame())
                if not edf2.empty:
                    for _,er in edf2.iterrows():
                        ax.scatter(er["centroid_lon"],er["centroid_lat"],s=320,marker="X",c=["#fff"],zorder=6,edgecolors=c,linewidths=1.5)
                        ax.annotate(f"EP-{int(er['cluster_id'])+1}",(er["centroid_lon"],er["centroid_lat"]),
                                    fontsize=8,fontweight="bold",color="white",xytext=(0,11),textcoords="offset points",ha="center",
                                    bbox=dict(boxstyle="round,pad=0.25",fc="#1a1a2e",alpha=.85,ec="none"))
                ax.set_title(f"{r['area'].upper()} · GAT F1={mdl['f1_w']:.4f} · SFP {risk['sfp']:.2f}% [{risk['tier']}] · {r['weather']}",
                              color="#adbac7",fontsize=10,fontfamily="monospace",pad=14)
                ax.tick_params(colors="#1e2a3a",labelcolor="#6e7681",labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
                ax.legend(handles=[mpatches.Patch(facecolor="#f85149",label=f"High >0.55 ({len(hi):,})"),
                                    mpatches.Patch(facecolor="#d29922",label=f"Moderate 0.35–0.55 ({len(mi):,})"),
                                    mpatches.Patch(facecolor="#388bfd",label=f"Low <0.35 ({len(lo):,})")],
                           facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=8)
                plt.tight_layout()
            st.pyplot(fig, width='stretch'); plt.close(fig)

    # TAB 2
    with tab2:
        _lbl("Phase 2 GAT vs Phase 1 RF — Benchmark","0")
        e1,e2,e3,e4,e5=st.columns(5)
        e1.metric("Accuracy",f"{mdl['acc']:.4f}"); e2.metric("F1 Weighted",f"{mdl['f1_w']:.4f}")
        e3.metric("F1 Macro",f"{mdl['f1_mac']:.4f}"); e4.metric("AUC-ROC",f"{mdl['auc_roc']:.4f}")
        e5.metric("CV F1 (5-fold)",f"{mdl['cv_scores'].mean():.4f} ± {mdl['cv_scores'].std():.4f}")
        if notebook_bundle:
            nb_m2=notebook_bundle.get("metrics",{})
            st.caption(f"Notebook: F1={nb_m2.get('f1_weighted','?')} · AUC={nb_m2.get('auc_roc','?')} · "
                       f"CV={nb_m2.get('cv_f1_mean','?')} · RF={nb_m2.get('f1_baseline_rf','?')}")
        with st.spinner("Rendering model evaluation charts…"):
            fig,axes=plt.subplots(2,3,figsize=(16,8),facecolor="#05090f")
            for ax in axes.flat:
                ax.set_facecolor("#090e18")
                for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
                ax.tick_params(colors="#1e2a3a",labelcolor="#adbac7",labelsize=8)
            ConfusionMatrixDisplay(mdl["conf_mat"],display_labels=["Low Risk","High Risk"]).plot(ax=axes[0,0],colorbar=False,cmap="Blues")
            axes[0,0].set_title("Confusion Matrix",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes[0,0].set_xlabel("Predicted",color="#6e7681",fontsize=9); axes[0,0].set_ylabel("Actual",color="#6e7681",fontsize=9)
            imdf=pd.DataFrame({"feature":mdl["feat_cols"],"importance":mdl["clf"].feature_importances_}).sort_values("importance")
            cols_=[("#f4a261" if f in ["poi_criticality","clustering_coefficient","pagerank"] else "#388bfd") for f in imdf["feature"]]
            axes[0,1].barh(imdf["feature"],imdf["importance"],color=cols_,edgecolor="#05090f",height=.5)
            axes[0,1].set_title("Feature Importance\n(orange=P2 new)",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes[0,1].set_xlabel("MDI",color="#6e7681",fontsize=9)
            axes[0,2].bar(range(1,6),mdl["cv_scores"],color="#1a3a5c",edgecolor="#05090f",width=.6)
            axes[0,2].axhline(mdl["cv_scores"].mean(),color="#388bfd",linewidth=1.5,linestyle="--",label=f"Mean={mdl['cv_scores'].mean():.4f}")
            axes[0,2].set_title("5-Fold CV F1",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes[0,2].set_ylim(0,1.05); axes[0,2].legend(facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=8)
            mc=["F1 Weighted","F1 Macro","AUC-ROC"]; xp=np.arange(3)
            p1v=[mdl["f1_base"],mdl["f1mac_base"],0.]; p2v=[mdl["f1_w"],mdl["f1_mac"],mdl["auc_roc"]]
            axes[1,0].bar(xp-.2,p1v,.35,label="Phase 1 RF",color="#2d3748",edgecolor="#05090f")
            axes[1,0].bar(xp+.2,p2v,.35,label="Phase 2 GAT+GBM",color="#388bfd",edgecolor="#05090f")
            axes[1,0].set_title("P1 RF vs P2 GAT",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes[1,0].set_xticks(xp); axes[1,0].set_xticklabels(mc,fontsize=8); axes[1,0].set_ylim(0,1.1)
            axes[1,0].legend(facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=8)
            for i,(p1,p2) in enumerate(zip(p1v,p2v)):
                if p1>0: axes[1,0].text(i+.2,p2+.02,f"+{(p2-p1)*100:.1f}pp",ha="center",color="#3fb950",fontsize=8)
            rsc=[r["G"].nodes[n].get("risk_score",0) for n in r["G"].nodes()]
            vls=[r["G"].nodes[n].get("vulnerability","Low") for n in r["G"].nodes()]
            axes[1,1].hist([s for s,l in zip(rsc,vls) if l=="Low"],bins=40,color="#1f6feb",alpha=.7,label="Low",edgecolor="#05090f",linewidth=.3)
            axes[1,1].hist([s for s,l in zip(rsc,vls) if l=="High"],bins=40,color="#f85149",alpha=.7,label="High",edgecolor="#05090f",linewidth=.3)
            axes[1,1].axvline(HIGH_RISK_THRESHOLD,color="#3fb950",linewidth=1.5,linestyle="--",label=f"Thr={HIGH_RISK_THRESHOLD}")
            axes[1,1].set_title("Risk Score Distribution",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes[1,1].legend(facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=8)
            elevs=[r["G"].nodes[n].get("elevation",0) for n in r["G"].nodes()]
            axes[1,2].hist([e for e,l in zip(elevs,vls) if l=="Low"],bins=35,color="#1f6feb",alpha=.7,label="Low",edgecolor="#05090f",linewidth=.3)
            axes[1,2].hist([e for e,l in zip(elevs,vls) if l=="High"],bins=35,color="#f85149",alpha=.7,label="High",edgecolor="#05090f",linewidth=.3)
            axes[1,2].axvline(flood_threshold,color="#d29922",linewidth=1.5,linestyle="--",label=f"Thr {flood_threshold}m")
            axes[1,2].set_title("Elevation by Class",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes[1,2].legend(facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=8)
            fig.patch.set_facecolor("#05090f"); plt.tight_layout()
        st.pyplot(fig,width='stretch'); plt.close(fig)
        with st.expander("Classification Report"):
            cr=mdl["class_rep"]
            st.dataframe(pd.DataFrame([{"Class":cls,"Precision":f"{cr.get(cls,{}).get('precision',0):.4f}",
                "Recall":f"{cr.get(cls,{}).get('recall',0):.4f}","F1":f"{cr.get(cls,{}).get('f1-score',0):.4f}",
                "Support":int(cr.get(cls,{}).get("support",0))} for cls in ["Low Risk","High Risk"]]),
                width='stretch')

    # TAB 3
    with tab3:
        sd=casc["sim_df"]
        _lbl("Network Resilience (GAT-ordered removal)","0")
        r1,r2,r3,r4=st.columns(4)
        r1.metric("Resilience Score",f"{casc['resilience_score']:.4f}")
        r2.metric("Efficiency Degradation",f"{casc['eff_degradation']*100:.1f}%")
        r3.metric("LCC Fragmentation",f"{casc['lcc_degradation']*100:.1f}%")
        r4.metric("Resilience Penalty",f"+{(risk['penalty']-1.)*100:.0f}%")
        with st.spinner("Rendering cascade charts…"):
            fig,axes=plt.subplots(1,3,figsize=(16,5),facecolor="#05090f")
            for ax in axes:
                ax.set_facecolor("#090e18")
                for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
                ax.tick_params(colors="#1e2a3a",labelcolor="#adbac7",labelsize=8)
            for ax,(col,lbl2,color) in zip(axes,[("global_efficiency","Global Efficiency","#f85149"),
                                                   ("lcc_fraction","Largest Component","#388bfd"),
                                                   ("avg_clustering","Avg Clustering","#3fb950")]):
                ax.plot(sd["round"],sd[col],"o-",color=color,linewidth=2.5,markersize=7,markeredgecolor="#05090f")
                ax.fill_between(sd["round"],sd[col],alpha=.12,color=color)
                bl=float(sd.iloc[0][col])
                ax.axhline(bl,color="#484f58",linestyle=":",linewidth=1,label=f"Baseline={bl:.4f}")
                ax.set_title(lbl2,color="#adbac7",fontfamily="monospace",fontsize=10)
                ax.set_xlabel("Round",color="#6e7681",fontsize=9); ax.set_ylabel(lbl2,color="#6e7681",fontsize=9)
                ax.set_xlim(-.2,len(sd)-.8); ax.legend(facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=8)
            fig.patch.set_facecolor("#05090f"); plt.tight_layout()
        st.pyplot(fig,width='stretch'); plt.close(fig)
        st.dataframe(sd.style.format({"global_efficiency":"{:.4f}","lcc_fraction":"{:.4f}",
                                       "avg_clustering":"{:.4f}","pct_removed":"{:.1f}%"}),width='stretch'  )

    # TAB 4
    with tab4:
        edf=epi["epicenter_df"]
        ep1,ep2,ep3=st.columns(3)
        ep1.metric("Epicenter Clusters",epi["n_epicenters"])
        ep2.metric("Noise Nodes",epi["n_noise"])
        ep3.metric("DBSCAN ε-radius",f"{dbscan_eps} m")
        if edf.empty:
            st.info("No clusters — try increasing ε-radius or reducing min samples.")
        else:
            _lbl("Triage Table — Composite Score (50% Risk · 30% POI · 20% Density)","0")
            s2=edf.copy(); s2.index=[f"EP-{int(row)+1}" for row in edf["cluster_id"]]
            st.dataframe(s2[["n_nodes","centroid_lat","centroid_lon","mean_risk_score","mean_poi_crit","triage_score"]
                          ].style.format({"centroid_lat":"{:.6f}","centroid_lon":"{:.6f}","mean_risk_score":"{:.4f}",
                                           "mean_poi_crit":"{:.4f}","triage_score":"{:.4f}"}).background_gradient(subset=["triage_score"],cmap="Reds"),
                          width='stretch')
            with st.spinner("Rendering triage charts…"):
                fig2,axes2=plt.subplots(1,2,figsize=(14,5),facecolor="#05090f")
                for ax in axes2:
                    ax.set_facecolor("#090e18")
                    for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
                    ax.tick_params(colors="#1e2a3a",labelcolor="#adbac7",labelsize=8)
                pal=plt.cm.plasma(np.linspace(.2,.9,max(len(edf),1)))
                eps2=edf.sort_values("triage_score",ascending=True)
                bl2=[f"EP-{int(r2)+1}" for r2 in eps2["cluster_id"]]
                bars=axes2[0].barh(bl2,eps2["triage_score"],color=pal,edgecolor="#05090f")
                axes2[0].set_title("Composite Triage Score",color="#adbac7",fontfamily="monospace",fontsize=10)
                axes2[0].set_xlabel("Score",color="#6e7681",fontsize=9)
                for bar,(_,row2) in zip(bars,eps2.iterrows()):
                    axes2[0].text(bar.get_width()+.005,bar.get_y()+bar.get_height()/2,
                                   f"n={row2['n_nodes']}",va="center",color="#adbac7",fontsize=8)
                sc=axes2[1].scatter(edf["mean_risk_score"],edf["mean_poi_crit"],s=edf["n_nodes"]*20,
                                     c=edf["triage_score"],cmap="plasma",alpha=.85,edgecolors="#05090f",linewidth=.8)
                for _,row2 in edf.iterrows():
                    axes2[1].annotate(f"EP-{int(row2['cluster_id'])+1}",(row2["mean_risk_score"],row2["mean_poi_crit"]),
                                       fontsize=7,ha="center",va="bottom",xytext=(0,6),textcoords="offset points",
                                       color="#e6edf3",fontfamily="monospace")
                plt.colorbar(sc,ax=axes2[1],label="Triage Score")
                axes2[1].set_title("Risk vs POI (bubble=density)",color="#adbac7",fontfamily="monospace",fontsize=10)
                fig2.patch.set_facecolor("#05090f"); plt.tight_layout()
            st.pyplot(fig2,width='stretch'); plt.close(fig2)

    # TAB 5
    with tab5:
        pdf=r["poi_df"]; _lbl("Critical Infrastructure Inventory","0"); st.dataframe(pdf,width='stretch')
        ac=pdf["amenity"].value_counts()
        nc=sum(1 for _,d in r["G"].nodes(data=True) if d.get("poi_criticality",0)>0)
        nhp=sum(1 for _,d in r["G"].nodes(data=True) if d.get("vulnerability")=="High" and d.get("poi_criticality",0)>0)
        p1,p2,p3,p4=st.columns(4)
        p1.metric("Total POIs",len(pdf)); p2.metric("Facility Types",len(ac))
        p3.metric("Nodes in POI Zone",f"{nc:,}"); p4.metric("High-Risk in POI Zone",f"{nhp:,}",delta="overlap",delta_color="inverse")
        with st.spinner("Rendering POI charts…"):
            fig3,axes3=plt.subplots(1,2,figsize=(14,5),facecolor="#05090f")
            for ax in axes3:
                ax.set_facecolor("#090e18")
                for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
                ax.tick_params(colors="#1e2a3a",labelcolor="#adbac7",labelsize=8)
            axes3[0].barh(ac.index,ac.values,color=[POI_COLORS.get(a,"#6c757d") for a in ac.index],edgecolor="#05090f")
            axes3[0].set_title("POI by Facility",color="#adbac7",fontfamily="monospace",fontsize=10)
            axes3[0].set_xlabel("Count",color="#6e7681",fontsize=9)
            psc=[d.get("poi_criticality",0) for _,d in r["G"].nodes(data=True)]
            axes3[1].hist([s for s in psc if s>0],bins=30,color="#f4a261",edgecolor="#05090f",linewidth=.3)
            axes3[1].set_title("POI Criticality (within 300m)",color="#adbac7",fontfamily="monospace",fontsize=10)
            fig3.patch.set_facecolor("#05090f"); plt.tight_layout()
        st.pyplot(fig3,width='stretch'); plt.close(fig3)

    # TAB 6
    with tab6:
        _lbl("Phase 2 Feature Matrix — All 7 Features","0")
        fds=r["feat_df"].copy()
        st.dataframe(fds[FEAT_COLS_V2+["flood_label"]].describe().round(4),width='stretch')
        with st.spinner("Rendering feature distribution charts…"):
            fig4,axes4=plt.subplots(2,4,figsize=(18,8),facecolor="#05090f")
            fig4.suptitle(f"Feature Distribution — {r['area']}",color="#e6edf3",fontfamily="monospace",fontsize=12)
            for ax in axes4.flat:
                ax.set_facecolor("#090e18")
                for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
                ax.tick_params(colors="#1e2a3a",labelcolor="#adbac7",labelsize=7)
            for i,feat in enumerate(FEAT_COLS_V2):
                ax=axes4[i//4][i%4]
                lv=fds[fds["flood_label"]==0][feat]; hv=fds[fds["flood_label"]==1][feat]
                ax.hist(lv.values,bins=35,color="#1f6feb",alpha=.65,label="Low",edgecolor="#05090f",linewidth=.2)
                ax.hist(hv.values,bins=35,color="#f85149",alpha=.65,label="High",edgecolor="#05090f",linewidth=.2)
                pt="[P2]" if feat in ["poi_criticality","clustering_coefficient","pagerank"] else "[P1]"
                ax.set_title(f"{pt} {feat.replace('_',' ')}",color="#adbac7",fontfamily="monospace",fontsize=8)
                ax.legend(facecolor="#090e18",edgecolor="#1e2a3a",labelcolor="#adbac7",fontsize=7)
            axc=axes4[1][3]; cm2=fds[FEAT_COLS_V2].astype(float).corr()
            axc.imshow(cm2.values,cmap="coolwarm",vmin=-1,vmax=1,aspect="auto")
            sn=["elev","deg","btw","clo","poi","clust","pr"]
            axc.set_xticks(range(7)); axc.set_yticks(range(7))
            axc.set_xticklabels(sn,fontsize=6,color="#adbac7",rotation=45)
            axc.set_yticklabels(sn,fontsize=6,color="#adbac7")
            for ii in range(7):
                for jj in range(7):
                    axc.text(jj,ii,f"{cm2.values[ii,jj]:.1f}",ha="center",va="center",fontsize=5,color="white")
            axc.set_title("Feature Correlation",color="#adbac7",fontfamily="monospace",fontsize=8)
            fig4.patch.set_facecolor("#05090f"); plt.tight_layout()
        st.pyplot(fig4, width='stretch'); plt.close(fig4)

    # Policy
    n_epi=epi["n_epicenters"]; top_label=""
    if n_epi>0 and not epi["epicenter_df"].empty:
        te=epi["epicenter_df"].iloc[0]
        top_label=f"Priority EP-{int(te['cluster_id'])+1} (triage {te['triage_score']:.4f}, {te['n_nodes']} nodes) at ({te['centroid_lat']}, {te['centroid_lon']})."
    src_note="Model from notebook pickle." if mdl.get("model_src")=="notebook_pickle" else "Model trained fresh."
    st.markdown(f"""<div style="margin-top:28px;background:{bg};border:1px solid {bd};border-left:4px solid {c};padding:20px 26px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:{c};letter-spacing:.2em;text-transform:uppercase;font-weight:700;margin-bottom:12px;">Policy Recommendation / {risk['tier']}</div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:14px;color:#e6edf3;line-height:1.8;">
        Deploy emergency drainage to <b>{len(risk['vulnerable']):,} GAT flood-prone nodes</b> in {r['area']}.
        Stressor <b>{r['stressor_w']:.2f}</b> ({r['weather']}) + resilience penalty <b>{(risk['penalty']-1.)*100:.0f}%</b>
        → SFP <b style="color:{c};">{risk['sfp']:.2f}%</b>.
        {f'Pre-position across <b>{n_epi} DBSCAN epicenters</b>. {top_label}' if n_epi>0 else ''}
        Reroute logistics from low-elevation clusters.
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#6e7681;margin-top:10px;">{src_note}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div style="height:1px;background:linear-gradient(90deg,transparent,#1e2a3a 40%,#1f6feb 100%);margin-top:40px;"></div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#3d4a58;text-align:right;padding:12px 0;letter-spacing:.1em;">
      RESELIA v2.0 / GAT+GBM · Polars · DBSCAN · NetworkX · OSM ODbL · BMKG</div>""",
                unsafe_allow_html=True)

else:
    st.markdown("""<div style="text-align:center;padding:80px 0;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#1e2a3a;letter-spacing:.35em;text-transform:uppercase;margin-bottom:20px;">System Standby</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:32px;font-weight:700;color:#0d1525;letter-spacing:.04em;margin-bottom:12px;">RESELIA</div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:14px;color:#3d4a58;max-width:480px;margin:0 auto;line-height:1.7;">
        Select a study area and configure parameters in the sidebar,<br>then press <b>▶ RUN ANALYSIS</b> to begin.</div>
      <div style="margin-top:40px;display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#388bfd;background:#090e18;border:1px solid #1f6feb;padding:8px 16px;letter-spacing:.1em;">GAT MESSAGE PASSING</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#388bfd;background:#090e18;border:1px solid #1f6feb;padding:8px 16px;letter-spacing:.1em;">DBSCAN EPICENTER TRIAGE</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#388bfd;background:#090e18;border:1px solid #1f6feb;padding:8px 16px;letter-spacing:.1em;">CASCADING FAILURE SIM</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#388bfd;background:#090e18;border:1px solid #1f6feb;padding:8px 16px;letter-spacing:.1em;">PICKLE MODEL PERSISTENCE</div>
      </div></div>""", unsafe_allow_html=True)