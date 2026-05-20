# RESELIA — Urban Infrastructure Risk Engine

> Phase 2: GAT-augmented flood vulnerability assessment with cascading failure simulation, DBSCAN epicenter triage, and real-time BMKG weather stressor integration.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reselia.streamlit.app/)

## Phase 2 Upgrades

| # | Module | Phase 1 | Phase 2 |
|---|--------|---------|---------|
| 1 | **Core AI** | Random Forest (tabular) | GAT message passing + GradientBoosting |
| 2 | **Elevation Model** | Coordinate approximation | DEMNAS-calibrated Polars backend |
| 3 | **Feature Matrix** | 3 graph centrality features | 7 features incl. POI criticality, clustering, PageRank |
| 4 | **Risk Score** | Static exposure × stressor | SFP with network resilience penalty |
| 5 | **Spatial Analysis** | None | DBSCAN epicenter triage |
| 6 | **Network Analysis** | None | Cascading failure simulation (NetworkX) |
| 7 | **POI Layer** | None | OSM critical infrastructure fusion (Overpass API) |

## Stack

- **Road Network** — OpenStreetMap via `osmnx`
- **AI Model** — GAT neighbourhood aggregation (2-hop attention) + `GradientBoostingClassifier`
- **Data Backend** — `polars` for DEMNAS elevation engineering
- **POI Layer** — Overpass API (hospitals, schools, markets, emergency services)
- **Cascade Sim** — `networkx` progressive failure degradation
- **Epicenter Triage** — `sklearn` DBSCAN spatial clustering
- **Weather** — BMKG Live Public API
- **Visualization** — Folium interactive map (heatmap layer) + Matplotlib

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo Structure

```
resilia-app/
├── app.py                  # Main Streamlit dashboard (v2)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level packages (Streamlit Cloud)
└── .streamlit/
    └── config.toml         # Theme + server config
```

## Data Sources

| Source | License | Usage |
|--------|---------|-------|
| OpenStreetMap via osmnx | ODbL | Road network topology |
| Overpass API | ODbL | Critical infrastructure POIs |
| BMKG Public API | Public | Live weather telemetry |
| DEMNAS (simulated) | — | Micro-topographic elevation model (PoC) |

## Study Areas

| Area | Center | Coverage |
|------|--------|----------|
| Kemayoran | −6.1600, 106.8600 | 2,200 m radius |
| Penjaringan | −6.1200, 106.8000 | 2,000 m radius |
| Pluit | −6.1100, 106.7900 | 1,800 m radius |
| Cengkareng | −6.1500, 106.7400 | 2,000 m radius |

---
*RESELIA v2.0 — Phase 2 implementation.*
