# RESILIA — Urban Infrastructure Risk Engine

> Real-time flood vulnerability assessment for Jakarta's logistics network.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resilia-dashboard.streamlit.app)

## Stack

- **Road Network** — OpenStreetMap via `osmnx`
- **ML Model** — Random Forest (Phase 1 baseline)
- **Weather** — BMKG Live Public API
- **Visualization** — Folium interactive map + Matplotlib

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo Structure

```
resilia-app/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── config.toml         # Theme + server config
```

## Data Sources

| Source | License | Usage |
|--------|---------|-------|
| OpenStreetMap via osmnx | ODbL | Road network topology |
| BMKG Public API | Public | Live weather telemetry |
| DEM (simulated) | — | Elevation proxy (Phase 1 PoC) |

---
*RESILIA PoC — Competition submission.*
