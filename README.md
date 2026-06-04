# RESILIA v2.0

> **Autonomous Triage & Cascading Failure Prediction for Urban Logistics**
>
> Resilience & Early-Warning System for Infrastructure & Logistic Impact Assessment
> 
> *Official Submission for AI Impact Challenge Datathon 2026 - Urban Resilience & Smart City*

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)](https://resilia.streamlit.app/)

RESILIA adalah mesin intelijen tata ruang (*Spatial Intelligence Engine*) berbasis *Business-to-Government* (B2G) yang dirancang untuk mencegah kelumpuhan logistik kota akibat anomali cuaca ekstrem. Menggeser paradigma dari "dasbor pemetaan banjir statis", RESILIA mengorkestrasi tiga lapisan *Artificial Intelligence* (AI) untuk memprediksi probabilitas efek domino (*Cascading Failure*) pada infrastruktur kritis.

## The "AI Triad" & Cloud Architecture

RESILIA tidak mengandalkan model *Machine Learning* tabular konvensional. Sistem ini direkayasa dengan pendekatan **Physics-Informed Neural Networks** dan interoperabilitas **Hybrid-Cloud**:

1. **Graph Attention Network (GAT):** Memodelkan jalan raya Jakarta sebagai graf terarah. GAT menghitung atensi kerentanan (*Message Passing*) menggunakan proksi elevasi fisika (DEMNAS) dan topologi OSMnx.
2. **DBSCAN Spatial Triage:** Algoritma *Unsupervised Learning* beroperasi dengan metrik *Haversine* untuk secara otonom mengisolasi area kerentanan kritis menjadi "Episenter", memberikan rekomendasi pengerahan alat berat BPBD yang presisi.
3. **Autonomous LLM Policy Agent:** Menggunakan model instruksional (*OpenRouter API*) untuk merender output spasial menjadi draf Surat Perintah Kerja (SPK) secara instan.
4. **Microsoft Azure Hybrid Sync:** Arsitektur *Graceful Degradation* (Failover). Menggunakan `azure.storage.blob` sebagai repositori *Data Lake* asinkron untuk sinkronisasi draf kebijakan B2G dengan kepastian *Zero-Downtime*.

## Enterprise Capabilities (v2.0)

- **Offline Mode Failover:** Sistem inferensi dieksekusi 100% secara lokal (*Edge Compute*). Jika API cuaca publik terputus saat bencana (Blackout), sistem menyimulasikan skenario *stressor* maksimum tanpa *crash*.
- **Cascading Failure Simulation:** Menggunakan algoritma pemangkasan node (`NetworkX`) iteratif untuk menghitung degradasi efisiensi jaringan (*Network Efficiency Drop*) saat urat nadi arteri lumpuh.
- **Micro-Targeted POI Protection:** Menghitung radius isolasi spasial terhadap fasilitas vital (Rumah Sakit, Polisi, Pemadam Kebakaran) secara *real-time* via Overpass API.

## Open Data Compliance & Sources

Seluruh pipeline data mematuhi standar lisensi *Open Data* sesuai regulasi operasional pemerintah:

| Komponen Data | Lisensi | Peran dalam Arsitektur |
|---------------|---------|------------------------|
| **OSMnx & Overpass API** | ODbL | Ekstraksi topologi graf jalan raya dan telemetri *Point of Interest* (Aset Kritis). |
| **BMKG Public API** | Public | Telemetri atmosfer & proksi *weather stressor* berbasis kode administrasi level-4. |
| **DEMNAS Proxy** | Public | Model elevasi digital (disimulasikan) untuk perhitungan ambang batas kegagalan gravitasi. |
| **OpenRouter LLM** | API | Orkestrasi instruksional untuk *Autonomous Policy Agent* (Gemma/Mistral/Qwen). |

## Active Deployment Zones (DKI Jakarta)

RESILIA telah diinisialisasi untuk memantau 10 wilayah kritis logistik kota:
`Kemayoran`, `Penjaringan`, `Cengkareng`, `Jatinegara`, `Pulo Gadung`, `Kebayoran Baru`, `Cilincing`, `Kelapa Gading`, `Grogol Petamburan`, dan `Mampang Prapatan`.

## Local Deployment Instructions

### 1. Prerequisites
Pastikan Python 3.10+ telah terinstal. Eksekusi instalasi pustaka infrastruktur:
```bash
pip install -r requirements.txt
2. Environment Variables (B2G Cloud & LLM)
Untuk mengaktifkan fitur Autonomous Policy Agent dan Azure Cloud Sync, buat berkas .env di direktori utama (atau konfigurasikan secrets di host Anda):
OPENROUTER_API_KEY="your_openrouter_api_key"
AZURE_STORAGE_CONNECTION_STRING="your_azure_connection_string"
(Catatan: Jika koneksi Azure tidak ditemukan, sistem akan berjalan secara mandiri dalam Offline Mode berkat arsitektur Graceful Degradation).
3. Initialize Dashboard
streamlit run app.py

--------------------------------------------------------------------------------
Developed for Dicoding AI Impact Challenge 2026. Transforming static data into urban resilience.
