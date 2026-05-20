
# SURAT PERINTAH KERJA (SPK)
## Sistem Peringatan Dini Banjir - RESILIA v2.0

---

**Nomor SPK :** SPK/BPBD/202605191403/LOW
**Tanggal   :** 19 May 2026, 14:03 WIB
**Instansi  :** Badan Penanggulangan Bencana Daerah (BPBD) DKI Jakarta
**Wilayah   :** Kelurahan Kemayoran, Jakarta Utara
**Run ID    :** `20260519_140130`

---

### I. KONDISI TERKINI (AI-Generated Risk Assessment)

| Parameter | Nilai |
|-----------|-------|
| Kondisi Cuaca (BMKG) | **Cerah** |
| Model AI | GAT Phase 2 (F1=0.8557, AUC=0.9331) |
| Probabilitas Kegagalan Sistem | **2.85%** |
| Tier Risiko | **LOW** |
| Node Kritis Teridentifikasi | 1319 dari 3389 node (38.9%) |
| Skor Resiliensi Jaringan | 0.534 |

### II. EPISENTER PRIORITAS (DBSCAN Triage)

Berdasarkan analisis klaster spasial DBSCAN, ditemukan **3 episenter** dengan urutan prioritas sebagai berikut:


**1. EP-1**
- Koordinat Pusat: (-6.16234, 106.85859)
- Skor Triage Komposit: **0.7995**
- Jumlah Node: 1306
- Skor Risiko Rata-rata: 0.899
- Kritikalitas POI: 0.5067


**2. EP-3**
- Koordinat Pusat: (-6.1628, 106.87558)
- Skor Triage Komposit: **0.4892**
- Jumlah Node: 6
- Skor Risiko Rata-rata: 0.8557
- Kritikalitas POI: 0.2016


**3. EP-2**
- Koordinat Pusat: (-6.17688, 106.879)
- Skor Triage Komposit: **0.489**
- Jumlah Node: 4
- Skor Risiko Rata-rata: 0.9767
- Kritikalitas POI: 0.0



### III. PERINTAH KERJA

Berdasarkan asesmen AI di atas, dengan ini diperintahkan kepada Tim Lapangan BPBD:

1. **Prioritas Utama:** Kerahkan armada pompa darurat ke episenter EP-1 dalam **waktu 2 jam** sejak diterbitkannya SPK ini.
2. **Koordinasi:** Hubungi 257 fasilitas kritis yang teridentifikasi (rumah sakit, sekolah, pasar) untuk prosedur evakuasi standby.
3. **Pemantauan:** Perbarui data BMKG setiap 15 menit; laporkan perubahan tier risiko ke Posko Utama.
4. **Dokumentasi:** Semua tindakan lapangan dicatat dengan referensi Run ID `20260519_140130` untuk audit.

### IV. DASAR HUKUM

- UU No. 24 Tahun 2007 tentang Penanggulangan Bencana
- Perka BNPB No. 3 Tahun 2016 tentang Sistem Komando Penanganan Darurat Bencana
- Peraturan Gubernur DKI Jakarta No. 97 Tahun 2012 tentang Penanggulangan Banjir

---

*Dokumen ini dihasilkan secara otomatis oleh RESILIA v2.0 Phase 2 AI Engine.*
*Memerlukan validasi dan penandatanganan oleh Kepala BPBD sebelum dieksekusi.*
*Audit trail tersedia di file: `resilia_model_metadata.json` (Run ID: `20260519_140130`)*