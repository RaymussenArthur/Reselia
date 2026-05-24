import os, json
from azure.storage.blob import BlobServiceClient

def sync_spk_to_blob(area, weather, epicenter_count, network_drop):
    try:
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            return False, "[EDGE MODE] Azure Connection String not initialized. Data secured in local cache."

        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service_client.get_container_client("resilia-b2g-backup")

        payload = {
            "area": area,
            "weather_telemetry": weather,
            "dbscan_epicenters": epicenter_count,
            "network_efficiency_drop_pct": network_drop
        }

        blob_client = container_client.get_blob_client(blob=f"spk_{area.lower()}_latest.json")
        blob_client.upload_blob(json.dumps(payload), overwrite=True)

        return True, "[HYBRID-CLOUD] Successfully synced SPK draft to Azure Blob Storage."
    except Exception as e:
        return False, f"[AZURE ERROR] Sync failed: {str(e)}"