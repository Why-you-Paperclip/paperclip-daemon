"""
HTTP client for the Paperclip API.
Used by both the CLI commands and the daemon worker loop.
"""
import os
import httpx
from typing import Optional
from paperclip import config


def _client(token: Optional[str] = None) -> httpx.Client:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=config.api_url(), headers=headers, timeout=30)


def authenticate_device(provider_token: str, device_name: Optional[str] = None, gpu_model: Optional[str] = None) -> dict:
    """POST /provider/authenticate — register this machine."""
    with _client() as c:
        r = c.post("/provider/authenticate", json={
            "provider_token": provider_token,
            "device_name": device_name,
            "gpu_model": gpu_model,
        })
        r.raise_for_status()
        return r.json()


def set_allocation(device_token: str, allocation_pct: int) -> dict:
    """PUT /provider/allocate — update GPU allocation."""
    with _client(device_token) as c:
        r = c.put("/provider/allocate", json={"allocation_pct": allocation_pct})
        r.raise_for_status()
        return r.json()


def send_heartbeat(device_token: str, gpu_model: Optional[str] = None) -> dict:
    """POST /provider/heartbeat — signal device is alive."""
    with _client(device_token) as c:
        r = c.post("/provider/heartbeat", json={"gpu_model": gpu_model})
        r.raise_for_status()
        return r.json()


def get_next_job(device_token: str) -> Optional[dict]:
    """GET /provider/jobs/next — poll for an assigned job."""
    with _client(device_token) as c:
        r = c.get("/provider/jobs/next")
        r.raise_for_status()
        data = r.json()
        return data  # None or job dict


def download_job_files(device_token: str, job_id: str, dest_dir: str) -> list[str]:
    """Download all uploaded dataset files for a job into dest_dir. Returns list of local paths."""
    os.makedirs(dest_dir, exist_ok=True)
    # Use a longer timeout — audio files can be large
    with httpx.Client(base_url=config.api_url(), headers={"Authorization": f"Bearer {device_token}"}, timeout=120) as c:
        r = c.get(f"/provider/jobs/{job_id}/files")
        r.raise_for_status()
        filenames = r.json().get("files", [])

        downloaded = []
        for filename in filenames:
            r = c.get(f"/provider/jobs/{job_id}/file/{filename}")
            r.raise_for_status()
            dest = os.path.join(dest_dir, filename)
            with open(dest, "wb") as f:
                f.write(r.content)
            downloaded.append(dest)
        return downloaded


def report_progress(device_token: str, job_id: str, progress: int, status: str, error_msg: Optional[str] = None) -> dict:
    """POST /provider/jobs/{id}/progress — stream progress back."""
    with _client(device_token) as c:
        r = c.post(f"/provider/jobs/{job_id}/progress", json={
            "progress": progress,
            "status": status,
            "error_msg": error_msg,
        })
        r.raise_for_status()
        return r.json()
