import io
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from photo_enhancer.pipeline import encode_image  # noqa: E402
from photo_enhancer.server import app  # noqa: E402

client = TestClient(app)


def test_index_and_info():
    assert "Photo Enhancer" in client.get("/").text
    info = client.get("/api/info").json()
    assert {m["key"] for m in info["models"]} >= {"general-x4", "realesrgan-x4"}


def test_job_roundtrip(photo):
    files = {"file": ("pic.png", encode_image(photo, None, "PNG"), "image/png")}
    data = {"model": "classic", "scale": "2", "contrast": "0.3", "format": "jpg"}
    job = client.post("/api/jobs", files=files, data=data).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert (job["width"], job["height"]) == (128, 96)
    res = client.get(f"/api/jobs/{job['id']}/result")
    assert res.headers["content-type"] == "image/jpeg"
    assert 'filename="pic_enhanced.jpg"' in res.headers["content-disposition"]


def test_bad_options_rejected(photo):
    files = {"file": ("pic.png", encode_image(photo, None, "PNG"), "image/png")}
    assert client.post("/api/jobs", files=files, data={"scale": "50"}).status_code == 400
    assert client.post("/api/jobs", files=files, data={"format": "gif"}).status_code == 400


def test_unknown_job():
    assert client.get("/api/jobs/nope").status_code == 404


def test_metadata_option(tmp_path):
    from PIL import Image

    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0)}
    src = io.BytesIO()
    Image.new("RGB", (32, 24)).save(src, "JPEG", exif=exif)
    files = {"file": ("cam.jpg", src.getvalue(), "image/jpeg")}

    assert client.post("/api/jobs", files=files, data={"metadata": "everything"}).status_code == 400
    job = client.post("/api/jobs", files=files, data={"model": "none", "scale": "1", "format": "jpg",
                                                      "metadata": "no-gps"}).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    out = Image.open(io.BytesIO(client.get(f"/api/jobs/{job['id']}/result").content)).getexif()
    assert out[0x010F] == "Canon" and not out.get_ifd(0x8825)
