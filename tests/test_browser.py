"""Real browser checks, opt in after: playwright install chromium.

CI runs these; local sandboxes without a browser must not claim a browser pass.
Requests are routed to TestClient: real HTML/JS/rendering, no open test server.
"""

import os
import subprocess
import re
import socket
import threading
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from backend.app.main import app
from backend.app.config import settings
from backend.app.db.database import SessionLocal
from backend.app.db.models import User
from backend.app.services.auth import hash_password
from tests.test_application import image

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BROWSER_TESTS") != "1",
    reason="Requires installed Chromium; opt in with RUN_BROWSER_TESTS=1",
)


@pytest.fixture
def browser_page(client, monkeypatch):
    """Real browser -> real HTTP -> ASGI/API -> DB/media (no request-body mocking)."""
    import uvicorn
    from playwright.sync_api import sync_playwright

    monkeypatch.setattr(settings, "registration_enabled", False)
    ready = threading.Event()

    class TestServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    listener = socket.socket()
    listener.bind(("0.0.0.0", 0))
    base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    # The client fixture already owns lifespan/isolated database initialization.
    server = TestServer(
        uvicorn.Config(app, lifespan="off", access_log=False, log_level="error")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        assert ready.wait(10), "Test HTTP server did not start"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=os.getenv("CHROMIUM_EXECUTABLE") or None,
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                ],
            )
            context = browser.new_context(accept_downloads=True, base_url=base_url)

            def route_request(route):
                url = urlsplit(route.request.url)
                if url.hostname != "127.0.0.1":
                    route.abort()  # Optional fonts should not affect an offline test.
                    return
                if os.getenv("UI_REPRO_REF") and url.path == "/app":
                    html = subprocess.check_output(
                        [
                            "git",
                            "show",
                            f"{os.environ['UI_REPRO_REF']}:frontend/app.html",
                        ]
                    )
                    route.fulfill(status=200, content_type="text/html", body=html)
                    return
                if os.getenv("UI_REPRO_REF") and url.path.startswith("/api/media/"):
                    # Legacy HTML used public media. Remove only its timestamp suffix
                    # in this diagnostic, so it doesn't invalidate the new signature.
                    route.continue_(url=route.request.url.split("?t=", 1)[0])
                    return
                route.continue_()

            context.route("**/*", route_request)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            try:
                yield page
                assert not errors, errors
            finally:
                context.close()
                browser.close()
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "Test HTTP server did not stop"


def test_browser_login_guard(browser_page):
    from playwright.sync_api import expect

    page = browser_page
    page.goto("/app")
    expect(page).to_have_url(re.compile(r"/login$"))
    expect(page.locator("#registrationControls")).to_be_hidden()
    expect(page.locator("#demoCredentials")).to_be_hidden()


def test_browser_officer_image_and_pdf(browser_page, detector):
    from playwright.sync_api import expect

    page = browser_page
    email = f"browser-{uuid4().hex}@example.com"
    with SessionLocal() as db:
        db.add(
            User(
                email=email,
                full_name="Browser Test",
                role="officer",
                hashed_password=hash_password("browser-test-password"),
            )
        )
        db.commit()
    page.goto("/login")
    page.locator("#email").fill(email)
    page.locator("#password").fill("browser-test-password")
    page.locator("#submitBtn").click()
    expect(page).to_have_url(re.compile(r"/app$"))
    filename, data, mime = image()
    page.locator("#fileInput").set_input_files(
        {"name": filename, "mimeType": mime, "buffer": data.getvalue()}
    )
    page.locator("#createTickets").check()
    page.locator("#runBtn").click()
    expect(page.locator("#sObjects")).to_have_text("2")
    expect(page.locator("#resultImg")).to_be_visible()
    page.wait_for_function("document.querySelector('#resultImg').naturalWidth > 0")
    page.goto("/violations")
    page.locator("#violBody tr[data-id]").first.click()
    with page.expect_download() as download:
        page.locator("#pdfBtn").click()
    assert download.value.suggested_filename.endswith(".pdf")


def login_officer(page):
    from playwright.sync_api import expect

    email = f"browser-{uuid4().hex}@example.com"
    with SessionLocal() as db:
        db.add(
            User(
                email=email,
                full_name="Browser Test",
                role="officer",
                hashed_password=hash_password("browser-test-password"),
            )
        )
        db.commit()
    page.goto("/login")
    page.locator("#email").fill(email)
    page.locator("#password").fill("browser-test-password")
    page.locator("#submitBtn").click()
    expect(page).to_have_url(re.compile(r"/app$"))


def upload_image(page):
    filename, data, mime = image()
    page.locator("#fileInput").set_input_files(
        {"name": filename, "mimeType": mime, "buffer": data.getvalue()}
    )
    page.locator("#runBtn").click()
    page.wait_for_function(
        "document.querySelector('#resultImg').naturalWidth > 0 && !document.querySelector('#resultImg').classList.contains('hidden')"
    )


def test_browser_video_error_preserves_annotation(browser_page, detector):
    from playwright.sync_api import expect

    page = browser_page
    login_officer(page)
    page.locator("#createTickets").uncheck()
    upload_image(page)
    # Reproduce the precise old shared error-handler failure without depending on codecs.
    page.evaluate(
        "document.querySelector('#resultVid').dispatchEvent(new Event('error'))"
    )
    expect(page.locator("#resultImg")).to_be_visible()
    expect(page.locator("#detList")).to_contain_text("NoHelmet")
    expect(page.locator("#sObjects")).to_have_text("2")


def test_browser_missing_model_message(browser_page):
    from playwright.sync_api import expect

    page = browser_page
    login_officer(page)
    expect(page.locator("#modelName")).to_have_text("Custom YOLOv5 unavailable")
    expect(page.locator("#modelNote")).to_contain_text("Supply trusted models/best.pt")
    filename, data, mime = image()
    page.locator("#fileInput").set_input_files(
        {"name": filename, "mimeType": mime, "buffer": data.getvalue()}
    )
    expect(page.locator("#runBtn")).to_be_disabled()


def test_browser_failed_second_request_clears_stale_result(
    browser_page, detector, monkeypatch
):
    from playwright.sync_api import expect

    page = browser_page
    login_officer(page)
    upload_image(page)

    def fail(*args, **kwargs):
        raise ValueError("Corrupt input")

    monkeypatch.setattr(detector, "process_image_file", fail)
    filename, data, mime = image()
    page.locator("#fileInput").set_input_files(
        {"name": filename, "mimeType": mime, "buffer": data.getvalue()}
    )
    page.locator("#runBtn").click()
    expect(page.locator("#resultEmpty")).to_contain_text(
        "Detection failed: Invalid image"
    )
    expect(page.locator("#resultImg")).to_be_hidden()
    expect(page.locator("#summaryBox")).to_be_hidden()
    expect(page.locator("#detList")).not_to_contain_text("NoHelmet")


def test_browser_refresh_recovers_preview_without_rerun(browser_page, detector):
    from playwright.sync_api import expect

    page = browser_page
    login_officer(page)
    upload_image(page)
    job_id = page.locator("#sJob").inner_text()
    page.evaluate("document.querySelector('#resultImg').src = '/missing-preview.jpg'")
    expect(page.locator("#mediaNotice")).to_contain_text("could not be fetched")
    page.locator("#refreshResult").click()
    expect(page.locator("#resultImg")).to_be_visible()
    page.wait_for_function("document.querySelector('#resultImg').naturalWidth > 0")
    expect(page.locator("#sJob")).to_have_text(job_id)
    expect(page.locator("#mediaNotice")).to_be_hidden()


def test_browser_actual_yolov5_image_flow(browser_page, monkeypatch, tmp_path):
    """Real network inference + DOM + signed image, but RANDOM, UNTRAINED weights."""
    from playwright.sync_api import expect
    from tests.checkpoint_factory import create
    from backend.app.services import detector as runtime

    path = tmp_path / "browser-untrained-fixture.pt"
    create(path)
    monkeypatch.setattr(settings, "model_path", path)
    monkeypatch.setattr(runtime, "_detector", None)
    page = browser_page
    login_officer(page)
    expect(page.locator("#modelName")).to_have_text(
        "Custom YOLOv5 checkpoint loaded", timeout=30000
    )
    upload_image(page)
    expect(page.locator("#resultImg")).to_be_visible()
    expect(page.locator("#sFile")).to_have_text("traffic.jpg")
    assert page.locator("#resultImg").evaluate("img => img.naturalWidth") == 100


def test_browser_video_upload_and_refresh(browser_page, detector, tmp_path):
    import cv2
    import numpy as np
    from playwright.sync_api import expect

    source = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 80)
    )
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()
    page = browser_page
    login_officer(page)
    page.locator('.tab[data-tab="video"]').click()
    page.locator("#fileInput").set_input_files(str(source))
    page.locator("#runBtn").click()
    expect(page.locator("#resultImg")).to_be_visible()
    page.wait_for_function("document.querySelector('#resultImg').naturalWidth > 0")
    # Exercise actual browser decoding of the H.264 artifact, not just file existence.
    page.evaluate(
        "() => { const v = document.querySelector('#resultVid'); v.muted = true; return v.play(); }"
    )
    page.wait_for_function("document.querySelector('#resultVid').readyState >= 2")
    expect(page.locator("#mediaNotice")).to_be_hidden()
    job_id = page.locator("#sJob").inner_text()
    page.locator("#refreshResult").click()
    expect(page.locator("#resultImg")).to_be_visible()
    expect(page.locator("#sJob")).to_have_text(job_id)
    expect(page.locator("#downloadResult")).to_have_attribute(
        "href", re.compile(r".*\.mp4\?token=.*")
    )


def test_browser_webcam_capture(browser_page, detector):
    from playwright.sync_api import expect

    page = browser_page
    login_officer(page)
    page.goto("/live")
    page.locator("#startBtn").click()
    expect(page.locator("#snapBtn")).to_be_enabled()
    page.locator("#snapBtn").click()
    expect(page.locator("#overlay")).to_be_visible()
    page.wait_for_function("document.querySelector('#overlay').naturalWidth > 0")
    expect(page.locator("#sObj")).to_have_text("2")
    page.locator("#stopBtn").click()
    expect(page.locator("#snapBtn")).to_be_disabled()
