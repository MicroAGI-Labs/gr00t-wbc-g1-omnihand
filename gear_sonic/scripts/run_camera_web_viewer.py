"""Serve the existing ZMQ camera stream as a small browser-based MJPEG UI.

The HTTP server binds to loopback by default. Access it from another computer
through an SSH local forward instead of exposing the viewer on the network::

    ssh -N -L 8080:127.0.0.1:8080 unitree@ROBOT_HOST

Then open http://127.0.0.1:8080 in a local browser.
"""

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import signal
import threading
import time

import cv2
import numpy as np
import tyro
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorClient

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SONIC Simulation</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #11151a; color: #edf2f7; }
    header { display: flex; align-items: center; gap: 12px; padding: 14px 18px;
             background: #1b222b; border-bottom: 1px solid #303a46; }
    h1 { margin: 0; font-size: 17px; font-weight: 600; }
    #camera-status { margin-left: auto; font-size: 13px; color: #f6c85f; }
    main { min-height: calc(100vh - 58px); display: flex; flex-direction: column;
           align-items: center; justify-content: center; gap: 14px;
           padding: 18px; box-sizing: border-box; }
    img { display: block; max-width: 100%; max-height: calc(100vh - 175px);
          border-radius: 6px; background: #090b0e; box-shadow: 0 8px 30px #0008; }
    .recorder { width: min(1100px, 100%); display: flex; align-items: center;
                gap: 12px; padding: 12px 14px; box-sizing: border-box;
                background: #1b222b; border: 1px solid #303a46; border-radius: 8px; }
    #record-state { min-width: 96px; padding: 6px 10px; text-align: center;
                    font-weight: 700; border-radius: 999px; background: #303a46; }
    #record-state.recording { background: #b4232f; color: white; animation: pulse 1.2s infinite; }
    #record-state.saving { background: #9a6700; color: white; }
    @keyframes pulse { 50% { opacity: .65; } }
    .recorder-info { min-width: 0; flex: 1; }
    #record-message { font-size: 14px; }
    #record-detail { margin-top: 3px; color: #9eabb8; font-size: 12px;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    button { border: 0; border-radius: 6px; padding: 10px 15px; color: white;
             font-weight: 650; cursor: pointer; background: #287a4d; }
    button.stop { background: #b4232f; }
    button.discard { background: #59636f; }
    button:disabled { cursor: not-allowed; opacity: .4; }
  </style>
</head>
<body>
  <header><h1>SONIC · G1 OmniHand simulation</h1><span id="camera-status">connecting…</span></header>
  <main>
    <img id="stream" src="/stream.mjpg" alt="Waiting for simulation camera stream">
    <section class="recorder">
      <span id="record-state">CONNECTING</span>
      <div class="recorder-info">
        <div id="record-message">Waiting for recorder…</div>
        <div id="record-detail"></div>
      </div>
      <button id="record-toggle" disabled>Start Recording</button>
      <button id="record-discard" class="discard" disabled>Discard</button>
    </section>
  </main>
  <script>
    const cameraStatus = document.getElementById('camera-status');
    const recordState = document.getElementById('record-state');
    const recordMessage = document.getElementById('record-message');
    const recordDetail = document.getElementById('record-detail');
    const recordToggle = document.getElementById('record-toggle');
    const recordDiscard = document.getElementById('record-discard');
    let commandPending = false;

    async function updateCameraStatus() {
      try {
        const response = await fetch('/healthz', {cache: 'no-store'});
        const health = await response.json();
        cameraStatus.textContent = health.streaming
          ? `${health.camera_count} camera${health.camera_count === 1 ? '' : 's'} · live`
          : 'waiting for simulator…';
        cameraStatus.style.color = health.streaming ? '#72d69c' : '#f6c85f';
      } catch (_) {
        cameraStatus.textContent = 'viewer disconnected';
        cameraStatus.style.color = '#ef7777';
      }
    }

    async function updateRecorderStatus() {
      try {
        const response = await fetch('/recording/status', {cache: 'no-store'});
        const status = await response.json();
        if (!status.connected) {
          recordState.textContent = 'OFFLINE';
          recordState.className = '';
          recordMessage.textContent = 'Recorder unavailable';
          recordDetail.textContent = '';
          recordToggle.disabled = true;
          recordDiscard.disabled = true;
          return;
        }
        const state = status.recording ? 'RECORDING' : (status.saving ? 'SAVING' : 'IDLE');
        recordState.textContent = state;
        recordState.className = status.recording ? 'recording' : (status.saving ? 'saving' : '');
        recordMessage.textContent = status.message || state;
        const sources = status.sources || {};
        const ready = sources.proprio && sources.camera && sources.hands;
        recordDetail.textContent = `episode ${status.episode_index} · ${status.frame_count} frames · ${status.dataset_root} · ${ready ? 'sources ready' : 'source missing'} · headset: release X+B to record/save, release Y+A to discard`;
        recordToggle.textContent = status.recording ? 'Stop & Save' : 'Start Recording';
        recordToggle.className = status.recording ? 'stop' : '';
        recordToggle.disabled = commandPending || status.saving || (!status.recording && !ready);
        recordDiscard.disabled = commandPending || !status.recording;
      } catch (_) {
        recordMessage.textContent = 'Recorder status request failed';
      }
    }

    async function sendRecorderCommand(path) {
      commandPending = true;
      recordToggle.disabled = true;
      recordDiscard.disabled = true;
      try {
        await fetch(path, {method: 'POST'});
      } finally {
        setTimeout(() => { commandPending = false; updateRecorderStatus(); }, 350);
      }
    }

    recordToggle.addEventListener('click', () => sendRecorderCommand('/recording/toggle'));
    recordDiscard.addEventListener('click', () => sendRecorderCommand('/recording/discard'));
    updateCameraStatus(); updateRecorderStatus();
    setInterval(updateCameraStatus, 1500);
    setInterval(updateRecorderStatus, 500);
  </script>
</body>
</html>
""".encode("utf-8")


@dataclass
class CameraWebViewerConfig:
    """Configuration for the browser camera viewer."""

    camera_host: str = "localhost"
    """ZMQ camera publisher hostname."""

    camera_port: int = 5555
    """ZMQ camera publisher port."""

    http_host: str = "127.0.0.1"
    """HTTP bind address. Keep loopback when using an SSH forward."""

    http_port: int = 8080
    """HTTP port forwarded to the user's computer."""

    fps: int = 20
    """Maximum browser stream frame rate."""

    jpeg_quality: int = 80
    """JPEG quality used for the browser stream."""

    max_tile_width: int = 960
    """Maximum width of each camera tile."""

    recording_command_port: int = 5580
    """ZMQ PUB port used to send recorder commands."""

    recording_status_host: str = "localhost"
    """Host publishing authoritative recorder status."""

    recording_status_port: int = 5581
    """ZMQ SUB port used to receive recorder status."""


class CameraFrameHub:
    """Receive camera messages once and fan the latest JPEG out to browsers."""

    def __init__(self, config: CameraWebViewerConfig):
        self.config = config
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._camera_count = 0
        self._last_frame_time = 0.0
        self._running = True
        self._client = SensorClient()
        self._client.start_client(config.camera_host, config.camera_port)
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._client.stop_client()
        with self._condition:
            self._condition.notify_all()

    def health(self) -> dict[str, object]:
        with self._condition:
            age = time.monotonic() - self._last_frame_time if self._last_frame_time else None
            return {
                "streaming": age is not None and age < 2.0,
                "camera_count": self._camera_count,
                "last_frame_age_s": round(age, 3) if age is not None else None,
            }

    def wait_for_jpeg(self, sequence: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != sequence or not self._running,
                timeout=timeout,
            )
            return self._sequence, self._jpeg

    def _receive_loop(self) -> None:
        frame_period = 1.0 / max(self.config.fps, 1)
        next_frame_time = 0.0
        while self._running:
            message = self._client.receive_message_nonblocking(timeout_ms=200)
            if message is None:
                continue

            now = time.monotonic()
            if now < next_frame_time:
                continue

            images = ImageMessageSchema.deserialize(message).images
            jpeg = compose_camera_jpeg(
                images,
                max_tile_width=self.config.max_tile_width,
                jpeg_quality=self.config.jpeg_quality,
            )
            if jpeg is None:
                continue

            next_frame_time = now + frame_period
            with self._condition:
                self._jpeg = jpeg
                self._sequence += 1
                self._camera_count = len(images)
                self._last_frame_time = now
                self._condition.notify_all()


class RecorderControlHub:
    """Bridge HTTP requests to the recorder and cache its authoritative status."""

    def __init__(self, config: CameraWebViewerConfig):
        self.config = config
        self._commands: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._status: dict[str, object] | None = None
        self._status_received_at = 0.0
        self._running = True
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def send(self, command: str) -> None:
        if command not in {"c", "x"}:
            raise ValueError(f"unsupported recorder command: {command}")
        self._commands.put(command)

    def status(self) -> dict[str, object]:
        with self._lock:
            payload = dict(self._status or {})
            age = time.monotonic() - self._status_received_at if self._status else None
        payload["connected"] = age is not None and age < 2.0
        payload["last_status_age_s"] = round(age, 3) if age is not None else None
        return payload

    def _run(self) -> None:
        context = zmq.Context()
        command_socket = context.socket(zmq.PUB)
        command_socket.setsockopt(zmq.SNDHWM, 10)
        command_socket.bind(f"tcp://*:{self.config.recording_command_port}")
        status_socket = context.socket(zmq.SUB)
        status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        status_socket.setsockopt(zmq.CONFLATE, 1)
        status_socket.connect(
            f"tcp://{self.config.recording_status_host}:{self.config.recording_status_port}"
        )
        poller = zmq.Poller()
        poller.register(status_socket, zmq.POLLIN)
        self._ready.set()
        try:
            while self._running:
                try:
                    while True:
                        command_socket.send_string(self._commands.get_nowait())
                except queue.Empty:
                    pass
                if status_socket in dict(poller.poll(50)):
                    payload = status_socket.recv_json()
                    with self._lock:
                        self._status = payload
                        self._status_received_at = time.monotonic()
        finally:
            command_socket.close(linger=0)
            status_socket.close(linger=0)
            context.term()


def compose_camera_jpeg(images: dict[str, np.ndarray], max_tile_width: int, jpeg_quality: int) -> bytes | None:
    """Label and horizontally tile camera frames into one browser-ready JPEG."""
    tiles: list[np.ndarray] = []
    for name in sorted(images):
        image = images[name]
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            continue

        if image.ndim == 2:
            tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            tile = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Camera clients expose RGB; OpenCV's encoder expects BGR.
            tile = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            continue

        height, width = tile.shape[:2]
        if width > max_tile_width:
            scale = max_tile_width / width
            tile = cv2.resize(tile, (max_tile_width, max(1, int(height * scale))))

        cv2.putText(
            tile,
            name,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 230, 140),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return None

    max_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        if tile.shape[0] < max_height:
            tile = cv2.copyMakeBorder(
                tile,
                0,
                max_height - tile.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(10, 12, 15),
            )
        padded_tiles.append(tile)

    canvas = cv2.hconcat(padded_tiles)
    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    return encoded.tobytes() if ok else None


def make_handler(
    frame_hub: CameraFrameHub, recorder_hub: RecorderControlHub
) -> type[BaseHTTPRequestHandler]:
    class CameraWebHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes("text/html; charset=utf-8", _INDEX_HTML)
            elif path == "/healthz":
                payload = json.dumps(frame_hub.health()).encode("utf-8")
                self._send_bytes("application/json", payload)
            elif path == "/recording/status":
                payload = json.dumps(recorder_hub.status()).encode("utf-8")
                self._send_bytes("application/json", payload)
            elif path == "/snapshot.jpg":
                _, jpeg = frame_hub.wait_for_jpeg(-1, timeout=2.0)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No camera frame yet")
                else:
                    self._send_bytes("image/jpeg", jpeg)
            elif path == "/stream.mjpg":
                self._stream_mjpeg()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/recording/toggle":
                recorder_hub.send("c")
            elif path == "/recording/discard":
                recorder_hub.send("x")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes("application/json", b'{"accepted":true}')

        def _send_bytes(self, content_type: str, payload: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream_mjpeg(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while True:
                    sequence, jpeg = frame_hub.wait_for_jpeg(sequence)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    return CameraWebHandler


def main(config: CameraWebViewerConfig) -> None:
    if config.camera_port == config.http_port and config.camera_host in {"localhost", "127.0.0.1"}:
        raise ValueError("--camera-port and --http-port must be different")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    frame_hub = CameraFrameHub(config)
    recorder_hub = RecorderControlHub(config)
    server = ThreadingHTTPServer(
        (config.http_host, config.http_port), make_handler(frame_hub, recorder_hub)
    )
    server.daemon_threads = True

    def stop_server(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    frame_hub.start()
    recorder_hub.start()
    print(
        f"SONIC browser viewer listening on http://{config.http_host}:{config.http_port}\n"
        "Use an SSH local port forward when viewing from another computer."
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        frame_hub.close()
        recorder_hub.close()


if __name__ == "__main__":
    main(tyro.cli(CameraWebViewerConfig))
