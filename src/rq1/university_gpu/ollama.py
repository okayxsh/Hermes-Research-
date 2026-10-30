"""Isolated local Ollama server lifecycle; no installation, pull, or inference by default."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from rq1.university_gpu.manifest import MODEL_DIGEST, MODEL_QUANTIZATION, MODEL_TAG
from rq1.university_gpu.models import GPUDevice, WorkerSpec


class OllamaError(RuntimeError):
    pass


def worker_specs(devices: Sequence[GPUDevice], *, count: int, base_port: int, root: Path) -> list[WorkerSpec]:
    if count not in {1, 2, 3, 4} or count > len(devices):
        raise OllamaError("worker count must be 1-4 and cannot exceed the discovered GPU count")
    selected = sorted(devices, key=lambda device: device.uuid)[:count]
    ports = list(range(base_port, base_port + count))
    if len(set(ports)) != count or any(not 1024 <= port <= 65535 for port in ports):
        raise OllamaError("worker endpoint ports must be unique unprivileged TCP ports")
    return [WorkerSpec(f"gpu-worker-{index + 1}", device.uuid, f"http://127.0.0.1:{ports[index]}", ports[index], root / f"worker-{index + 1}")
            for index, device in enumerate(selected)]


def server_environment(spec: WorkerSpec, *, model_dir: Path) -> dict[str, str]:
    if not model_dir.is_dir():
        raise OllamaError(f"existing model directory is required: {model_dir}")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": spec.gpu_uuid,
        "OLLAMA_HOST": f"127.0.0.1:{spec.port}",
        "OLLAMA_MODELS": str(model_dir.resolve()),
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    return env


def _request(endpoint: str, path: str, payload: Mapping[str, Any] | None = None, *, timeout: float = 5) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(endpoint.rstrip("/") + path, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise OllamaError(f"Ollama returned a non-object at {path}")
    return value


def probe_endpoint(endpoint: str, *, expected_model: str = MODEL_TAG, expected_digest: str = MODEL_DIGEST,
                   expected_quantization: str | None = None) -> dict[str, Any]:
    try:
        version = _request(endpoint, "/api/version")
        tags = _request(endpoint, "/api/tags")
        models = tags.get("models")
        model = next((item for item in models if isinstance(item, dict) and item.get("name") == expected_model), None) if isinstance(models, list) else None
        if model is None:
            return {"status": "FAIL", "reason": "frozen model tag is absent; automatic pull/substitution is forbidden", "ollama_version": version.get("version")}
        digest = str(model.get("digest", ""))
        if digest != expected_digest:
            return {"status": "FAIL", "reason": "installed model digest differs from the frozen digest", "observed_digest": digest,
                    "expected_digest": expected_digest, "ollama_version": version.get("version")}
        details = model.get("details") if isinstance(model.get("details"), dict) else {}
        quantization = details.get("quantization_level")
        quantization_required = expected_quantization if expected_quantization is not None else (
            MODEL_QUANTIZATION if expected_model == MODEL_TAG and expected_digest == MODEL_DIGEST else None
        )
        if quantization_required is not None and quantization != quantization_required:
            return {"status": "FAIL", "reason": "installed model quantization differs from the frozen identity",
                    "observed_quantization": quantization, "expected_quantization": quantization_required,
                    "observed_digest": digest, "ollama_version": version.get("version")}
        return {"status": "PASS", "model": expected_model, "digest": digest, "quantization": quantization,
                "ollama_version": version.get("version"), "inference_run": False}
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED", "reason": f"Ollama endpoint unavailable: {type(exc).__name__}", "inference_run": False}


@dataclass
class OllamaServer:
    spec: WorkerSpec
    model_dir: Path
    binary: str = "ollama"
    startup_timeout: float = 30
    process: subprocess.Popen[Any] | None = None
    log_handle: Any = None

    def start(self, *, runner: Any = subprocess.Popen) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            raise OllamaError(f"server for {self.spec.worker_id} is already running")
        self.spec.worker_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.spec.worker_dir / "ollama-server.log").open("ab")
        env = server_environment(self.spec, model_dir=self.model_dir)
        self.process = runner([self.binary, "serve"], cwd=self.spec.worker_dir, env=env,
                              stdin=subprocess.DEVNULL, stdout=self.log_handle, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise OllamaError(f"Ollama server exited with code {self.process.returncode}; inspect {self.spec.worker_dir / 'ollama-server.log'}")
            try:
                _request(self.spec.endpoint, "/api/version", timeout=1)
                return {"worker_id": self.spec.worker_id, "pid": self.process.pid, "endpoint": self.spec.endpoint,
                        "gpu_uuid": self.spec.gpu_uuid, "status": "started", "model_dir": str(self.model_dir.resolve())}
            except Exception:
                time.sleep(0.2)
        self.stop()
        raise OllamaError(f"Ollama server did not become healthy within {self.startup_timeout}s")

    def stop(self, *, timeout: float = 10) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.process = None
