"""Unit tests for the torch-CUDA-DLL registration helper in backend.face_detect.

Not a GPU test -- these run everywhere, including CI with no GPU. They check
the registration logic (idempotent, never raises, only touches Windows) rather
than whether CUDAExecutionProvider actually loads, which depends on the host.

Context: onnxruntime-gpu's CUDA provider failed to load
("cublasLt64_12.dll ... module could not be found") even with a working GPU
and torch.cuda.is_available() True, because torch bundles its own CUDA 12
runtime in torch/lib and never exposes it on PATH. Confirmed fix, and confirmed
identical detection output (48/48 faces, dataset/img01.jpg) CPU vs GPU.
"""

from __future__ import annotations

import sys

from backend import face_detect


def _reset() -> None:
    face_detect._cuda_dll_dir_registered = False


def test_registers_only_once(monkeypatch) -> None:
    """The guard flag must make a second call a no-op.

    Not asserted as an exact call count: importing torch itself calls
    os.add_dll_directory for its own DLLs on Windows, and the patch below
    catches that too. What matters is that OUR code does not add a second
    call on the repeat invocation.
    """
    _reset()
    calls = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "os.add_dll_directory", lambda p: calls.append(p), raising=False
    )
    monkeypatch.setattr("os.path.isdir", lambda p: True)

    face_detect._register_torch_cuda_dlls()
    assert face_detect._cuda_dll_dir_registered is True
    after_first = len(calls)

    face_detect._register_torch_cuda_dlls()
    assert len(calls) == after_first, "second call must add no further registration"


def test_no_op_on_non_windows(monkeypatch) -> None:
    _reset()
    monkeypatch.setattr(sys, "platform", "linux")
    called = []
    monkeypatch.setattr(
        "os.add_dll_directory", lambda p: called.append(p), raising=False
    )

    face_detect._register_torch_cuda_dlls()

    assert called == []


def test_never_raises_when_torch_is_unavailable(monkeypatch) -> None:
    """A face detector must still be constructible without CUDA acceleration
    if torch's DLL directory cannot be found -- this can only fail to speed
    things up, never break face detection outright."""
    _reset()
    monkeypatch.setattr(sys, "platform", "win32")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    face_detect._register_torch_cuda_dlls()  # must not raise


def test_never_raises_when_dll_directory_missing(monkeypatch) -> None:
    _reset()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("os.path.isdir", lambda p: False)
    called = []
    monkeypatch.setattr(
        "os.add_dll_directory", lambda p: called.append(p), raising=False
    )

    face_detect._register_torch_cuda_dlls()  # must not raise

    assert called == [], "must not register a directory that does not exist"
