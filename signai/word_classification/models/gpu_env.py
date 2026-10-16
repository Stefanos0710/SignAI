"""Call ensure_gpu_env() as the very first thing a script does, before importing tensorflow --
sets up the GPU environment this particular workstation needs for TF's Conv3D to work, then
re-execs the process so the dynamic linker picks up LD_LIBRARY_PATH (unlike most env vars,
that one can't be applied to an already-running process for a *new* dlopen call -- the loader
reads it once, at process start).

Two separate, unrelated problems on this machine (see the project-tf-gpu-libdevice memory note):
1. XLA (elementwise GPU ops: exp, pow, ...) needs an nvvm/libdevice/ directory layout; the
   system's CUDA toolkit package ships libdevice.10.bc but not in that shape.
2. TF 2.16's compiled binary needs cuDNN 8.9.7.29 specifically for Conv3D/Conv2D ("No DNN in
   stream executor" otherwise) -- but torch (used by hand_stream.py/face_stream.py's DINOv3
   encoder, always a separate process from training) needs cuDNN 9.x, so the venv's pip-installed
   nvidia-cudnn-cu12 has to stay on 9.x. TF instead gets its own cuDNN 8 copy via LD_LIBRARY_PATH
   (extracted from the 8.9.7.29 wheel, not pip-installed), so the two never conflict.
"""
import os
import sys

CUDA_XLA_DIR = "/loctmp/zzm01651/cuda-xla"
CUDNN8_DIR = "/loctmp/zzm01651/cudnn8-for-tf"
_REEXEC_MARKER = "_SIGNAI_GPU_ENV_SET"


def ensure_gpu_env():
    """No-op if already applied, already re-exec'd, or this workstation's fix directories don't
    exist (e.g. a different machine, or TF/cuDNN get fixed upstream) -- never forces paths that
    don't exist. Otherwise re-execs the current process with the environment TF needs.
    """
    if os.environ.get(_REEXEC_MARKER) == "1":
        return
    if not (os.path.isdir(CUDA_XLA_DIR) and os.path.isdir(CUDNN8_DIR)):
        return

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CUDNN8_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={CUDA_XLA_DIR}")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env[_REEXEC_MARKER] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
