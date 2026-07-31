#!/usr/bin/env python3
"""client_postprocess.py — No-op postprocessing step."""
import os

# Match the ml-inference reference: mute submission-stage output while leaving
# the parent harness output visible.
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)

print("[client_postprocess] No-op.", flush=True)
