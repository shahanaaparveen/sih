"""Aero3D backend services. Only the live single-pass pipeline is exported here."""
from .video_processor import get_video_metadata, process_video_pipeline

__all__ = [
    "get_video_metadata",
    "process_video_pipeline",
]
