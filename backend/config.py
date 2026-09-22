"""
Central configuration and runtime limits (see ROADMAP.md P2.1).

Values come from environment variables / a project-root .env file, else the safe defaults below.
Import as:  from config import settings
"""
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server / CORS (single-user local demo: only localhost is allowed) ---
    CORS_ORIGINS: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # --- Upload guards (P2.3) ---
    MAX_UPLOAD_MB: int = 512
    ALLOWED_VIDEO_EXT: List[str] = [".mp4", ".mov", ".avi", ".mkv"]

    # --- Pipeline CPU budget (P3.1): cap frames written to disk on long/4K clips ---
    MAX_FRAMES_ON_DISK: int = 1500

    # --- Models (CPU) ---
    # Depth Anything V2 Small is Apache-2.0; the Large variant is non-commercial (do not ship it).
    DEPTH_MODEL: str = "depth-anything/Depth-Anything-V2-Small-hf"
    YOLO_MODEL: str = "yolov8n-seg.pt"       # segmentation = pixel masks (tighter than boxes)
    MASK_DYNAMIC: bool = True                # mask people/vehicles/animals before SfM + fusion

    # --- Data layer: SQLite by default; flip USE_SUPABASE=true once keys are set (P1) ---
    USE_SUPABASE: bool = False
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    @property
    def MAX_UPLOAD_BYTES(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
