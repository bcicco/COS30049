"""Settings read from the environment or backend/.env."""

from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT: Final = Path(__file__).parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", env_prefix="AIVHUMAN_")

    cors_origins: list[str] = ["http://localhost:5173"]
    model_dir: Path = BACKEND_ROOT / "models" / "p5-adv"
    device: str = "cpu"  # "cuda" to run GPT-2 on the GPU
    db_path: Path = BACKEND_ROOT / "data" / "checks.db"
    # 100k chars of escaped emoji JSON is ~1.2 MB, so real requests stay well under
    max_body_bytes: int = 2_000_000

    # scoring jobs running at once, and how many may wait for a slot before 503s
    score_workers: int = 1
    score_queue: int = 4
    score_wait_seconds: float = 30.0


settings = Settings()
