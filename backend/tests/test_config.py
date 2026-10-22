from aivhuman_api.config import settings


def test_model_files_present() -> None:
    for name in ("model.pt", "calibrator.json", "evaluation.json"):
        assert (settings.model_dir / name).is_file()
