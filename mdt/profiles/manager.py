"""Profile lifecycle management."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import soundfile as sf

from mdt.profiles.schema import BackendStatus, ProfileMetadata

_DEFAULT_PROFILES_DIR = Path.home() / ".mdt" / "profiles"


def get_profiles_dir() -> Path:
    import os
    return Path(os.environ.get("MDT_PROFILES_DIR", str(_DEFAULT_PROFILES_DIR)))


def _audio_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def list_profiles() -> list[ProfileMetadata]:
    profiles_dir = get_profiles_dir()
    if not profiles_dir.exists():
        return []
    results = []
    for d in sorted(profiles_dir.iterdir()):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            results.append(ProfileMetadata.load(meta_path))
    return results


def load_profile(name: str) -> ProfileMetadata:
    meta_path = get_profiles_dir() / name / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found at {meta_path}")
    return ProfileMetadata.load(meta_path)


def get_profile_dir(name: str) -> Path:
    return get_profiles_dir() / name


def create_profile(
    name: str,
    audio_path: str | Path,
    backends: list[str] | None = None,
    epochs: int = 20,
    batch_size: int = 4,
) -> ProfileMetadata:
    """Create a new voice profile: copy audio, train requested backends."""
    from mdt.vc.registry import get_backend

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    profile_dir = get_profiles_dir() / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference audio (convert to WAV if needed)
    ref_path = profile_dir / "reference_audio.wav"
    if audio_path.suffix.lower() in (".wav",):
        shutil.copy2(audio_path, ref_path)
    else:
        # Convert via ffmpeg
        import subprocess
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-ar", "44100", "-ac", "1", str(ref_path), "-y"],
            capture_output=True,
        )

    meta = ProfileMetadata(
        name=name,
        source_audio=str(audio_path),
        source_audio_hash=_audio_hash(ref_path),
    )

    if backends is None:
        backends = ["applio", "sovits", "seedvc"]

    for backend_name in backends:
        backend = get_backend(backend_name)
        if not backend.is_available():
            meta.backends[backend_name] = BackendStatus(trained=False)
            continue

        result = backend.train(
            profile_dir=profile_dir,
            audio_path=ref_path,
            epochs=epochs,
            batch_size=batch_size,
        )

        meta.backends[backend_name] = BackendStatus(
            trained=result.model_path is not None,
            model_path=str(result.model_path) if result.model_path else None,
            index_path=str(result.index_path) if result.index_path else None,
            trained_at=meta.created_at,
            epochs=result.epochs,
            train_time_seconds=result.train_time_seconds,
        )

    meta.save(profile_dir / "metadata.json")
    return meta


def refresh_profile(
    name: str,
    audio_path: str | Path | None = None,
    backends: list[str] | None = None,
    epochs: int = 20,
    batch_size: int = 4,
) -> ProfileMetadata:
    """Retrain backends for an existing profile."""
    from mdt.vc.registry import get_backend

    meta = load_profile(name)
    profile_dir = get_profiles_dir() / name

    if audio_path:
        audio_path = Path(audio_path)
        ref_path = profile_dir / "reference_audio.wav"
        if audio_path.suffix.lower() in (".wav",):
            shutil.copy2(audio_path, ref_path)
        else:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-ar", "44100", "-ac", "1", str(ref_path), "-y"],
                capture_output=True,
            )
        meta.source_audio = str(audio_path)
        meta.source_audio_hash = _audio_hash(ref_path)

    ref_path = profile_dir / "reference_audio.wav"

    if backends is None:
        backends = list(meta.backends.keys()) or ["applio", "sovits", "seedvc"]

    for backend_name in backends:
        backend = get_backend(backend_name)
        if not backend.is_available():
            continue

        result = backend.train(
            profile_dir=profile_dir,
            audio_path=ref_path,
            epochs=epochs,
            batch_size=batch_size,
        )

        meta.backends[backend_name] = BackendStatus(
            trained=result.model_path is not None,
            model_path=str(result.model_path) if result.model_path else None,
            index_path=str(result.index_path) if result.index_path else None,
            trained_at=meta.updated_at,
            epochs=result.epochs,
            train_time_seconds=result.train_time_seconds,
        )

    meta.touch()
    meta.save(profile_dir / "metadata.json")
    return meta
