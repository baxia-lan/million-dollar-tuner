"""Seed-VC zero-shot voice conversion backend.

Uses the seed-vc pip package (V1 API, noF0 mode) for best voice similarity
with no training required -- only a short reference clip of the target voice.

Encapsulates all known workarounds:
  1. MPS float64 disabled before any model loading
  2. BigVGAN._from_pretrained patched for missing default params
  3. Hydra module aliasing (seed_vc.modules.* -> modules.*)
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf

from mdt.vc.base import InferResult, TrainResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level patching (applied once)
# ---------------------------------------------------------------------------

_patched = False


def _apply_patches() -> None:
    """Apply all seed-vc compatibility patches.  Idempotent."""
    global _patched
    if _patched:
        return

    # -- 1. Disable MPS to avoid float64 errors on Apple Silicon -----------
    import torch

    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]

    # -- 2. Hydra module aliasing ------------------------------------------
    #    Hydra configs inside seed_vc reference bare ``modules.*`` paths.
    #    We add the seed_vc package dir to sys.path and pre-alias key
    #    sub-packages so hydra resolves them against the patched versions.
    import seed_vc as _seed_vc_pkg

    _seed_vc_dir = os.path.dirname(_seed_vc_pkg.__file__)
    if _seed_vc_dir not in sys.path:
        sys.path.insert(0, _seed_vc_dir)

    _modules_pkg = importlib.import_module("seed_vc.modules")
    sys.modules.setdefault("modules", _modules_pkg)

    _submodules = [
        "bigvgan",
        "bigvgan.bigvgan",
        "campplus",
        "campplus.DTDNN",
        "astral_quantization",
        "astral_quantization.default_model",
        "astral_quantization.convnext",
        "astral_quantization.bsq",
        "audio",
        "v2",
        "v2.vc_wrapper",
        "v2.hf_utils",
        "v2.cfm",
        "v2.dit_wrapper",
        "v2.length_regulator",
        "v2.ar",
        "v2.dit_model",
        "v2.model",
    ]
    for sub in _submodules:
        full = f"seed_vc.modules.{sub}"
        try:
            mod = importlib.import_module(full)
            sys.modules.setdefault(f"modules.{sub}", mod)
        except Exception as exc:
            logger.debug("Could not pre-import %s: %s", full, exc)

    # -- 3. BigVGAN._from_pretrained default-value patch -------------------
    #    The huggingface_hub caller may pass ``proxies`` and
    #    ``resume_download`` as keyword arguments, but the original
    #    classmethod does not declare defaults for them, causing a
    #    TypeError.  We wrap it with safe defaults.
    from seed_vc.modules.bigvgan import bigvgan

    _orig_from_pretrained = bigvgan.BigVGAN._from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _patched_from_pretrained(
        cls,
        *,
        model_id,
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        map_location="cpu",
        strict=False,
        use_cuda_kernel=False,
        **kw: Any,
    ):
        return _orig_from_pretrained(
            cls,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            use_cuda_kernel=use_cuda_kernel,
            **kw,
        )

    bigvgan.BigVGAN._from_pretrained = _patched_from_pretrained

    _patched = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_audio_data(path: Path):
    """Load a WAV/audio file into a seed_vc ``AudioData`` (mono, int16)."""
    from seed_vc.Models.audio import AudioData

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    samples_int16 = (data * 32767).astype(np.int16)
    duration = len(samples_int16) / sr
    return AudioData(
        samples=samples_int16.tolist(),
        mel_chunks=None,
        duration=duration,
        samples_count=len(samples_int16),
        sample_rate=sr,
        metadata=None,
    )


def _find_reference_audio(profile_dir: Path) -> Path | None:
    """Return the reference audio file stored in *profile_dir*, if any."""
    for candidate in ("reference.wav", "reference.flac", "reference.mp3"):
        p = profile_dir / candidate
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class SeedVCBackend:
    """Zero-shot voice conversion via Seed-VC (V1 noF0 mode)."""

    name: str = "seedvc"
    requires_training: bool = False

    # Default inference hyper-parameters (V1 noF0 -- best voice similarity).
    _diffusion_steps: int = 25
    _inference_cfg_rate: float = 0.7
    _fp16: bool = False

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the seed-vc package is importable."""
        try:
            import seed_vc  # noqa: F401
            return True
        except ImportError:
            return False

    def train(
        self,
        profile_dir: Path,
        audio_path: Path,
        *,
        epochs: int = 20,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> TrainResult:
        """No training needed -- just copy the reference audio into the profile.

        The reference clip is stored as ``reference.wav`` inside
        *profile_dir* so that :meth:`infer` can find it later.
        """
        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        dest = profile_dir / "reference.wav"

        audio_path = Path(audio_path)
        if audio_path.suffix.lower() == ".wav":
            shutil.copy2(audio_path, dest)
        else:
            # Re-encode to WAV for consistency.
            data, sr = sf.read(str(audio_path), dtype="float32")
            sf.write(str(dest), data, sr)

        return TrainResult(
            model_path=dest,
            metadata={"backend": self.name, "reference": str(dest)},
        )

    def infer(
        self,
        profile_dir: Path,
        source_vocals: Path,
        output_path: Path,
        *,
        pitch_shift: int = 0,
        diffusion_steps: int | None = None,
        inference_cfg_rate: float | None = None,
        fp16: bool | None = None,
        **kwargs: Any,
    ) -> InferResult:
        """Run Seed-VC V1 (noF0) voice conversion.

        Parameters
        ----------
        profile_dir:
            Directory containing ``reference.wav`` (written by :meth:`train`).
        source_vocals:
            Path to the source vocal track to convert.
        output_path:
            Where to write the converted WAV.
        pitch_shift:
            Semitone shift (passed as ``semi_tone_shift``).
        diffusion_steps:
            Override default diffusion steps (25).
        inference_cfg_rate:
            Override default CFG rate (0.7).
        fp16:
            Override default fp16 flag (False -- safest on CPU/MPS).
        """
        _apply_patches()

        profile_dir = Path(profile_dir)
        source_vocals = Path(source_vocals)
        output_path = Path(output_path)

        # Resolve reference audio
        ref_path = _find_reference_audio(profile_dir)
        if ref_path is None:
            raise FileNotFoundError(
                f"No reference audio found in {profile_dir}. "
                "Run train() first to store reference.wav."
            )

        # Resolve hyper-parameters
        steps = diffusion_steps if diffusion_steps is not None else self._diffusion_steps
        cfg_rate = inference_cfg_rate if inference_cfg_rate is not None else self._inference_cfg_rate
        use_fp16 = fp16 if fp16 is not None else self._fp16

        logger.info(
            "Seed-VC infer: source=%s ref=%s steps=%d cfg=%.2f fp16=%s",
            source_vocals, ref_path, steps, cfg_rate, use_fp16,
        )

        # Load audio into AudioData objects
        source_ad = _load_audio_data(source_vocals)
        target_ad = _load_audio_data(ref_path)

        # Run V1 inference (non-streaming, f0_condition=False)
        from seed_vc.api import inference as seedvc_inference

        t0 = time.monotonic()
        result_ad = seedvc_inference(
            source=source_ad,
            target=target_ad,
            diffusion_steps=steps,
            length_adjust=1.0,
            inference_cfg_rate=cfg_rate,
            f0_condition=False,
            auto_f0_adjust=False,
            semi_tone_shift=pitch_shift,
            fp16=use_fp16,
            streaming=False,
            realtime=False,
        )
        elapsed = time.monotonic() - t0

        # Extract waveform from AudioData
        samples = np.array(result_ad.samples, dtype=np.int16)
        waveform = samples.astype(np.float32) / 32767.0
        sr_out = int(result_ad.sample_rate)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, sr_out)

        logger.info(
            "Seed-VC done: %.1fs audio in %.1fs (RTF %.2f), saved to %s",
            len(waveform) / sr_out,
            elapsed,
            elapsed / (len(waveform) / sr_out) if len(waveform) > 0 else 0,
            output_path,
        )

        return InferResult(
            output_path=output_path,
            sample_rate=sr_out,
            infer_time_seconds=elapsed,
            metadata={
                "backend": self.name,
                "diffusion_steps": steps,
                "inference_cfg_rate": cfg_rate,
                "f0_condition": False,
                "fp16": use_fp16,
                "semi_tone_shift": pitch_shift,
            },
        )
