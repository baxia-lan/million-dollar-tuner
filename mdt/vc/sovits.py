"""so-vits-svc-fork voice conversion backend.

Uses the ``so-vits-svc-fork`` pip package via subprocess CLI commands.
All working directories are scoped under ``profile_dir/sovits/`` so that
dataset, config and checkpoint files never pollute the project root.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

from mdt.vc.base import InferResult, TrainResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PYTHON = sys.executable
SPEAKER = "speaker0"

# Relative paths within the sovits work-dir that the CLI expects.
_RAW_REL = Path("dataset_raw") / SPEAKER
_DS_REL = Path("dataset") / "44k" / SPEAKER
_CONFIG_REL = Path("configs") / "44k" / "config.json"
_LOGS_REL = Path("logs") / "44k"
_FILELISTS_REL = Path("filelists") / "44k"


def _run(
    args: list[str],
    *,
    cwd: Path,
    label: str,
    timeout: int = 3600,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess inside *cwd*, log output, and raise on failure."""
    import os

    env = {**os.environ, **(extra_env or {})}
    log.info("[%s] running: %s (cwd=%s)", label, " ".join(args), cwd)
    t0 = time.time()
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
        env=env,
    )
    elapsed = time.time() - t0
    if result.stdout:
        for line in result.stdout.splitlines()[-30:]:
            log.debug("[%s stdout] %s", label, line)
    if result.stderr:
        for line in result.stderr.splitlines()[-30:]:
            log.debug("[%s stderr] %s", label, line)
    log.info("[%s] finished in %.1fs (rc=%d)", label, elapsed, result.returncode)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (rc={result.returncode}).\n"
            f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )
    return result


def _work_dir(profile_dir: Path) -> Path:
    """Return the sovits working directory for a profile."""
    return profile_dir / "sovits"


def _ensure_dirs(work: Path) -> None:
    """Create the directory skeleton that so-vits-svc-fork expects."""
    (work / _RAW_REL).mkdir(parents=True, exist_ok=True)
    (work / _DS_REL).mkdir(parents=True, exist_ok=True)
    (work / _CONFIG_REL.parent).mkdir(parents=True, exist_ok=True)
    (work / _LOGS_REL).mkdir(parents=True, exist_ok=True)
    (work / _FILELISTS_REL).mkdir(parents=True, exist_ok=True)


def _split_audio(
    audio_path: Path,
    dest_dir: Path,
    segment_seconds: float = 10.0,
    min_seconds: float = 2.0,
) -> int:
    """Split *audio_path* into ~segment_seconds WAV files under *dest_dir*.

    Returns the number of segments written.
    """
    audio, sr = sf.read(str(audio_path))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    segment_len = int(segment_seconds * sr)
    count = 0
    for i in range(0, len(audio), segment_len):
        seg = audio[i : i + segment_len]
        if len(seg) < int(min_seconds * sr):
            continue
        out = dest_dir / f"segment_{count:04d}.wav"
        sf.write(str(out), seg, sr)
        count += 1
    return count


def _find_best_checkpoint(logs_dir: Path) -> Path | None:
    """Return the generator checkpoint with the highest epoch number."""
    best: Path | None = None
    best_epoch = 0
    for p in sorted(logs_dir.glob("G_*.pth")):
        if p.name == "G_0.pth":
            continue
        try:
            epoch = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best = p
    return best


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class SoVITSBackend:
    """Voice-conversion backend powered by so-vits-svc-fork."""

    name: str = "sovits"
    requires_training: bool = True

    # -- availability -------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return *True* when ``so_vits_svc_fork`` can be imported."""
        try:
            import so_vits_svc_fork  # noqa: F401

            return True
        except Exception:
            return False

    # -- training -----------------------------------------------------------

    def train(
        self,
        profile_dir: Path,
        audio_path: Path,
        *,
        epochs: int = 400,
        batch_size: int = 6,
        **kwargs,
    ) -> TrainResult:
        """Run the full so-vits-svc preprocessing + training pipeline.

        Parameters
        ----------
        profile_dir:
            Root directory for this voice profile.  All artifacts are placed
            under ``profile_dir/sovits/``.
        audio_path:
            Path to a single WAV file of the target speaker.
        epochs:
            Number of training epochs.
        batch_size:
            Training batch size.

        Extra *kwargs* forwarded to config:
            ``eval_interval``, ``keep_ckpts``, ``num_workers``.
        """
        profile_dir = Path(profile_dir)
        audio_path = Path(audio_path)
        work = _work_dir(profile_dir)

        # ---- 1. clean + scaffold -----------------------------------------
        if work.exists():
            shutil.rmtree(work)
        _ensure_dirs(work)

        # ---- 2. split audio into segments --------------------------------
        raw_dir = work / _RAW_REL
        n_segs = _split_audio(audio_path, raw_dir)
        log.info("Split %s into %d segments under %s", audio_path, n_segs, raw_dir)
        if n_segs == 0:
            raise RuntimeError(
                f"No usable segments from {audio_path} "
                "(audio shorter than 2 seconds?)"
            )

        # ---- 3. preprocess -----------------------------------------------
        _run(
            [PYTHON, "-m", "so_vits_svc_fork", "pre-resample",
             "-i", str(work / "dataset_raw"),
             "-o", str(work / "dataset" / "44k")],
            cwd=work,
            label="pre-resample",
        )

        _run(
            [PYTHON, "-m", "so_vits_svc_fork", "pre-config",
             "-i", str(work / "dataset" / "44k"),
             "-f", str(work / _FILELISTS_REL),
             "-c", str(work / _CONFIG_REL)],
            cwd=work,
            label="pre-config",
        )

        _run(
            [PYTHON, "-m", "so_vits_svc_fork", "pre-hubert",
             "-i", str(work / "dataset" / "44k"),
             "-c", str(work / _CONFIG_REL),
             "-fm", "dio", "-n", "1"],
            cwd=work,
            label="pre-hubert",
            timeout=1200,
        )

        # ---- 4. update config --------------------------------------------
        config_path = work / _CONFIG_REL
        with open(config_path) as fh:
            config = json.load(fh)

        config["train"]["epochs"] = epochs
        config["train"]["batch_size"] = batch_size
        config["train"]["eval_interval"] = kwargs.get("eval_interval", 100)
        config["train"]["keep_ckpts"] = kwargs.get("keep_ckpts", 3)
        config["train"]["num_workers"] = kwargs.get("num_workers", 2)

        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)
        log.info(
            "Updated config: epochs=%d, batch_size=%d, eval_interval=%d",
            epochs,
            batch_size,
            config["train"]["eval_interval"],
        )

        # ---- 5. train ----------------------------------------------------
        logs_dir = work / _LOGS_REL
        t0 = time.time()
        _run(
            [PYTHON, "-m", "so_vits_svc_fork", "train",
             "-c", str(config_path),
             "-m", str(logs_dir)],
            cwd=work,
            label="train",
            timeout=kwargs.get("train_timeout", 7200),
            extra_env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
        )
        train_time = time.time() - t0

        # ---- 6. locate best checkpoint -----------------------------------
        best_ckpt = _find_best_checkpoint(logs_dir)
        if best_ckpt is None:
            raise RuntimeError(
                f"Training completed but no G_*.pth checkpoint found in {logs_dir}"
            )

        best_epoch = int(best_ckpt.stem.split("_")[1])
        log.info("Best checkpoint: %s (epoch %d)", best_ckpt, best_epoch)

        return TrainResult(
            model_path=best_ckpt,
            config_path=config_path,
            train_time_seconds=train_time,
            epochs=best_epoch,
            metadata={
                "work_dir": str(work),
                "segments": n_segs,
                "batch_size": batch_size,
                "requested_epochs": epochs,
            },
        )

    # -- inference ----------------------------------------------------------

    def infer(
        self,
        profile_dir: Path,
        source_vocals: Path,
        output_path: Path,
        *,
        pitch_shift: int = 0,
        **kwargs,
    ) -> InferResult:
        """Convert *source_vocals* using the trained model in *profile_dir*.

        Parameters
        ----------
        profile_dir:
            Root profile directory (must contain ``sovits/`` from a prior
            ``train()`` call).
        source_vocals:
            Path to the source WAV to convert.
        output_path:
            Where to write the converted WAV.
        pitch_shift:
            Semitone transpose (``-t`` flag).  0 = no shift.

        Extra *kwargs* (with defaults matching the reference script):
            ``noise_scale`` (0.4), ``pad_seconds`` (0.5),
            ``chunk_seconds`` (0.5), ``max_chunk_seconds`` (30),
            ``device`` ("cpu"), ``f0_method`` ("dio"),
            ``speaker`` (auto-detected from config).
        """
        profile_dir = Path(profile_dir)
        source_vocals = Path(source_vocals)
        output_path = Path(output_path)
        work = _work_dir(profile_dir)

        config_path = work / _CONFIG_REL
        logs_dir = work / _LOGS_REL

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found at {config_path}. Has train() been run?"
            )

        # Auto-detect speaker name from config.
        speaker: str = kwargs.get("speaker", "")
        if not speaker:
            with open(config_path) as fh:
                cfg = json.load(fh)
            speakers = list(cfg.get("spk", {}).keys())
            speaker = speakers[0] if speakers else SPEAKER
        log.info("Using speaker: %s", speaker)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        noise_scale = str(kwargs.get("noise_scale", 0.4))
        pad_seconds = str(kwargs.get("pad_seconds", 0.5))
        chunk_seconds = str(kwargs.get("chunk_seconds", 0.5))
        max_chunk_seconds = str(kwargs.get("max_chunk_seconds", 30))
        device = kwargs.get("device", "cpu")
        f0_method = kwargs.get("f0_method", "dio")

        cmd = [
            PYTHON, "-m", "so_vits_svc_fork", "infer",
            "-o", str(output_path),
            "-s", speaker,
            "-m", str(logs_dir),
            "-c", str(config_path),
            "-fm", f0_method,
            "-d", device,
            "-t", str(pitch_shift),
            "-n", noise_scale,
            "-p", pad_seconds,
            "-ch", chunk_seconds,
            "-mc", max_chunk_seconds,
            str(source_vocals),
        ]

        t0 = time.time()
        _run(
            cmd,
            cwd=work,
            label="infer",
            timeout=kwargs.get("infer_timeout", 1800),
        )
        infer_time = time.time() - t0

        if not output_path.exists():
            raise FileNotFoundError(
                f"Inference completed but output file not found at {output_path}"
            )

        # Read sample rate from the produced file.
        info = sf.info(str(output_path))

        return InferResult(
            output_path=output_path,
            sample_rate=info.samplerate,
            infer_time_seconds=infer_time,
            metadata={
                "speaker": speaker,
                "pitch_shift": pitch_shift,
                "noise_scale": float(noise_scale),
                "f0_method": f0_method,
                "device": device,
            },
        )
