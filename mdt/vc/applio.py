"""Applio RVC voice-conversion backend.

Every heavy operation (preprocess, extract, train, infer) runs inside a
subprocess whose cwd is APPLIO_DIR, with ``sys.path`` set so that Applio's
own ``rvc`` package is found instead of the ``rvc-clean`` package used by
the host process.  This avoids import conflicts and segfaults.

The ``rvc.pth`` file (if it exists in the active venv's site-packages)
is temporarily renamed before each subprocess launch and restored
afterwards, preventing the venv from auto-importing the wrong ``rvc``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .base import InferResult, TrainResult

logger = logging.getLogger(__name__)

APPLIO_DIR = Path(os.environ.get("MDT_APPLIO_DIR", "/Users/sarsa/claude/Applio"))

_SAMPLE_RATE = 40000
_F0_METHOD = "rmvpe"
_EMBEDDER_MODEL = "contentvec"
_VOCODER = "HiFi-GAN"
_SAVE_EVERY_EPOCH = 10


# ---------------------------------------------------------------------------
# rvc.pth guard – prevent the venv from auto-importing the wrong rvc package
# ---------------------------------------------------------------------------

def _find_rvc_pth() -> Path | None:
    """Return the path to ``rvc.pth`` inside the active venv, or *None*."""
    prefix = Path(sys.prefix)
    candidates = list(prefix.glob("lib/python*/site-packages/rvc.pth"))
    if candidates:
        return candidates[0]
    return None


class _RvcPthGuard:
    """Context manager that disables ``rvc.pth`` for the duration of a block.

    On enter the file is renamed to ``rvc.pth.bak``; on exit it is restored.
    If no ``rvc.pth`` exists the guard is a no-op.
    """

    def __init__(self) -> None:
        self._pth: Path | None = None
        self._bak: Path | None = None

    def __enter__(self) -> "_RvcPthGuard":
        pth = _find_rvc_pth()
        if pth is not None and pth.exists():
            bak = pth.with_suffix(".pth.bak")
            try:
                pth.rename(bak)
                self._pth = pth
                self._bak = bak
                logger.debug("Disabled %s -> %s", pth, bak)
            except OSError:
                logger.warning("Could not rename %s; proceeding anyway", pth)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._pth is not None and self._bak is not None:
            try:
                self._bak.rename(self._pth)
                logger.debug("Restored %s", self._pth)
            except OSError:
                logger.warning("Could not restore %s from %s", self._pth, self._bak)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _subprocess_env() -> dict[str, str]:
    """Build an environment dict suitable for Applio subprocesses."""
    env = os.environ.copy()
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "29501"
    # Remove PYTHONPATH entries that could pull in rvc-clean
    if "PYTHONPATH" in env:
        parts = [p for p in env["PYTHONPATH"].split(os.pathsep) if "rvc-clean" not in p]
        env["PYTHONPATH"] = os.pathsep.join(parts) if parts else ""
    return env


def _run_script(script: str, *, timeout: int = 7200, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Write *script* to a temp file and execute it as a subprocess in APPLIO_DIR."""
    env = _subprocess_env()
    if extra_env:
        env.update(extra_env)

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="mdt_applio_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)

        with _RvcPthGuard():
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=str(APPLIO_DIR),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return result
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _run_applio_script(argv: list[str], *, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run an Applio helper script (e.g. train.py) as a subprocess."""
    env = _subprocess_env()
    with _RvcPthGuard():
        return subprocess.run(
            [sys.executable, *argv],
            cwd=str(APPLIO_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Script templates
# ---------------------------------------------------------------------------

_SCRIPT_PREAMBLE = textwrap.dedent("""\
    import os, sys, types, glob, json, time
    import numpy as np

    APPLIO_DIR = {applio_dir!r}

    # Isolate sys.path
    sys.path = [p for p in sys.path if "rvc-clean" not in p]
    if APPLIO_DIR in sys.path:
        sys.path.remove(APPLIO_DIR)
    sys.path.insert(0, APPLIO_DIR)
    train_dir = os.path.join(APPLIO_DIR, "rvc", "train")
    if train_dir in sys.path:
        sys.path.remove(train_dir)
    sys.path.insert(1, train_dir)

    # Purge stale rvc modules
    for key in list(sys.modules.keys()):
        if key == "rvc" or key.startswith("rvc."):
            mod = sys.modules[key]
            if mod and hasattr(mod, "__file__") and mod.__file__ and "rvc-clean" in str(mod.__file__):
                del sys.modules[key]

    os.chdir(APPLIO_DIR)

    # Fake torchfcpe to avoid optional-dependency crash
    _fake = types.ModuleType("torchfcpe")
    _fake.spawn_infer_model_from_pt = None
    sys.modules["torchfcpe"] = _fake
""")


def _preamble() -> str:
    return _SCRIPT_PREAMBLE.format(applio_dir=str(APPLIO_DIR))


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class ApplioBackend:
    """Applio RVC voice-conversion backend.

    All heavy work is delegated to subprocesses running inside APPLIO_DIR
    so that the Applio ``rvc`` package does not conflict with the host
    environment's ``rvc-clean`` package.
    """

    name: str = "applio"
    requires_training: bool = True

    # -- availability -------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when APPLIO_DIR looks like a valid Applio checkout."""
        if not APPLIO_DIR.is_dir():
            logger.warning("APPLIO_DIR %s does not exist", APPLIO_DIR)
            return False

        required = [
            APPLIO_DIR / "rvc" / "train" / "preprocess" / "preprocess.py",
            APPLIO_DIR / "rvc" / "train" / "train.py",
            APPLIO_DIR / "rvc" / "train" / "process" / "extract_index.py",
            APPLIO_DIR / "rvc" / "train" / "extract" / "preparing_files.py",
            APPLIO_DIR / "rvc" / "infer" / "pipeline.py",
            APPLIO_DIR / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G40k.pth",
            APPLIO_DIR / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0D40k.pth",
        ]
        for p in required:
            if not p.exists():
                logger.warning("Applio prerequisite missing: %s", p)
                return False
        return True

    # -- training -----------------------------------------------------------

    def train(
        self,
        profile_dir: Path,
        audio_path: Path,
        *,
        epochs: int = 30,
        batch_size: int = 4,
        **kwargs,
    ) -> TrainResult:
        profile_dir = Path(profile_dir)
        audio_path = Path(audio_path)
        model_name = profile_dir.name
        t0 = time.monotonic()

        logger.info("Applio train: model=%s audio=%s epochs=%d", model_name, audio_path, epochs)

        # 1) Preprocess + extract features (single subprocess)
        self._run_preprocess_and_extract(model_name, audio_path)

        # 2) Train (uses Applio's own train.py)
        self._run_training(model_name, epochs, batch_size)

        # 3) Build FAISS index
        self._run_index(model_name)

        # 4) Locate artefacts and copy to profile_dir
        exp_dir = APPLIO_DIR / "logs" / model_name
        model_path = self._find_trained_model(exp_dir, model_name)
        index_path = self._find_index(exp_dir, model_name)

        dest_dir = profile_dir / "applio"
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_model: Path | None = None
        dest_index: Path | None = None
        dest_config: Path | None = None

        if model_path:
            dest_model = dest_dir / model_path.name
            shutil.copy2(model_path, dest_model)
            logger.info("Copied model to %s", dest_model)
        else:
            raise RuntimeError(
                f"Training completed but no model file found in {exp_dir}"
            )

        if index_path:
            dest_index = dest_dir / index_path.name
            shutil.copy2(index_path, dest_index)
            logger.info("Copied index to %s", dest_index)

        config_src = exp_dir / "config.json"
        if config_src.exists():
            dest_config = dest_dir / "config.json"
            shutil.copy2(config_src, dest_config)

        elapsed = time.monotonic() - t0
        logger.info("Applio training finished in %.1fs", elapsed)

        return TrainResult(
            model_path=dest_model,
            index_path=dest_index,
            config_path=dest_config,
            train_time_seconds=elapsed,
            epochs=epochs,
            metadata={
                "backend": self.name,
                "sample_rate": _SAMPLE_RATE,
                "vocoder": _VOCODER,
                "exp_dir": str(exp_dir),
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
        profile_dir = Path(profile_dir)
        source_vocals = Path(source_vocals)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        applio_dir = profile_dir / "applio"
        model_path = self._find_profile_model(applio_dir)
        index_path = self._find_profile_index(applio_dir)

        # FAISS index causes SIGSEGV on macOS — disable until fixed
        index_rate = 0.0
        index_path = None

        logger.info(
            "Applio infer: model=%s index=%s input=%s pitch=%d",
            model_path, index_path, source_vocals, pitch_shift,
        )

        t0 = time.monotonic()

        script = _preamble() + textwrap.dedent("""\
            import torch
            import soundfile as sf

            MODEL_PATH = {model_path!r}
            INDEX_PATH = {index_path!r}
            INPUT_AUDIO = {input_audio!r}
            OUTPUT_AUDIO = {output_audio!r}
            PITCH = {pitch}
            INDEX_RATE = {index_rate}

            # ---- Load model checkpoint ----
            from rvc.lib.algorithm.synthesizers import Synthesizer
            from rvc.lib.utils import load_audio_infer, load_embedding
            from rvc.configs.config import Config

            config = Config()

            cpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            use_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            text_enc_hidden_dim = 768 if version == "v2" else 256
            vocoder = cpt.get("vocoder", "HiFi-GAN")

            net_g = Synthesizer(
                *cpt["config"],
                use_f0=use_f0,
                text_enc_hidden_dim=text_enc_hidden_dim,
                vocoder=vocoder,
            )
            del net_g.enc_q
            net_g.load_state_dict(cpt["weight"], strict=False)
            net_g = net_g.to(config.device).float()
            net_g.eval()

            # ---- Load embedder ----
            hubert_model = load_embedding("contentvec").to(config.device).float()
            hubert_model.eval()

            # ---- Load audio ----
            audio = load_audio_infer(INPUT_AUDIO, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            # ---- Run pipeline ----
            from rvc.infer.pipeline import Pipeline
            vc = Pipeline(tgt_sr, config)

            audio_opt = vc.pipeline(
                model=hubert_model,
                net_g=net_g,
                sid=0,
                audio=audio,
                pitch=PITCH,
                f0_method="rmvpe",
                file_index=INDEX_PATH if INDEX_PATH else "",
                index_rate=INDEX_RATE,
                pitch_guidance=bool(use_f0),
                volume_envelope=1.0,
                version=version,
                protect=0.5,
                f0_autotune=False,
                f0_autotune_strength=1.0,
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
            )

            sf.write(OUTPUT_AUDIO, audio_opt, tgt_sr, format="WAV")

            # Write metadata to stdout for the parent process
            import json as _json
            print("__MDT_META__" + _json.dumps({{"sample_rate": tgt_sr}}))
        """).format(
            model_path=str(model_path),
            index_path=str(index_path) if index_path else "",
            input_audio=str(source_vocals),
            output_audio=str(output_path),
            pitch=pitch_shift,
            index_rate=index_rate,
        )

        result = _run_script(script)
        elapsed = time.monotonic() - t0

        if not output_path.exists():
            raise RuntimeError(
                f"Applio inference did not produce output at {output_path}.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Parse sample rate from subprocess metadata
        sample_rate = 44100
        for line in result.stdout.splitlines():
            if line.startswith("__MDT_META__"):
                meta = json.loads(line[len("__MDT_META__"):])
                sample_rate = meta.get("sample_rate", sample_rate)
                break

        logger.info("Applio inference finished in %.1fs -> %s", elapsed, output_path)
        return InferResult(
            output_path=output_path,
            sample_rate=sample_rate,
            infer_time_seconds=elapsed,
            metadata={"backend": self.name},
        )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    # -- preprocess + feature extraction ------------------------------------

    def _run_preprocess_and_extract(self, model_name: str, audio_path: Path) -> None:
        """Preprocess audio and extract F0 + embeddings in a single subprocess."""
        script = _preamble() + textwrap.dedent("""\
            import shutil
            from tqdm import tqdm

            MODEL_NAME = {model_name!r}
            AUDIO_PATH = {audio_path!r}
            SAMPLE_RATE = {sample_rate}
            EMBEDDER_MODEL = {embedder!r}

            exp_dir = os.path.join("logs", MODEL_NAME)
            os.makedirs(exp_dir, exist_ok=True)

            # ---------- Preprocess ----------
            dataset_dir = os.path.join(exp_dir, "dataset")
            os.makedirs(dataset_dir, exist_ok=True)
            target_wav = os.path.join(dataset_dir, os.path.basename(AUDIO_PATH))
            if not os.path.exists(target_wav):
                shutil.copy2(AUDIO_PATH, target_wav)

            from rvc.train.preprocess.preprocess import PreProcess, save_dataset_duration

            pp = PreProcess(sr=SAMPLE_RATE, exp_dir=exp_dir)
            audio_length = pp.process_audio(
                path=target_wav,
                idx0=0,
                sid=0,
                cut_preprocess="Automatic",
                process_effects=True,
                noise_reduction=False,
                reduction_strength=0.7,
                chunk_len=3.0,
                overlap_len=0.3,
                normalization_mode="pre",
            )
            save_dataset_duration(
                os.path.join(exp_dir, "model_info.json"),
                dataset_duration=audio_length,
            )

            n_slices = len(glob.glob(os.path.join(exp_dir, "sliced_audios", "*.wav")))
            n_16k = len(glob.glob(os.path.join(exp_dir, "sliced_audios_16k", "*.wav")))
            print(f"Preprocess done: {{n_slices}} slices, {{n_16k}} 16k versions, {{audio_length:.1f}}s")

            # ---------- Feature extraction ----------
            wav_path = os.path.join(exp_dir, "sliced_audios_16k")
            os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

            files = []
            for f in sorted(glob.glob(os.path.join(wav_path, "*.wav"))):
                fn = os.path.basename(f)
                files.append([
                    f,
                    os.path.join(exp_dir, "f0", fn + ".npy"),
                    os.path.join(exp_dir, "f0_voiced", fn + ".npy"),
                    os.path.join(exp_dir, "extracted", fn.replace("wav", "npy")),
                ])

            if not files:
                print("ERROR: No sliced audio files found for feature extraction")
                sys.exit(1)

            print(f"Extracting features from {{len(files)}} files...")

            # --- F0 with RMVPE ---
            import torch
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor
            from rvc.lib.utils import load_audio_16k

            class _F0Extractor:
                def __init__(self):
                    self.f0_bin = 256
                    self.f0_max = 1100.0
                    self.f0_min = 50.0
                    self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
                    self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
                    self.model = RMVPE0Predictor(
                        os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                        device="cpu",
                    )

                def process(self, file_info):
                    inp_path, coarse_path, full_path, _ = file_info
                    if os.path.exists(coarse_path) and os.path.exists(full_path):
                        return
                    try:
                        np_arr = load_audio_16k(inp_path)
                        f0 = self.model.infer_from_audio(np_arr, thred=0.03)
                        np.save(full_path, f0, allow_pickle=False)
                        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
                        f0_mel = np.clip(
                            (f0_mel - self.f0_mel_min) * (self.f0_bin - 2)
                            / (self.f0_mel_max - self.f0_mel_min) + 1,
                            1, self.f0_bin - 1,
                        )
                        np.save(coarse_path, np.rint(f0_mel).astype(int), allow_pickle=False)
                    except Exception as e:
                        print(f"F0 error on {{inp_path}}: {{e}}")

            fe = _F0Extractor()
            for fi in tqdm(files, desc="F0"):
                fe.process(fi)

            # --- Embeddings with ContentVec ---
            from rvc.lib.utils import load_embedding
            emb_model = load_embedding(EMBEDDER_MODEL).to("cpu").float()
            emb_model.eval()

            for fi in tqdm(files, desc="Embeddings"):
                wav_file, _, _, out_file = fi
                if os.path.exists(out_file):
                    continue
                feats = torch.from_numpy(load_audio_16k(wav_file)).float().view(1, -1)
                with torch.no_grad():
                    result = emb_model(feats)["last_hidden_state"]
                feats_out = result.squeeze(0).float().cpu().numpy()
                if not np.isnan(feats_out).any():
                    np.save(out_file, feats_out, allow_pickle=False)
                else:
                    print(f"WARNING: NaN embeddings for {{wav_file}}")

            del emb_model

            # --- Save embedder info ---
            model_info_path = os.path.join(exp_dir, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, "r") as f:
                    data = json.load(f)
            else:
                data = {{}}
            data["embedder_model"] = EMBEDDER_MODEL
            with open(model_info_path, "w") as f:
                json.dump(data, f, indent=4)

            # --- Generate config + filelist ---
            from rvc.train.extract.preparing_files import generate_config, generate_filelist
            generate_config(SAMPLE_RATE, exp_dir)
            generate_filelist(exp_dir, SAMPLE_RATE, include_mutes=2)

            n_f0 = len(glob.glob(os.path.join(exp_dir, "f0", "*.npy")))
            n_emb = len(glob.glob(os.path.join(exp_dir, "extracted", "*.npy")))
            print(f"Feature extraction done: {{n_f0}} F0, {{n_emb}} embeddings")
        """).format(
            model_name=model_name,
            audio_path=str(audio_path),
            sample_rate=_SAMPLE_RATE,
            embedder=_EMBEDDER_MODEL,
        )

        logger.info("Running preprocess + extract for %s", model_name)
        result = _run_script(script, timeout=3600)
        logger.info("Preprocess stdout:\n%s", result.stdout[-3000:] if result.stdout else "(empty)")
        if result.stderr:
            logger.debug("Preprocess stderr:\n%s", result.stderr[-3000:])

        if result.returncode != 0:
            raise RuntimeError(
                f"Preprocess/extract failed (rc={result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # Verify outputs exist
        exp_dir = APPLIO_DIR / "logs" / model_name
        filelist = exp_dir / "filelist.txt"
        if not filelist.exists():
            raise RuntimeError(
                f"Preprocessing did not produce {filelist}.\n"
                f"stdout:\n{result.stdout[-2000:]}"
            )

    # -- training -----------------------------------------------------------

    def _run_training(self, model_name: str, epochs: int, batch_size: int) -> None:
        """Launch Applio's train.py as a subprocess."""
        pretrained_g = str(
            APPLIO_DIR / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G40k.pth"
        )
        pretrained_d = str(
            APPLIO_DIR / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0D40k.pth"
        )

        # Fall back to empty strings if pretraineds somehow vanished
        if not Path(pretrained_g).exists():
            logger.warning("Pretrained G not found at %s", pretrained_g)
            pretrained_g = ""
        if not Path(pretrained_d).exists():
            logger.warning("Pretrained D not found at %s", pretrained_d)
            pretrained_d = ""

        save_every = min(_SAVE_EVERY_EPOCH, epochs)

        argv = [
            str(APPLIO_DIR / "rvc" / "train" / "train.py"),
            model_name,             # model_name
            str(save_every),        # save_every_epoch
            str(epochs),            # total_epoch
            pretrained_g,           # pretrainG
            pretrained_d,           # pretrainD
            "-",                    # gpus ("-" = CPU)
            str(batch_size),        # batch_size
            str(_SAMPLE_RATE),      # sample_rate
            "True",                 # save_only_latest
            "True",                 # save_every_weights
            "False",                # cache_data_in_gpu
            "False",                # overtraining_detector
            "50",                   # overtraining_threshold
            "True",                 # cleanup
            _VOCODER,               # vocoder
            "False",                # checkpointing
        ]

        logger.info("Training: %s", " ".join(argv))
        result = _run_applio_script(argv, timeout=14400)
        logger.info("Train stdout (tail):\n%s", result.stdout[-3000:] if result.stdout else "(empty)")
        if result.stderr:
            logger.debug("Train stderr (tail):\n%s", result.stderr[-3000:])

        # Applio's train.py calls os._exit(2333333) on success, which maps to
        # a non-zero return code.  We treat that as success.
        # On macOS the shell sees (2333333 % 256) or the signal-based encoding.
        # We accept rc==0 as well as the known Applio exit pattern.
        rc = result.returncode
        # os._exit(2333333) -> waitpid returns 2333333 on some platforms, or
        # 2333333 & 0xFF = 85 on POSIX (low byte).  Safest: verify model exists.
        model_found = bool(self._find_trained_model(APPLIO_DIR / "logs" / model_name, model_name))
        if not model_found:
            raise RuntimeError(
                f"Training failed (rc={rc}): no model file produced.\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )
        logger.info("Training completed (rc=%d), model file found", rc)

    # -- index building -----------------------------------------------------

    def _run_index(self, model_name: str) -> None:
        """Build a FAISS index via Applio's extract_index.py."""
        exp_dir = str(APPLIO_DIR / "logs" / model_name)
        argv = [
            str(APPLIO_DIR / "rvc" / "train" / "process" / "extract_index.py"),
            exp_dir,
            "Auto",
        ]
        logger.info("Building index: %s", " ".join(argv))
        result = _run_applio_script(argv, timeout=600)
        if result.returncode != 0:
            logger.warning(
                "Index build returned %d (non-fatal).\nstderr: %s",
                result.returncode,
                result.stderr[-1000:],
            )
        else:
            logger.info("Index build succeeded")

    # -- artefact location helpers ------------------------------------------

    @staticmethod
    def _find_trained_model(exp_dir: Path, model_name: str) -> Path | None:
        """Find the newest exported .pth model in exp_dir."""
        pattern = str(exp_dir / f"{model_name}_*e_*s.pth")
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
        return None

    @staticmethod
    def _find_index(exp_dir: Path, model_name: str) -> Path | None:
        """Find the .index file in exp_dir."""
        matches = glob.glob(str(exp_dir / "*.index"))
        if matches:
            return Path(matches[0])
        return None

    @staticmethod
    def _find_profile_model(applio_dir: Path) -> Path:
        """Find the model .pth inside a profile's applio/ directory."""
        matches = sorted(applio_dir.glob("*.pth"))
        if not matches:
            raise FileNotFoundError(
                f"No .pth model found in {applio_dir}. Has training been run?"
            )
        return matches[-1]

    @staticmethod
    def _find_profile_index(applio_dir: Path) -> Path | None:
        """Find the .index file inside a profile's applio/ directory."""
        matches = list(applio_dir.glob("*.index"))
        return matches[0] if matches else None
