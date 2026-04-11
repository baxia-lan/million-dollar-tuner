"""Synthesis engine: render MIDI back to audio using FluidSynth."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pretty_midi

from mdt.audio.effects import apply_effects
from mdt.audio.io import save_audio
from mdt.config import DEFAULT_SOUNDFONT, OUTPUT_SR


def synthesize_midi(
    midi_path: str | Path,
    output_path: str | Path | None = None,
    soundfont_path: str | Path | None = None,
    sr: int = OUTPUT_SR,
    apply_fx: bool = True,
) -> np.ndarray:
    """Render a MIDI file to audio using FluidSynth.

    Parameters
    ----------
    midi_path : path
        Path to the MIDI file.
    output_path : path or None
        If provided, save the rendered audio here.
    soundfont_path : path or None
        Path to a .sf2 SoundFont file.
        None uses the default soundfont.
    sr : int
        Output sample rate.
    apply_fx : bool
        Whether to apply basic effects (EQ, reverb).

    Returns
    -------
    np.ndarray
        Rendered audio samples.
    """
    midi_path = Path(midi_path)
    sf_path = Path(soundfont_path) if soundfont_path else DEFAULT_SOUNDFONT

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    # Render to audio
    if sf_path.exists():
        audio = midi_data.fluidsynth(fs=sr, sf2_path=str(sf_path))
    else:
        click.echo(
            f"  WARNING: SoundFont not found at {sf_path}. "
            f"Using default FluidSynth rendering."
        )
        audio = midi_data.fluidsynth(fs=sr)

    audio = audio.astype(np.float32)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    if apply_fx:
        audio = apply_effects(audio, sr)

    if output_path is not None:
        save_audio(output_path, audio, sr)
        click.echo(f"  Saved audio: {output_path}")

    return audio


def resynthesize_all_stems(
    midi_dir: str | Path,
    output_dir: str | Path,
    soundfont_path: str | Path | None = None,
    sr: int = OUTPUT_SR,
) -> dict[str, np.ndarray]:
    """Render all MIDI stems in a directory and mix them.

    Parameters
    ----------
    midi_dir : path
        Directory containing stem MIDI files (``vocals.mid``, etc.).
    output_dir : path
        Directory to save rendered audio files.
    soundfont_path : path or None
        SoundFont path.
    sr : int
        Sample rate.

    Returns
    -------
    dict[str, np.ndarray]
        Stem name -> rendered audio.
    """
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = {}
    mix = None

    for midi_file in sorted(midi_dir.glob("*.mid")):
        stem_name = midi_file.stem
        click.echo(f"  Synthesizing: {stem_name}")

        out_path = output_dir / f"{stem_name}_synth.wav"
        audio = synthesize_midi(
            midi_path=midi_file,
            output_path=out_path,
            soundfont_path=soundfont_path,
            sr=sr,
        )
        rendered[stem_name] = audio

        if mix is None:
            mix = audio.copy()
        else:
            min_len = min(len(mix), len(audio))
            mix = mix[:min_len] + audio[:min_len]

    # Save the full mix
    if mix is not None:
        peak = np.max(np.abs(mix))
        if peak > 1.0:
            mix = mix / peak * 0.95
        mix_path = output_dir / "full_mix_synth.wav"
        save_audio(mix_path, mix, sr)
        click.echo(f"  Saved full mix: {mix_path}")

    return rendered
