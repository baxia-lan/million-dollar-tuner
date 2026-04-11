"""Audio-to-MIDI transcription for separated stems (experimental)."""

from __future__ import annotations

from pathlib import Path

import click
import librosa
import numpy as np
import pretty_midi

from mdt.analysis.pitch import detect_pitch
from mdt.analysis.rhythm import analyze_rhythm
from mdt.config import ANALYSIS_SR, HOP_LENGTH


def transcribe_stem(
    audio_path: str | Path,
    stem_type: str,
    output_path: str | Path | None = None,
) -> pretty_midi.PrettyMIDI:
    """Transcribe an audio stem to MIDI.

    Parameters
    ----------
    audio_path : path
        Path to the stem audio file.
    stem_type : str
        One of: ``'vocals'``, ``'bass'``, ``'drums'``, ``'other'``.
    output_path : path or None
        If provided, save the MIDI file here.

    Returns
    -------
    pretty_midi.PrettyMIDI
    """
    y, sr = librosa.load(str(audio_path), sr=ANALYSIS_SR, mono=True)

    if stem_type == "drums":
        midi = _transcribe_drums(y, sr)
    elif stem_type in ("bass", "vocals"):
        midi = _transcribe_monophonic(y, sr, stem_type)
    else:
        midi = _transcribe_chords(y, sr)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))
        click.echo(f"  Saved MIDI: {output_path}")

    return midi


def _transcribe_monophonic(
    y: np.ndarray,
    sr: int,
    name: str = "melody",
) -> pretty_midi.PrettyMIDI:
    """Transcribe a monophonic stem (bass or vocals) to MIDI.

    Uses PYIN pitch detection + onset detection to segment notes.
    """
    pitch_curve = detect_pitch(y, sr)
    rhythm = analyze_rhythm(y, sr)

    midi = pretty_midi.PrettyMIDI(initial_tempo=rhythm.tempo)
    program = 0 if name == "vocals" else 33  # Acoustic bass
    instrument = pretty_midi.Instrument(
        program=program,
        name=name,
    )

    # Find note segments: contiguous voiced regions
    onsets = rhythm.onset_times
    if len(onsets) == 0:
        return midi

    # Add a final offset at the end of audio
    audio_end = len(y) / sr
    offsets = np.append(onsets[1:], audio_end)

    for onset, offset in zip(onsets, offsets):
        # Get average pitch in this segment
        mask = (
            (pitch_curve.times >= onset)
            & (pitch_curve.times < offset)
            & pitch_curve.voiced_flag
        )
        if not np.any(mask):
            continue

        avg_freq = np.nanmedian(pitch_curve.frequencies[mask])
        if np.isnan(avg_freq) or avg_freq <= 0:
            continue

        # Convert to MIDI note
        midi_note = int(round(69 + 12 * np.log2(avg_freq / 440.0)))
        midi_note = max(0, min(127, midi_note))

        # Estimate velocity from RMS energy
        start_samp = int(onset * sr)
        end_samp = int(offset * sr)
        segment = y[start_samp:end_samp]
        if len(segment) == 0:
            continue
        rms = np.sqrt(np.mean(segment ** 2))
        velocity = int(np.clip(rms * 1000, 30, 127))

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=midi_note,
            start=onset,
            end=offset,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def _transcribe_drums(
    y: np.ndarray,
    sr: int,
) -> pretty_midi.PrettyMIDI:
    """Transcribe drums using onset detection + spectral classification.

    Maps detected hits to General MIDI drum map:
    - Kick: 36 (Bass Drum 1)
    - Snare: 38 (Acoustic Snare)
    - Hi-hat: 42 (Closed Hi-Hat)
    - Crash: 49 (Crash Cymbal 1)
    """
    rhythm = analyze_rhythm(y, sr)
    midi = pretty_midi.PrettyMIDI(initial_tempo=rhythm.tempo)
    drum_inst = pretty_midi.Instrument(
        program=0,
        is_drum=True,
        name="drums",
    )

    onsets = rhythm.onset_times

    for onset in onsets:
        # Classify drum hit by spectral characteristics
        start_samp = int(onset * sr)
        end_samp = min(start_samp + int(0.05 * sr), len(y))  # 50ms window
        segment = y[start_samp:end_samp]

        if len(segment) < 256:
            continue

        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)

        # Energy in frequency bands
        low_mask = freqs < 200
        mid_mask = (freqs >= 200) & (freqs < 2000)
        high_mask = freqs >= 2000

        low_energy = np.sum(spectrum[low_mask] ** 2)
        mid_energy = np.sum(spectrum[mid_mask] ** 2)
        high_energy = np.sum(spectrum[high_mask] ** 2)
        total = low_energy + mid_energy + high_energy + 1e-10

        low_ratio = low_energy / total
        high_ratio = high_energy / total

        # Classification heuristic
        if low_ratio > 0.5:
            midi_note = 36  # Kick
        elif high_ratio > 0.4:
            midi_note = 42  # Hi-hat
        elif mid_energy > low_energy:
            midi_note = 38  # Snare
        else:
            midi_note = 36  # Default to kick

        # Velocity from RMS
        rms = np.sqrt(np.mean(segment ** 2))
        velocity = int(np.clip(rms * 2000, 40, 127))

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=midi_note,
            start=onset,
            end=onset + 0.1,  # Short note for drums
        )
        drum_inst.notes.append(note)

    midi.instruments.append(drum_inst)
    return midi


def _transcribe_chords(
    y: np.ndarray,
    sr: int,
) -> pretty_midi.PrettyMIDI:
    """Approximate chord transcription for polyphonic stems.

    Uses chroma features to detect chord changes and generates
    block chords. This is approximate – polyphonic transcription
    is an unsolved problem.
    """
    rhythm = analyze_rhythm(y, sr)
    midi = pretty_midi.PrettyMIDI(initial_tempo=rhythm.tempo)
    instrument = pretty_midi.Instrument(
        program=0,  # Acoustic Grand Piano
        name="other",
    )

    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )

    # Use beat times as chord boundaries
    if len(rhythm.beat_times) < 2:
        midi.instruments.append(instrument)
        return midi

    beat_times = rhythm.beat_times
    audio_end = len(y) / sr

    for i in range(len(beat_times)):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else audio_end

        # Get average chroma in this beat
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            continue

        avg_chroma = np.mean(chroma[:, mask], axis=1)

        # Find top 3-4 pitch classes (chord tones)
        threshold = np.max(avg_chroma) * 0.6
        active_pcs = np.where(avg_chroma > threshold)[0]

        if len(active_pcs) == 0:
            continue

        # Limit to top 4 notes
        if len(active_pcs) > 4:
            indices = np.argsort(avg_chroma[active_pcs])[-4:]
            active_pcs = active_pcs[indices]

        # Generate notes in octave 4 (MIDI 60-71)
        for pc in active_pcs:
            midi_note = 60 + pc
            velocity = int(np.clip(avg_chroma[pc] * 100, 40, 100))

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=midi_note,
                start=start,
                end=end,
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi
