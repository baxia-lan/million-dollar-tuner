#!/usr/bin/env python3
"""
Analyze vocal track to identify song sections (intro, rap, melodic, bridge, outro).

Uses energy envelope, pitch analysis (mean F0, F0 range), and onset density
to classify sections. Tuned for "越界 (Duck The Rope)" which has both rap
and melodic/chorus sections.

Key observations from the audio:
- Rap sections: mean F0 ~145-170Hz, narrow F0 range (<80Hz), flat pitch contour
- Melodic sections: mean F0 ~200-270Hz, wider F0 range (>80Hz), varied pitch
- Intro: low energy (first ~22s)
- Outro: near-silence after ~222s
"""

import librosa
import numpy as np
import json

VOCALS_PATH = "/Users/sarsa/claude/million-dollar-tuner/stems_dtr/vocals.wav"
OUTPUT_JSON = "/Users/sarsa/claude/million-dollar-tuner/song_sections.json"
OUTPUT_TXT = "/Users/sarsa/claude/million-dollar-tuner/song_sections.txt"

HOP_LENGTH = 512
SR = 44100
WINDOW_SEC = 4.0       # 4-second analysis windows for stable features
MIN_SECTION_SEC = 8.0   # Minimum section length to avoid fragmentation


def load_audio():
    print(f"Loading {VOCALS_PATH}...")
    y, sr = librosa.load(VOCALS_PATH, sr=SR, mono=True)
    duration = len(y) / sr
    print(f"  Duration: {duration:.1f}s, SR: {sr}")
    return y, sr, duration


def compute_frame_features(y, sr):
    """Compute per-frame features: energy, pitch, onsets."""
    print("Computing energy envelope...")
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    print("Computing pitch with pyin...")
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
        sr=sr, hop_length=HOP_LENGTH
    )

    print("Computing onset strength...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)

    return {
        'rms': rms,
        'f0': f0, 'voiced_flag': voiced_flag,
        'onset_env': onset_env,
    }


def compute_windowed_features(feats, duration):
    """Aggregate frame features into fixed-size windows."""
    window_frames = int(WINDOW_SEC * SR / HOP_LENGTH)
    n_frames = len(feats['rms'])
    n_windows = int(np.ceil(n_frames / window_frames))

    windows = []
    for i in range(n_windows):
        s = i * window_frames
        e = min((i + 1) * window_frames, n_frames)
        t_start = i * WINDOW_SEC
        t_end = min((i + 1) * WINDOW_SEC, duration)

        rms_slice = feats['rms'][s:e]
        f0_slice = feats['f0'][s:e]
        voiced_slice = feats['voiced_flag'][s:e]
        onset_slice = feats['onset_env'][s:e]

        mean_energy = float(np.mean(rms_slice))

        n_voiced = int(np.sum(voiced_slice)) if len(voiced_slice) > 0 else 0
        voiced_ratio = n_voiced / len(voiced_slice) if len(voiced_slice) > 0 else 0.0

        voiced_f0 = f0_slice[voiced_slice] if len(voiced_slice) > 0 else np.array([])
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)] if len(voiced_f0) > 0 else np.array([])

        if len(voiced_f0) > 2:
            mean_pitch = float(np.mean(voiced_f0))
            pitch_cents_std = float(np.std(1200 * np.log2(voiced_f0 / np.mean(voiced_f0))))
            f0_range_hz = float(np.max(voiced_f0) - np.min(voiced_f0))
        elif len(voiced_f0) == 1:
            mean_pitch = float(voiced_f0[0])
            pitch_cents_std = 0.0
            f0_range_hz = 0.0
        else:
            mean_pitch = 0.0
            pitch_cents_std = 0.0
            f0_range_hz = 0.0

        onset_density = float(np.mean(onset_slice))

        windows.append({
            't_start': t_start,
            't_end': t_end,
            'mean_energy': mean_energy,
            'voiced_ratio': voiced_ratio,
            'mean_pitch': mean_pitch,
            'pitch_cents_std': pitch_cents_std,
            'f0_range_hz': f0_range_hz,
            'onset_density': onset_density,
        })

    return windows


def classify_windows(windows):
    """
    Classify each window using:
    1. Energy thresholds for silence/low-energy
    2. Mean F0 register: <185Hz = rap register, >185Hz = melodic register
    3. F0 range: narrow range + low register = rap, wider range + higher register = melodic
    4. Pitch cents std as tiebreaker
    """
    energies = np.array([w['mean_energy'] for w in windows])
    max_energy = np.max(energies) if np.max(energies) > 0 else 1.0
    norm_energies = energies / max_energy

    # Energy thresholds (normalized)
    SILENCE_THRESH = 0.03
    LOW_ENERGY_THRESH = 0.50  # Intro sections have energy < 50% of max

    # Pitch thresholds (from empirical data)
    RAP_F0_CEILING = 185.0     # Hz - rap sections have mean F0 below this
    RAP_RANGE_CEILING = 80.0   # Hz - rap sections have narrow F0 range
    MELODIC_CENTS_FLOOR = 150  # cents std - melodic sections tend above this
    # For high-register flat sections: only classify as rap if pitch is in the
    # "rap-singing" zone (roughly 185-230Hz). Above 230Hz it's almost certainly
    # melodic even with flat pitch. Below 185Hz is already handled.
    RAP_SINGING_CEILING = 235.0  # Hz - flat pitch above this = still melodic

    for i, w in enumerate(windows):
        ne = norm_energies[i]

        if ne < SILENCE_THRESH:
            w['class'] = 'silent'
        elif ne < LOW_ENERGY_THRESH:
            w['class'] = 'low_energy'
        else:
            # Primary: register-based classification
            # The data shows a clear split: rap ~145-170Hz, melodic ~200-270Hz
            if w['mean_pitch'] > 0 and w['mean_pitch'] < RAP_F0_CEILING:
                # Low register -> rap
                w['class'] = 'rap'
            elif w['mean_pitch'] >= RAP_F0_CEILING:
                # Higher register - check if it's flat-pitched rap or melodic
                if (w['f0_range_hz'] < RAP_RANGE_CEILING and
                        w['pitch_cents_std'] < MELODIC_CENTS_FLOOR and
                        w['mean_pitch'] < RAP_SINGING_CEILING):
                    # High register but flat pitch in rap-singing zone
                    w['class'] = 'rap'
                else:
                    w['class'] = 'melodic'
            else:
                # No pitch detected but has energy - classify by onset density
                w['class'] = 'rap' if w['onset_density'] > 1.3 else 'low_energy'

    return windows


def smooth_classifications(windows, kernel_size=3):
    """Apply majority-vote smoothing to reduce isolated misclassifications."""
    classes = [w['class'] for w in windows]
    smoothed = list(classes)

    for i in range(len(classes)):
        # Only smooth active sections (don't smooth silent/low_energy)
        if classes[i] in ('rap', 'melodic'):
            start = max(0, i - kernel_size // 2)
            end = min(len(classes), i + kernel_size // 2 + 1)
            neighborhood = classes[start:end]
            # Count active types only
            rap_count = neighborhood.count('rap')
            mel_count = neighborhood.count('melodic')
            if rap_count > mel_count:
                smoothed[i] = 'rap'
            elif mel_count > rap_count:
                smoothed[i] = 'melodic'
            # tie: keep original

    for i, w in enumerate(windows):
        w['class'] = smoothed[i]

    return windows


def merge_windows_to_sections(windows, duration):
    """Merge consecutive windows of the same class into sections."""
    if not windows:
        return []

    raw_sections = []
    current_class = windows[0]['class']
    current_start = windows[0]['t_start']

    for i in range(1, len(windows)):
        if windows[i]['class'] != current_class:
            raw_sections.append({
                'start': current_start,
                'end': windows[i - 1]['t_end'],
                'type': current_class,
            })
            current_class = windows[i]['class']
            current_start = windows[i]['t_start']

    raw_sections.append({
        'start': current_start,
        'end': windows[-1]['t_end'],
        'type': current_class,
    })

    # Merge very short sections into neighbors
    merged = []
    for sec in raw_sections:
        sec_len = sec['end'] - sec['start']
        if sec_len < MIN_SECTION_SEC and merged:
            # Absorb into previous section
            merged[-1]['end'] = sec['end']
        else:
            merged.append(sec)

    # Second pass: merge consecutive same-type sections
    final = [merged[0]]
    for sec in merged[1:]:
        if sec['type'] == final[-1]['type']:
            final[-1]['end'] = sec['end']
        else:
            final.append(sec)

    return final


def post_process_sections(sections, duration):
    """Clean up edge cases before labeling."""
    # If the last active section before silence/outro is short (<15s)
    # and the same type as the section before it wouldn't be, merge it
    # with whatever makes more sense (extend previous or mark as outro).
    if len(sections) >= 3:
        last_active_idx = len(sections) - 1
        # Find last non-silent/low_energy section
        for idx in range(len(sections) - 1, -1, -1):
            if sections[idx]['type'] not in ('silent', 'low_energy'):
                last_active_idx = idx
                break

        last_active = sections[last_active_idx]
        # If last active section is short and at the tail of the song,
        # treat it as part of the outro (vocal fadeout)
        if (last_active['end'] - last_active['start'] <= 15.0 and
                last_active['end'] >= duration - 25.0 and
                last_active_idx > 0):
            # Merge with the section after it (the outro) if exists
            if last_active_idx < len(sections) - 1:
                sections[last_active_idx + 1]['start'] = last_active['start']
                sections[last_active_idx + 1]['type'] = 'low_energy'
                sections.pop(last_active_idx)
            else:
                # It IS the last section - change it to low_energy for outro
                last_active['type'] = 'low_energy'

    return sections


def assign_labels(sections, duration):
    """Assign musical labels to sections."""
    # Post-process before labeling
    sections = post_process_sections(sections, duration)

    rap_count = 0
    melodic_count = 0
    bridge_count = 0

    for i, sec in enumerate(sections):
        sec_type = sec['type']
        sec_start = sec['start']
        sec_end = sec['end']

        if sec_type in ('silent', 'low_energy'):
            if sec_start < 5.0:
                sec['label'] = 'intro'
                sec['type'] = 'intro'
            elif sec_end >= duration - 15.0:
                sec['label'] = 'outro'
                sec['type'] = 'outro'
            else:
                bridge_count += 1
                sec['label'] = f'bridge{bridge_count}'
                sec['type'] = 'bridge'
        elif sec_type == 'rap':
            rap_count += 1
            sec['label'] = f'verse{rap_count}_rap'
        elif sec_type == 'melodic':
            melodic_count += 1
            sec['label'] = f'chorus{melodic_count}'

    return sections


def print_summary(sections, duration):
    """Print and save a text summary of the song structure."""
    lines = []
    lines.append("Song Structure Analysis: 越界 (Duck The Rope)")
    lines.append(f"Total duration: {duration:.1f}s")
    lines.append("=" * 65)
    lines.append("")

    for sec in sections:
        sec_len = sec['end'] - sec['start']
        lines.append(
            f"  [{sec['start']:6.1f}s - {sec['end']:6.1f}s]  "
            f"({sec_len:5.1f}s)  "
            f"{sec['type']:10s}  {sec['label']}"
        )

    lines.append("")

    type_counts = {}
    type_durations = {}
    for sec in sections:
        t = sec['type']
        d = sec['end'] - sec['start']
        type_counts[t] = type_counts.get(t, 0) + 1
        type_durations[t] = type_durations.get(t, 0.0) + d

    lines.append("Summary:")
    for t in sorted(type_counts.keys()):
        lines.append(
            f"  {t:12s}: {type_counts[t]} section(s), "
            f"{type_durations[t]:.1f}s total"
        )

    text = "\n".join(lines)
    print(text)
    return text


def main():
    y, sr, duration = load_audio()
    feats = compute_frame_features(y, sr)

    print("Computing windowed features...")
    windows = compute_windowed_features(feats, duration)

    print("Classifying windows...")
    windows = classify_windows(windows)

    print("Smoothing classifications...")
    windows = smooth_classifications(windows, kernel_size=3)

    # Print window-level debug
    print(f"\nWindow-level classification ({len(windows)} windows of {WINDOW_SEC}s):")
    for w in windows:
        print(
            f"  {w['t_start']:5.0f}-{w['t_end']:5.0f}s  "
            f"E={w['mean_energy']:.4f}  "
            f"VR={w['voiced_ratio']:.2f}  "
            f"F0={w['mean_pitch']:6.1f}Hz  "
            f"Fstd={w['pitch_cents_std']:6.1f}c  "
            f"Frng={w['f0_range_hz']:6.1f}Hz  "
            f"Ons={w['onset_density']:.3f}  "
            f"=> {w['class']}"
        )

    print("\nMerging into sections...")
    sections = merge_windows_to_sections(windows, duration)

    print("Assigning labels...")
    sections = assign_labels(sections, duration)

    # Round times
    for sec in sections:
        sec['start'] = round(sec['start'], 1)
        sec['end'] = round(sec['end'], 1)

    # Save JSON
    output = {"sections": sections}
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON: {OUTPUT_JSON}")

    # Save text summary
    text = print_summary(sections, duration)
    with open(OUTPUT_TXT, 'w') as f:
        f.write(text + "\n")
    print(f"Saved text: {OUTPUT_TXT}")

    return sections


if __name__ == '__main__':
    main()
