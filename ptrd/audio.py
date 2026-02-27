"""
Audio I/O, resampling, normalization, and feature extraction.

The ears before the brain.
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_SR = 44100
DEFAULT_DURATION = 2.0  # seconds


def load(
    path: str | Path,
    sr: int = DEFAULT_SR,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file. Returns (audio, sample_rate).

    Audio is normalized to [-1, 1] and optionally trimmed/padded.
    """
    audio, orig_sr = librosa.load(
        str(path), sr=sr, mono=mono, duration=duration, offset=offset
    )
    audio = normalize(audio)
    return audio, sr


def save(path: str | Path, audio: np.ndarray, sr: int = DEFAULT_SR):
    """Save audio to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio


def trim_silence(audio: np.ndarray, sr: int = DEFAULT_SR, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    """Pad with zeros or trim to exact length."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)), mode='constant')


def to_tensor(audio: np.ndarray) -> torch.Tensor:
    """
    Convert numpy audio to torch tensor shaped for auraloss.
    Input: (n_samples,) or (channels, n_samples)
    Output: (1, 1, n_samples) — batch=1, channel=1
    """
    if audio.ndim == 1:
        return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    elif audio.ndim == 2:
        # Take first channel if stereo
        return torch.from_numpy(audio[0]).float().unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute mel spectrogram in dB. Returns (n_mels, n_frames)."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(S, ref=np.max)


def compute_mfcc(
    audio: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute MFCCs. Returns (n_mfcc, n_frames)."""
    return librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )


def detect_pitch(audio: np.ndarray, sr: int = DEFAULT_SR) -> Optional[float]:
    """
    Detect fundamental frequency (Hz) of a pitched sound.
    Returns None if no clear pitch detected.
    """
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    # Get the highest magnitude pitch per frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        pitch = pitches[idx, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if not pitch_values:
        return None

    # Return median pitch (robust to octave errors)
    return float(np.median(pitch_values))


def hz_to_midi(hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def midi_to_hz(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def estimate_duration(audio: np.ndarray, sr: int = DEFAULT_SR, top_db: float = 30.0) -> float:
    """Estimate the 'active' duration of a sound (excluding trailing silence)."""
    trimmed = trim_silence(audio, sr, top_db)
    return len(trimmed) / sr


def audio_stats(audio: np.ndarray, sr: int = DEFAULT_SR) -> dict:
    """Get quick stats about an audio signal."""
    return {
        "duration": len(audio) / sr,
        "peak": float(np.abs(audio).max()),
        "rms": float(np.sqrt(np.mean(audio ** 2))),
        "pitch_hz": detect_pitch(audio, sr),
        "sample_rate": sr,
        "samples": len(audio),
    }
