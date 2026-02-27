"""
Audio similarity / loss functions.

The ears of parted-record. These functions answer the question:
"How different do these two sounds... sound?"

Higher loss = more different. Zero = identical (or your ears are broken).
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any
from auraloss.freq import MultiResolutionSTFTLoss

from .audio import to_tensor, compute_mfcc, compute_mel_spectrogram, DEFAULT_SR


# ═══════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════


class MRSTFTLoss:
    """
    Multi-Resolution STFT Loss.

    The gold standard for perceptual audio comparison.
    Computes STFT at multiple resolutions (window sizes) to capture
    both transient detail (short windows) and tonal content (long windows).

    This is what most neural audio papers use. Start here.
    """

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 4096],
        hop_sizes: List[int] = [256, 512, 1024],
        win_lengths: List[int] = [1024, 2048, 4096],
        scale: Optional[str] = "mel",
        n_bins: int = 64,
        sr: int = DEFAULT_SR,
        perceptual_weighting: bool = True,
    ):
        self.loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            scale=scale,
            n_bins=n_bins,
            sample_rate=sr,
            perceptual_weighting=perceptual_weighting,
        )
        self.name = "mrstft"

    def __call__(self, audio: np.ndarray, target: np.ndarray) -> float:
        """
        Compute loss between two audio signals.
        Both inputs: 1D numpy arrays (mono audio).
        Returns: float loss value (lower = more similar).
        """
        a = to_tensor(audio)
        b = to_tensor(target)
        # Ensure same length
        min_len = min(a.shape[-1], b.shape[-1])
        a = a[..., :min_len]
        b = b[..., :min_len]
        with torch.no_grad():
            return self.loss_fn(a, b).item()


class MelSpectrogramLoss:
    """
    Mel Spectrogram L1 Loss.

    Simpler than MRSTFT but still perceptually meaningful.
    Compares mel spectrograms (which approximate human hearing).
    Cheaper to compute, good for rough matching.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.name = "mel"

    def __call__(self, audio: np.ndarray, target: np.ndarray) -> float:
        mel_a = compute_mel_spectrogram(
            audio, self.sr, self.n_mels, self.n_fft, self.hop_length
        )
        mel_b = compute_mel_spectrogram(
            target, self.sr, self.n_mels, self.n_fft, self.hop_length
        )
        # Match time dimension
        min_frames = min(mel_a.shape[1], mel_b.shape[1])
        mel_a = mel_a[:, :min_frames]
        mel_b = mel_b[:, :min_frames]
        return float(np.mean(np.abs(mel_a - mel_b)))


class MFCCLoss:
    """
    MFCC Distance Loss.

    The classic from Yee-King (2011). MFCCs capture timbral
    characteristics — the "color" of a sound. They lose phase info
    and fine temporal detail, but they're fast and surprisingly
    good for rough matching.

    Use as a regularizer alongside MRSTFT, not as the main loss.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.name = "mfcc"

    def __call__(self, audio: np.ndarray, target: np.ndarray) -> float:
        mfcc_a = compute_mfcc(audio, self.sr, self.n_mfcc, self.n_fft, self.hop_length)
        mfcc_b = compute_mfcc(target, self.sr, self.n_mfcc, self.n_fft, self.hop_length)
        min_frames = min(mfcc_a.shape[1], mfcc_b.shape[1])
        mfcc_a = mfcc_a[:, :min_frames]
        mfcc_b = mfcc_b[:, :min_frames]
        return float(np.mean((mfcc_a - mfcc_b) ** 2))


class EnvelopeLoss:
    """
    Amplitude Envelope Loss.

    Compares the loudness contour (ADSR shape) of two sounds.
    Important for matching transients and decay characteristics.
    A sound can have perfect spectral match but wrong envelope
    and still sound totally different.
    """

    def __init__(self, sr: int = DEFAULT_SR, hop_length: int = 512):
        self.sr = sr
        self.hop_length = hop_length
        self.name = "envelope"

    def __call__(self, audio: np.ndarray, target: np.ndarray) -> float:
        env_a = self._envelope(audio)
        env_b = self._envelope(target)
        min_len = min(len(env_a), len(env_b))
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]
        return float(np.mean((env_a - env_b) ** 2))

    def _envelope(self, audio: np.ndarray) -> np.ndarray:
        """RMS envelope."""
        return np.array([
            np.sqrt(np.mean(audio[i:i + self.hop_length] ** 2))
            for i in range(0, len(audio) - self.hop_length, self.hop_length)
        ])


# ═══════════════════════════════════════════════════
# Composite Loss
# ═══════════════════════════════════════════════════


class CompositeLoss:
    """
    Weighted combination of multiple loss functions.

    The real magic: combine spectral accuracy (MRSTFT) with
    timbral matching (MFCC) and envelope shape (Envelope).
    Weight them by importance.

    Default weights are a good starting point — tune based on
    what you're matching (pads want more spectral, plucks want
    more envelope).
    """

    def __init__(
        self,
        losses: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        sr: int = DEFAULT_SR,
    ):
        if losses is None:
            # Sensible defaults
            self.losses = {
                "mrstft": MRSTFTLoss(sr=sr),
                "mel": MelSpectrogramLoss(sr=sr),
                "mfcc": MFCCLoss(sr=sr),
                "envelope": EnvelopeLoss(sr=sr),
            }
        else:
            self.losses = losses

        if weights is None:
            # MRSTFT is the heavy hitter, others are supporting cast
            self.weights = {
                "mrstft": 1.0,
                "mel": 0.1,
                "mfcc": 0.01,
                "envelope": 0.5,
            }
        else:
            self.weights = weights

        self.name = "composite"

    def __call__(self, audio: np.ndarray, target: np.ndarray) -> float:
        total = 0.0
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            total += weight * loss_fn(audio, target)
        return total

    def detailed(self, audio: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Return individual loss values for debugging/logging."""
        results = {}
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            raw = loss_fn(audio, target)
            results[name] = {"raw": raw, "weighted": weight * raw, "weight": weight}
        results["total"] = sum(v["weighted"] for v in results.values())
        return results


# ═══════════════════════════════════════════════════
# Quick helpers
# ═══════════════════════════════════════════════════


def quick_compare(audio_a: np.ndarray, audio_b: np.ndarray, sr: int = DEFAULT_SR) -> Dict[str, float]:
    """
    Quick comparison of two audio signals using all loss functions.
    Returns a dict of {loss_name: value}.

    Great for sanity checking: compare a sound to itself (should be ~0),
    to noise (should be high), and to a similar sound (should be moderate).
    """
    losses = {
        "mrstft": MRSTFTLoss(sr=sr),
        "mel": MelSpectrogramLoss(sr=sr),
        "mfcc": MFCCLoss(sr=sr),
        "envelope": EnvelopeLoss(sr=sr),
    }
    return {name: fn(audio_a, audio_b) for name, fn in losses.items()}
