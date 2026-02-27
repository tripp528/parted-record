"""
Synthesizer backends for parted-record.

Start simple: a pure-Python synth with known parameters.
This lets us test the optimizer before dealing with VST hosting.

The architecture: every synth exposes the same interface:
    synth.n_params -> int
    synth.param_names -> list[str]
    synth.render(params, midi_note, duration, sr) -> np.ndarray

Parameters are always normalized to [0, 1]. The synth maps them
to musically meaningful ranges internally.
"""

import numpy as np
from numba import njit
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from .audio import DEFAULT_SR, midi_to_hz


@njit(cache=True)
def _svf_lowpass(signal, cutoff_hz, resonance, sr):
    """JIT-compiled state variable filter. Runs ~100x faster than pure Python."""
    n = len(signal)
    output = np.empty(n, dtype=np.float64)
    lp = 0.0
    bp = 0.0
    q = 1.0 - resonance
    nyquist = sr * 0.49

    for i in range(n):
        fc = cutoff_hz[i]
        if fc < 20.0:
            fc = 20.0
        elif fc > nyquist:
            fc = nyquist
        f = 2.0 * np.sin(np.pi * fc / sr)
        lp += f * bp
        hp = signal[i] - lp - q * bp
        bp += f * hp
        output[i] = lp

    return output


@dataclass
class ParamSpec:
    """Specification for a single synth parameter."""
    name: str
    min_val: float
    max_val: float
    default: float = 0.5
    log_scale: bool = False  # True for frequency-like params
    description: str = ""

    def denormalize(self, normalized: float) -> float:
        """Convert [0, 1] to actual parameter value."""
        n = np.clip(normalized, 0, 1)
        if self.log_scale:
            # Logarithmic mapping (for frequencies, times)
            log_min = np.log(max(self.min_val, 1e-6))
            log_max = np.log(max(self.max_val, 1e-6))
            return np.exp(log_min + n * (log_max - log_min))
        else:
            return self.min_val + n * (self.max_val - self.min_val)

    def normalize(self, value: float) -> float:
        """Convert actual parameter value to [0, 1]."""
        if self.log_scale:
            log_min = np.log(max(self.min_val, 1e-6))
            log_max = np.log(max(self.max_val, 1e-6))
            return (np.log(max(value, 1e-6)) - log_min) / (log_max - log_min)
        else:
            return (value - self.min_val) / (self.max_val - self.min_val)


class BaseSynth:
    """Base class for all synth backends."""

    def __init__(self):
        self.param_specs: List[ParamSpec] = []

    @property
    def n_params(self) -> int:
        return len(self.param_specs)

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.param_specs]

    def render(
        self,
        params: np.ndarray,
        midi_note: int = 60,
        duration: float = 2.0,
        sr: int = DEFAULT_SR,
    ) -> np.ndarray:
        """
        Render audio with given parameters.
        params: array of shape (n_params,), values in [0, 1]
        Returns: 1D float32 numpy array
        """
        raise NotImplementedError

    def random_params(self) -> np.ndarray:
        """Generate random parameters."""
        return np.random.uniform(0, 1, self.n_params).astype(np.float32)

    def default_params(self) -> np.ndarray:
        """Get default parameters."""
        return np.array([p.default for p in self.param_specs], dtype=np.float32)

    def describe_params(self, params: np.ndarray) -> Dict[str, float]:
        """Convert normalized params to human-readable dict."""
        return {
            spec.name: spec.denormalize(float(params[i]))
            for i, spec in enumerate(self.param_specs)
        }


class SubtractiveSynth(BaseSynth):
    """
    A classic subtractive synthesizer.

    2 oscillators (saw/square/sine/triangle) → mix → low-pass filter → ADSR amp envelope.

    17 parameters — small enough for CMA-ES to eat for breakfast,
    rich enough to make interesting sounds.
    """

    # Waveform types
    WAVEFORMS = ["sine", "triangle", "saw", "square"]

    def __init__(self):
        super().__init__()
        self.param_specs = [
            # Oscillator 1
            ParamSpec("osc1_waveform", 0, 3, 0.66, description="Waveform: sine/tri/saw/square"),
            ParamSpec("osc1_detune", -24, 24, 0.5, description="Semitone detune from MIDI note"),
            ParamSpec("osc1_level", 0, 1, 0.8, description="Oscillator 1 volume"),

            # Oscillator 2
            ParamSpec("osc2_waveform", 0, 3, 0.33, description="Waveform: sine/tri/saw/square"),
            ParamSpec("osc2_detune", -24, 24, 0.5, description="Semitone detune from MIDI note"),
            ParamSpec("osc2_level", 0, 1, 0.5, description="Oscillator 2 volume"),

            # Filter
            ParamSpec("filter_cutoff", 100, 16000, 0.7, log_scale=True, description="LP filter cutoff Hz"),
            ParamSpec("filter_resonance", 0, 0.99, 0.1, description="Filter resonance (Q)"),
            ParamSpec("filter_env_amount", 0, 8000, 0.3, description="Filter envelope depth Hz"),

            # Amp Envelope (ADSR)
            ParamSpec("amp_attack", 0.001, 2.0, 0.1, log_scale=True, description="Attack time (s)"),
            ParamSpec("amp_decay", 0.001, 2.0, 0.3, log_scale=True, description="Decay time (s)"),
            ParamSpec("amp_sustain", 0, 1, 0.6, description="Sustain level"),
            ParamSpec("amp_release", 0.001, 3.0, 0.3, log_scale=True, description="Release time (s)"),

            # Filter Envelope (ADSR)
            ParamSpec("filt_attack", 0.001, 1.0, 0.1, log_scale=True, description="Filter env attack (s)"),
            ParamSpec("filt_decay", 0.001, 2.0, 0.4, log_scale=True, description="Filter env decay (s)"),
            ParamSpec("filt_sustain", 0, 1, 0.3, description="Filter env sustain"),
            ParamSpec("filt_release", 0.001, 2.0, 0.3, log_scale=True, description="Filter env release (s)"),
        ]

    def render(
        self,
        params: np.ndarray,
        midi_note: int = 60,
        duration: float = 2.0,
        sr: int = DEFAULT_SR,
    ) -> np.ndarray:
        params = np.clip(params, 0, 1)
        p = {spec.name: spec.denormalize(float(params[i])) for i, spec in enumerate(self.param_specs)}

        n_samples = int(sr * duration)
        t = np.arange(n_samples, dtype=np.float64) / sr
        base_freq = midi_to_hz(midi_note)

        # Generate oscillators
        osc1 = self._oscillator(t, base_freq, p["osc1_detune"], int(round(p["osc1_waveform"])))
        osc2 = self._oscillator(t, base_freq, p["osc2_detune"], int(round(p["osc2_waveform"])))

        # Mix oscillators
        signal = osc1 * p["osc1_level"] + osc2 * p["osc2_level"]

        # Filter envelope
        filt_env = self._adsr(t, duration, p["filt_attack"], p["filt_decay"],
                              p["filt_sustain"], p["filt_release"])
        # Dynamic filter cutoff
        cutoff_curve = p["filter_cutoff"] + p["filter_env_amount"] * filt_env

        # Apply low-pass filter (simple one-pole per sample for smooth cutoff changes)
        signal = self._variable_lowpass(signal, cutoff_curve, p["filter_resonance"], sr)

        # Amp envelope
        amp_env = self._adsr(t, duration, p["amp_attack"], p["amp_decay"],
                             p["amp_sustain"], p["amp_release"])
        signal = signal * amp_env

        # Clip any infinities/NaN, normalize, convert to float32
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = np.abs(signal).max()
        if peak > 0:
            signal = signal / peak
        return signal.astype(np.float32)

    def _oscillator(self, t: np.ndarray, base_freq: float, detune_semitones: float, waveform: int) -> np.ndarray:
        """Generate oscillator waveform."""
        freq = base_freq * (2.0 ** (detune_semitones / 12.0))
        phase = 2.0 * np.pi * freq * t

        waveform = int(np.clip(waveform, 0, 3))
        if waveform == 0:  # Sine
            return np.sin(phase)
        elif waveform == 1:  # Triangle
            return 2.0 * np.abs(2.0 * (freq * t % 1.0) - 1.0) - 1.0
        elif waveform == 2:  # Saw
            return 2.0 * (freq * t % 1.0) - 1.0
        elif waveform == 3:  # Square
            return np.sign(np.sin(phase))
        return np.sin(phase)

    def _adsr(
        self, t: np.ndarray, duration: float,
        attack: float, decay: float, sustain: float, release: float,
    ) -> np.ndarray:
        """Generate ADSR envelope."""
        env = np.zeros_like(t)
        note_off = duration * 0.7  # Note-off at 70% of duration

        for i, ti in enumerate(t):
            if ti < attack:
                # Attack phase
                env[i] = ti / attack
            elif ti < attack + decay:
                # Decay phase
                env[i] = 1.0 - (1.0 - sustain) * ((ti - attack) / decay)
            elif ti < note_off:
                # Sustain phase
                env[i] = sustain
            else:
                # Release phase
                release_t = ti - note_off
                env[i] = sustain * np.exp(-release_t / max(release, 0.001) * 3)

        return env

    def _variable_lowpass(
        self, signal: np.ndarray, cutoff_hz: np.ndarray,
        resonance: float, sr: int,
    ) -> np.ndarray:
        """Resonant low-pass filter with time-varying cutoff (numba JIT)."""
        return _svf_lowpass(signal, cutoff_hz.astype(np.float64), resonance, sr)


class SimpleSineSynth(BaseSynth):
    """
    The simplest possible synth: a single sine wave with amplitude envelope.

    4 parameters. Perfect for testing the optimizer pipeline.
    If CMA-ES can't match THIS, something's broken.
    """

    def __init__(self):
        super().__init__()
        self.param_specs = [
            ParamSpec("detune", -12, 12, 0.5, description="Semitone detune"),
            ParamSpec("attack", 0.001, 1.0, 0.2, log_scale=True, description="Attack (s)"),
            ParamSpec("decay", 0.01, 2.0, 0.5, log_scale=True, description="Decay (s)"),
            ParamSpec("sustain", 0, 1, 0.7, description="Sustain level"),
        ]

    def render(
        self,
        params: np.ndarray,
        midi_note: int = 60,
        duration: float = 2.0,
        sr: int = DEFAULT_SR,
    ) -> np.ndarray:
        params = np.clip(params, 0, 1)
        p = {spec.name: spec.denormalize(float(params[i])) for i, spec in enumerate(self.param_specs)}

        n_samples = int(sr * duration)
        t = np.arange(n_samples, dtype=np.float64) / sr
        freq = midi_to_hz(midi_note) * (2.0 ** (p["detune"] / 12.0))

        # Pure sine
        signal = np.sin(2.0 * np.pi * freq * t)

        # Simple ADS envelope (no release for simplicity)
        env = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti < p["attack"]:
                env[i] = ti / p["attack"]
            elif ti < p["attack"] + p["decay"]:
                env[i] = 1.0 - (1.0 - p["sustain"]) * ((ti - p["attack"]) / p["decay"])
            else:
                env[i] = p["sustain"]

        signal = signal * env
        peak = np.abs(signal).max()
        if peak > 0:
            signal = signal / peak
        return signal.astype(np.float32)


# Registry of available synths
SYNTHS = {
    "sine": SimpleSineSynth,
    "subtractive": SubtractiveSynth,
}


def get_synth(name: str = "subtractive") -> BaseSynth:
    """Get a synth by name."""
    if name not in SYNTHS:
        raise ValueError(f"Unknown synth '{name}'. Available: {list(SYNTHS.keys())}")
    return SYNTHS[name]()
