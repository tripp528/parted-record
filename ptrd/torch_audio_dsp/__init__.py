
# https://github.com/mthrok/audio/tree/dsp/torchaudio/prototype/functional

from ._dsp import (
    adsr_envelope, 
    apply_time_varying_filter, 
    extend_pitch, 
    oscillator_bank, 
    sinc_filter
)
from .functional import (
    add_noise, 
    barkscale_fbanks, 
    convolve, 
    fftconvolve
)


__all__ = [
    "add_noise",
    "adsr_envelope",
    "apply_time_varying_filter",
    "barkscale_fbanks",
    "convolve",
    "extend_pitch",
    "fftconvolve",
    "oscillator_bank",
    "sinc_filter",
]