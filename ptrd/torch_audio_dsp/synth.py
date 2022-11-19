
import torch
import torchaudio

from ptrd import utils 
from ptrd.torch_audio_dsp import (
    oscillator_bank,
    extend_pitch,
    adsr_envelope,
    sinc_filter,
    apply_time_varying_filter,
)


def sine(f0=440, duration=2, sr=utils.FPS): 
    ''' 
        - f0 (float): fundamental note frequency
        - duration (float): seconds 
        - 
    '''
    n_frames = sr * duration 

    freq_arr = torch.full((n_frames, 1), 440)
    amp_arr = torch.ones((n_frames, 1))

    a0 = oscillator_bank(
        freq_arr, 
        amp_arr, 
        sample_rate=sr
    )
    return a0


def saw(f0=440, duration=2, sr=utils.FPS, ): 
    ''' 
        - f0 (float): fundamental note frequency
        - duration (float): seconds 
        - 
    '''
    # generate base frequency and amp arrays (sine wave) 
    n_frames = sr * duration 
    freq_arr = torch.full((n_frames, 1), 440)
    amp_arr = torch.ones((n_frames, 1))

    # create harmonics in frequency 
    num_pitches = int(sr / f0)
    freq_arr = extend_pitch(freq_arr, num_pitches)

    # modify amp array in some way ???
    mults = [-((-1) ** i) / (torch.pi * i) for i in range(1, 1+num_pitches)]
    amp_arr = extend_pitch(amp_arr, mults)
    
    a0 = oscillator_bank(
        freq_arr, 
        amp_arr, 
        sample_rate=sr
    )
    return a0
