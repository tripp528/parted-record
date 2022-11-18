

import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import rfft, rfftfreq
import librosa, librosa.display
from IPython.display import Audio
import IPython.display as ipd
import sounddevice as sd


FPS = 44100


''' General 
'''

def stft(sample, sr=FPS): 
    pass
    

''' Synth
'''

    

''' Visualize / play audio 
'''

def play(data, autoplay=False, out_gain=0.2, sr=FPS):
    # Only normalize the audio if it's too loud
    # if abs(data).max() > 1.0:
    data /= abs(data).max()
    
    # make it quieter 
    data *= out_gain

    return ipd.display(Audio(
        data=data, rate=sr, normalize=False, autoplay=autoplay
    ))


def wave_plot(audio, ax=None, show=True, sr=FPS):
    '''takes a tensor as input (from ddsp)'''
    # plot waveform
    librosa.display.waveshow(audio.squeeze(), ax=ax)
    if show:
        plt.show()


def fft_plot(sample, ax=None, show=False, sr=FPS, freq_min=0, freq_max=20000):
    ''' Plot a raw fourier transform of a sample.
        - sample: numpy array
    '''
    sample_normalized = sample / sample.max()
    yf = rfft(sample_normalized)
    xf = rfftfreq(len(sample_normalized), 1 / sr)

    # get rid of data outside (freq_min, freq_max)
    n_seconds = len(sample) // sr
    x = xf[freq_min * n_seconds : freq_max * n_seconds]
    y = np.abs(yf)[freq_min * n_seconds : freq_max * n_seconds]

    if ax is None: fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(x, y)
    if show: plt.show()
    return ax


def spec_plot(sample, ax=None, show=True, sr=FPS, freq_range=(0, 20000)):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    spec, freqs, times, im = ax.specgram(
        sample, 
        mode='magnitude', 
        Fs=sr, 
        cmap=plt.get_cmap('hot'), 
        vmin=-60, 
        vmax=-10,
    )

    ax.set_ylim(bottom=freq_range[0], top=freq_range[1])

    if show: plt.show


def visualize_sample(sample, sr=FPS, freq_range=(1, 600)):
    ''' Plot and visualize a sample. 
    '''
    sample = np.asarray(sample).squeeze()

    fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(15, 8))

    # wav (sound file style) plot
    wave_plot(sample, ax=ax2, sr=sr, show=False)

    # spectrogram
    spec_plot(sample, ax=ax3, show=False, sr=sr, freq_range=freq_range)

    # fft
    fft_plot(sample, ax=ax4, freq_min=freq_range[0], freq_max=freq_range[1], sr=sr)

    ipd.display(fig)
    plt.close()

    # audio player
    ipd.display(ipd.Audio(sample, rate=sr))

