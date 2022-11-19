from ptrd import jupylet_audio as ja
from ptrd.utils import FPS


def filtered_saw(cutoff=300, freq=40, duration=3, sr=FPS): 

    # n samples (frames) in duration seconds
    frames = sr * duration
    
    # sawtooth oscillator 
    osc = ja.Oscillator('sawtooth', freq=freq)
    a0 = osc(frames=frames)
    
    # gate, schedule time for open / close
    gate = ja.LatencyGate()
    gate.open(dt=0.1)
    gate.close(dt=0.5)
    g0 = gate(frames=frames)
    
    # envelope, long release
    adsr = ja.Envelope(attack=0.4, decay=0, sustain=1, release=duration*2/3)
    
    # apply the scheduled gate to the envelope ("play" a note)
    e0 = adsr(g0)
    
    # apply the envelope to the amplitude
    a0 *= e0

    # lowpass filter 
    fltr = ja.ResonantFilter(freq=cutoff)
    a0 = fltr(a0)
    # print(type(a0))

    return a0