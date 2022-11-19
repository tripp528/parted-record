### Trained synth simplest form:

Creating EDM has a huge sound design barrier that gets in the way sometimes.

A great way around this can be to utilize splice. However, if you find a synth bass sound you like, transposing can squash some of the beauty of the sample. 

You can also try to recreate the sound using a VST and some effects. 

Wouldn't it be cool if we just had a very simple optimizer, where each trainable parameter was a knob on the VST / effects? Even a few hundred trainable parameters (TINY for a machine learning model) would provide the ability to closely replicate most synth sounds found on splice for example. 

Training would be simple: you simply 'overfit' your model ('synth preset') on a single sample.

Loss function: feed the model the same fundamental frequency (MIDI note) & compare waveforms. 

For starters, this could be just an evolutionary algorithm. Nothing even needs to be differentiable. 

Current work worth looking into:

- https://medium.com/@LeonFedden/using-ai-to-defeat-software-synthesisers-caded8822846

- http://www.yeeking.net/matthew_yee-king_dphil_thesis_2011.pdf

- https://github.com/DBraun/DawDreamer

- https://github.com/spotify/pedalboard

- https://github.com/hq9000/py_headless_daw

- https://github.com/sergree/matchering

- https://www.youtube.com/watch?v=Q40qEg8Yq5c

- https://neutone.space/

- https://github.com/acids-ircam/RAVE

- https://github.com/hyakuchiki/realtimeDDSP

- https://mawf.io/

- https://github.com/pytorch/audio

- https://github.com/nir/jupylet

- https://github.com/pytorch/audio/issues/2835
