# parted-record: AI Synth Matching — Research Dump

> *You heard a banger Splice sample. Sounds perfect but won't transpose for shit. So you reverse-engineer the synth preset. With math. This is how.*

**The core idea:** Train an optimizer where each parameter maps to a VST knob. Overfit on a single sample. Reverse-engineer the preset. Get a playable, transposable instrument instead of a one-note sample.

---

## Table of Contents

1. [Original README Links — Status Check](#original-readme-links)
2. [DDSP Deep Dive](#ddsp-deep-dive)
3. [Broader Landscape: AI + Sound Design in 2026](#broader-landscape)
4. [Technical Approaches: Loss Functions & Optimizers](#technical-approaches)
5. [Headless Rendering on Linux](#headless-rendering-on-linux)
6. [Toolchain Recommendation](#toolchain-recommendation)

---

## Original README Links — Status Check

### 🟢 [Leon Fedden — "Using AI to Defeat Software Synthesisers" (2017)](https://medium.com/@LeonFedden/using-ai-to-defeat-software-synthesisers-caded8822846)

**Status: Alive, historically significant, but aged out.**

Leon Fedden was a student at Goldsmiths (Music Computing) who got obsessed with the same idea you have — using ML to automate synthesizer programming. He built **RenderMan** (see below), a C++/Python VSTi host for headless synth rendering. The Medium article is readable, genuine, and basically the OG pitch for this whole project.

Fedden went on to work at DeepMind and seems to have moved into broader ML research. His RenderMan project evolved into **DawDreamer** (maintained by someone else now). The 2017 article is more "here's the problem" than "here's the solution" — but it's the right problem.

### 📜 [Matthew Yee-King PhD Thesis (2011)](http://www.yeeking.net/matthew_yee-king_dphil_thesis_2011.pdf)

**Status: PDF presumably still alive, content is old but foundational.**

The OG. Yee-King used evolutionary algorithms (genetic algorithms + CMA-ES) to match synthesizer parameters to target sounds at the University of Sussex ~2011. He built a software synthesizer with known parameters, generated sounds, then evolved patches to match them. The thesis is dry academia but the ideas are all there.

Key contributions:
- Established that **evolutionary strategies work** for non-differentiable synth parameters
- Fitness functions based on MFCC distance
- Showed that direct optimization beats naive random search by a lot

Yee-King went on to teach at Goldsmiths and has stayed adjacent to music tech/AI. The work is old but the approach (ES/CMA-ES + audio loss) is still a valid baseline.

### 🟢 [DawDreamer](https://github.com/DBraun/DawDreamer)

**Status: VERY alive. Actively maintained. This is THE tool.**

~1.4k stars. Evolved directly from Fedden's RenderMan. Built by David Braun. This is a Python framework that gives you:
- Headless VST instrument/effect hosting on macOS, Windows, AND Linux ✅
- MIDI playback
- Automation at audio rate
- FAUST integration
- Transpilation to JAX (so you can differentiate through FAUST DSP)
- Full multiprocessing support
- Works on Google Colab and Docker

**This is your render engine.** For the parted-record project, DawDreamer is the backbone. You load a VST, set parameters, render audio, compute loss, repeat. It runs completely headless on Linux. Recent commits as of early 2026.

### 🟢 [Spotify Pedalboard](https://github.com/spotify/pedalboard)

**Status: Very alive, actively maintained by Spotify's Audio Intelligence Lab.**

~5k+ stars. This is what Spotify uses internally for data augmentation and powers their AI DJ. It wraps JUCE and gives you:
- VST3 and Audio Unit loading from Python
- Built-in effects (reverb, compression, EQ, etc.)
- Low latency, thread-safe, fast
- Works on Linux headlessly

**vs DawDreamer:** Pedalboard is simpler and more production-grade for effects chains. DawDreamer is better for full DAW-like rendering with MIDI and automation. For VST *instruments* (not just effects), DawDreamer wins. For VST effects processing, either works.

### 🟡 [py_headless_daw](https://github.com/hq9000/py_headless_daw)

**Status: Alive but basically abandonware. Low stars, old commits.**

A Python-native headless DAW concept. Supports MIDI tracks, audio routing, and some basic plugins. Looked promising in 2020 but hasn't kept up. DawDreamer does everything this does and more. Skip it.

### 🟢 [Matchering](https://github.com/sergree/matchering)

**Status: Very alive, popular, actually useful.**

~1k stars. But it's doing a different thing — **audio mastering** rather than synth matching. You give it a target track and a reference, it matches RMS, frequency response, peak amplitude, and stereo width. Used in UVR5, available as a ComfyUI node. It ranked #3 in a blind mastering test behind two pro engineers.

**Relevance to parted-record:** Low. It's mastering, not synthesis. Could be useful for post-processing your matched synth output to make it sit right in a mix, but it's not the optimizer you're looking for.

### 🟢 [Neutone](https://neutone.space/) → now [neutone.ai](https://neutone.ai/)

**Status: Alive, evolved, and genuinely cool.**

Neutone is a platform and VST plugin from QosmoInc that lets researchers deploy neural audio models *into your DAW*. It's a bridge between Python/PyTorch research and real-time DAW use. They've shipped:
- **Neutone FX plugin** — free VST that runs community-uploaded neural audio models
- **Morpho** — real-time tone-morphing plugin (their commercial product)
- **Max for Live devices** — neural tools for Ableton

The `realtimeDDSP` repo below specifically targets Neutone export. Highly relevant for the endgame: once you've matched the synth, you could deploy a neural version as a Neutone model.

### 🟢 [RAVE (ACIDS-IRCAM)](https://github.com/acids-ircam/RAVE)

**Status: Very alive, actively maintained, has a VST plugin now.**

RAVE = Realtime Audio Variational autoEncoder. From IRCAM (the French music research institute). It's a VAE-based model that:
- Trains on hours of audio (an instrument, a voice, environmental sound)
- Encodes to a compact latent space
- Decodes back to high-quality audio in real time

They now have a **RAVE VST** (Windows/Mac/Linux beta) that lets you load trained RAVE models into your DAW. The latent space is musical — you can morph, interpolate, and mess with it expressively.

**Relevance to parted-record:** Different approach — RAVE learns a *data-driven* model of a sound rather than mapping to synth parameters. More like "I recorded a bunch of this synth's sounds and learned its manifold" rather than "I reverse-engineered the preset." Cool direction though.

### 🟡 [realtimeDDSP (hyakuchiki)](https://github.com/hyakuchiki/realtimeDDSP)

**Status: Alive, small project, interesting.**

Realtime streaming-compatible DDSP in PyTorch, designed to export to the Neutone format. Uses harmonic synthesis + filtered noise (the DDSP core idea) with pitch/loudness input. Last commit was a couple years ago, not super active. But it works and the code is clean.

**Relevance:** High for architecture inspiration. This is basically a differentiable synth you could use as a *proxy* when your VST isn't differentiable. Train DDSP to match the VST sound, then optimize the DDSP parameters which ARE differentiable.

### ❌ [mawf.io](https://mawf.io/)

**Status: 404 DEAD.**

Gone. No idea what this was. Moving on.

---

## DDSP Deep Dive

### The Paper

**DDSP: Differentiable Digital Signal Processing** — Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, Adam Roberts (Google Magenta) — [ICLR 2020](https://arxiv.org/abs/2001.04643)

The key insight: instead of having a neural net generate raw audio (slow, data-hungry, black box), you have a neural net generate *parameters* for classical DSP components (oscillators, filters, reverb) that are themselves differentiable. Backprop flows all the way through.

The result: high-quality audio synthesis with way less data and fewer parameters. You can train a violin synthesizer on 13 minutes of audio.

### Where Are They Now?

- **Jesse Engel** — still at Google DeepMind, continues working on generative audio
- **Adam Roberts** — co-creator of Magenta, also at Google
- The **Magenta project** is somewhat winding down as Magenta team members move to other things, but the DDSP library is still maintained

### What Spun Off

- **DDSP-VST** — Google built a VST plugin that does timbre transfer in real time using DDSP. You can literally run it in your DAW.
- **realtimeDDSP** (above) — Neutone-compatible DDSP
- **Differentiable synthesizers** became a whole research subfield
- **torchsynth** — differentiable modular synth in PyTorch (below)
- **SynthAX** — differentiable modular synth in JAX (even faster)

### State of the Art in 2026

DDSP has been absorbed into the broader differentiable audio toolkit. The hot stuff now:

1. **DDSP + diffusion** — combining interpretable DSP structure with diffusion models for controllable generation
2. **Neural vocoders** (BigVGAN, HiFi-GAN, EnCodec) — fast, high-quality audio from compact representations
3. **Flow matching** for audio — more stable than diffusion, being applied to synthesis
4. **Latent audio diffusion** (Stable Audio, AudioLDM2) — whole tracks from text prompts

For the parted-record use case, DDSP-style differentiable synthesis is still the most directly relevant approach.

---

## Broader Landscape

### Neural Audio Synthesis / Generation

| Tool | What It Does | Relevance to parted-record |
|------|-------------|---------------------------|
| **MusicGen** (Meta/AudioCraft) | Text-to-music, conditions on audio | Low — generates new music, not matching |
| **AudioLDM2** | Text-to-audio latent diffusion | Low — same |
| **Stable Audio** (Stability AI) | Text/latent-guided audio generation | Low — same |
| **EnCodec** (Meta) | Neural audio codec (compression + tokens) | Medium — useful as perceptual loss |
| **RAVE** | VAE-based real-time neural synthesis | Medium — alternative architecture |
| **BigVGAN** (NVIDIA) | Universal neural vocoder | Medium — high-quality audio decoding |

These are all "generate audio from scratch" tools. Impressive but not your use case. You're doing *inverse synthesis*, not generation.

### Inverse Synthesis / Preset Matching

This is the real frontier and surprisingly sparse:

**[FlowSynth / acids-ircam](https://github.com/acids-ircam/flow_synthesizer)**
- Normalizing flows for universal synthesizer control
- Trained on Diva VST patches
- Maps from audio to parameter space
- MacOS only, a few years old, but the paper is excellent
- Paper: "Universal audio synthesizer control with normalizing flows" (2019)

**SynthBirds / preset transfer systems**
- Various academic projects mapping from audio embeddings to synth params
- Usually train a network on thousands of (param, audio) pairs, then invert

**Neuro-symbolic approaches**
- Treat each synth module as a node, optimize the graph
- Active research area but no killer open-source implementation yet

### Differentiable Synthesizers

These are the key infrastructure for the gradient-based approach:

**[torchsynth](https://github.com/torchsynth/torchsynth)** ⭐
- GPU-optional modular synth in PyTorch
- 16,200x faster than realtime on GPU
- Differentiable — gradients flow through
- Returns parameters alongside audio (great for training)
- Slightly dormant now but solid codebase

**[SynthAX](https://github.com/PapayaResearch/synthax)** ⭐⭐ 
- Modular synth in JAX
- 90,000x faster than realtime (!!!)
- Full differentiability via JAX autograd
- Newer than torchsynth, AES 2023 paper
- The fastest differentiable synth around

**[auraloss](https://github.com/csteinmetz1/auraloss)** ⭐⭐⭐
- Collection of audio loss functions in PyTorch
- MultiResolutionSTFTLoss (MRSTFT) — the go-to for perceptual matching
- Mel-scaled STFT with perceptual weighting
- Essential for any audio matching project

**DawDreamer's FAUST-to-JAX transpiler**
- Write a synth in FAUST → transpile to JAX → it's differentiable
- This is wild and underexplored
- Could make ANY FAUST instrument differentiable

### AI Plugins for DAWs (Producers Actually Using These)

- **Neutone** — free, growing library of neural models as VSTs
- **DDSP-VST** (Google) — timbre transfer in your DAW
- **Ozone** (iZotope) — AI-assisted mastering, very polished, not open-source
- **Udio / Suno** — browser-based music generation, not DAW integration
- **Infinite Drum Machine** (Google) — t-SNE based sample browser, fun
- **Clarity** (Accentize) — AI dialogue enhancement
- **Gullfoss** (Soundtheory) — "intelligent" EQ based on spectral balancing

Nothing has nailed the "hear a sound, get the preset" workflow at a commercial level. That's your opportunity.

---

## Technical Approaches

### Loss Functions for Audio Similarity

**Multi-Resolution STFT Loss (MRSTFT)** — *start here*
```python
# auraloss makes this easy
from auraloss.freq import MultiResolutionSTFTLoss
loss_fn = MultiResolutionSTFTLoss(
    fft_sizes=[512, 1024, 2048],
    hop_sizes=[50, 120, 240],
    win_lengths=[512, 1024, 2048],
    scale="mel",
    n_bins=128,
    sample_rate=44100,
    perceptual_weighting=True
)
```
This is the standard loss in modern neural audio work. Computes STFT at multiple resolutions (captures both transients and texture) and optionally mel-scales for perceptual accuracy. **Use this.**

**MFCC Loss** — classic from Yee-King era
- L2 distance between MFCC vectors
- Works but misses a lot (MFCCs lose phase info, transient detail)
- Good for rough matching, not fine-tuning

**Mel Spectrogram Loss**
- L1 or L2 on mel spectrograms
- Better than raw STFT, worse than MRSTFT
- Computationally cheap

**CDPAM / Perceptual Loss**
- Learned perceptual metrics from contrastive audio representations  
- CDPAM (Contrastive Deep Perceptual Audio Metric) — uses embeddings from a network trained on audio perception
- More "what sounds similar to a human" than "what's spectrally similar"
- Expensive but powerful

**EnCodec Embeddings as Loss**
- Run both sounds through EnCodec encoder, minimize embedding distance
- Surprisingly effective perceptual loss
- Neural codec embeddings capture timbre well

**Onset/Transient Loss**
- If your target has sharp attacks, add a loss on onset locations
- Can compute with librosa.onset.onset_detect on both signals

**Practical recommendation:** Start with MRSTFT + mel spectrogram. Add MFCC as regularization. If you're overfitting, that's fine — overfitting is the goal here.

### Optimizers for Non-Differentiable Parameters

When your VST parameters aren't differentiable (black box):

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
```python
import cma
es = cma.CMAEvolutionStrategy(x0, 0.5)  # x0 = initial params
while not es.stop():
    solutions = es.ask()
    fitnesses = [loss_fn(render(x)) for x in solutions]
    es.tell(solutions, fitnesses)
```
- The Yee-King classic. Works very well for ~50-100 parameters
- Adapts covariance to find the right search direction
- `pip install cma` — easy to use
- **Recommended for initial experiments**

**Bayesian Optimization (BO)**
```python
from bayes_opt import BayesianOptimization
```
- Works well for <20 parameters
- Builds a surrogate model of the loss landscape
- Efficient when evaluations are expensive (long renders)
- Struggles at high dimensions

**SPSA (Simultaneous Perturbation Stochastic Approximation)**
- Gradient estimation without backprop
- Perturb all params simultaneously, estimate gradient
- Scales better than finite differences

**ES (Evolution Strategies) via PyTorch**
```python
# OpenAI-style ES
noise = torch.randn(N, n_params) * sigma
rewards = [loss(render(params + n)) for n in noise]
# Update: params -= lr * (1/N*sigma) * sum(r_i * noise_i)
```
- Parallelizes well across CPU cores
- Works for arbitrary black-box renders
- OpenAI proved this scales to complex problems

**Gradient via DiffProxy approach**
- Use a differentiable synth (torchsynth/SynthAX) to *approximate* your VST
- Optimize parameters via gradient descent against the proxy
- Transfer those parameters to the actual VST
- This is clever and underexplored

**Practical recommendation:** Start with CMA-ES. It's well-understood, easy to tune, and handles synth parameter spaces well (most have ~50-200 parameters). If you have a differentiable proxy available, gradient descent will be much faster.

### When Parameters ARE Differentiable

If you're working with a differentiable synth (torchsynth, SynthAX, DDSP modules):

```python
params = torch.nn.Parameter(torch.rand(n_params))
optimizer = torch.optim.Adam([params], lr=0.01)

for step in range(1000):
    audio = synth(params.sigmoid())  # constrain to [0,1]
    loss = mrstft_loss(audio, target_audio)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This is the fast path. If you can build a differentiable proxy for your target VST, use it.

---

## Headless Rendering on Linux

Here's the real talk on what actually works:

### ✅ DawDreamer — **THE WINNER**
```python
import dawdreamer as daw
engine = daw.RenderEngine(44100, 128)
plugin = engine.make_plugin_processor("my_synth", "/path/to/synth.so")
plugin.load_preset("/path/to/preset.fxp")
plugin.set_parameter(0, 0.5)  # set param by index
plugin.add_midi_note(60, 100, 0, 2.0)  # note, vel, start, duration
engine.load_graph([(plugin, [])])
engine.render(4.0)  # render 4 seconds
audio = plugin.get_audio()  # numpy array
```
- Works on Linux headlessly ✅
- Can load VST3/VST2 plugins built for Linux ✅
- MIDI support ✅
- Parameter automation ✅
- Active maintenance ✅

**Caveat:** You need Linux builds of your VST. Most commercial synths (Serum, Massive, etc.) don't have Linux versions. **Open-source VSTs do:** Surge XT, ZynAddSubFX, OB-Xd, Vital (FOSS version is "Vitalium").

### ✅ Pedalboard
- Same deal, simpler API, better for effects chains
- No MIDI instrument support (effects only)

### ✅ SuperCollider (sclang/scsynth)
- Can run fully headless, powerful, scriptable
- Steep learning curve if you don't know it
- Excellent for synthesis, not great for VST hosting

### ✅ FAUST
```
# faust2lv2 or faust2supercollider — compile to LV2, host in Python
```
- Write synths in FAUST DSL, compile to native code
- DawDreamer can load FAUST processors directly
- And DawDreamer can transpile FAUST → JAX (!!!)

### ⚠️ pyo
- Pure Python audio library, works headlessly
- Good for custom synthesizers, not VST hosting
- Useful for building your own differentiable synth from scratch

### ❌ Most Commercial VSTs on Linux
- Serum: Mac/Windows only. No Linux.
- Massive: No Linux.
- Kontakt: No Linux.
- **Workaround:** Run via Wine + linVST or yabridge (experimental, cursed)

### Best Free VSTs with Linux Builds
- **Surge XT** — incredible free synth, fully open source, Linux native
- **Vital / Vitalium** — wavetable synth, Vitalium is the FOSS fork
- **ZynAddSubFX** — oldschool but powerful, Linux first
- **OB-Xd** — analog-style, has Linux build
- **Helm** — modern subtractive, FOSS, Linux ✅

---

## Toolchain Recommendation

For parted-record right now, here's the stack that makes sense:

```
Target Audio (from Splice, etc.)
        ↓
Feature Extraction (librosa, torchaudio)
        ↓
Optimizer Loop
  ├── [CMA-ES / ES] for black-box VSTs
  └── [Adam + backprop] for differentiable proxies
        ↓
Headless Rendering (DawDreamer)
  ├── Load VST (Surge XT, Vitalium, etc.)
  └── Set params → render → get audio numpy array
        ↓
Loss Function (auraloss MRSTFT)
        ↓
Update Parameters → repeat
        ↓
Export Best Preset
```

**Quick start deps:**
```bash
pip install dawdreamer pedalboard auraloss cma librosa torch torchaudio
```

---

## Papers Worth Reading

1. **DDSP** (Engel et al., 2020) — https://arxiv.org/abs/2001.04643
2. **Universal audio synthesizer control with normalizing flows** (Esling et al., 2019) — https://arxiv.org/abs/1907.00971
3. **torchsynth** (Turian et al., 2021) — https://arxiv.org/abs/2104.12922
4. **SynthAX** (Parker et al., 2023) — https://www.aes.org/e-lib/browse.cfm?elib=22261
5. **auraloss** (Steinmetz & Reiss, 2020) — https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf
6. **RAVE** (Caillon & Esling, 2021) — https://arxiv.org/abs/2111.05011
7. **Matchering** → https://github.com/sergree/matchering (code, not paper)

---

## TL;DR

The 2017 dream of "AI to defeat synthesizers" is now **totally buildable** with modern tooling. The stack exists:
- **DawDreamer** renders VSTs headlessly on Linux
- **auraloss** gives you real perceptual loss functions
- **CMA-ES** works for black-box parameter optimization
- **torchsynth/SynthAX** let you build differentiable proxies for speed
- **DDSP** is the theoretical backbone

The gap in the market: a clean, usable **preset-from-sample** tool. Nobody has shipped this as a nice package. The research exists, the tools exist. Let's build it.
