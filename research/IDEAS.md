# parted-record: Concrete Ideas & Next Steps

> The "oh HELL yeah" list. Ranked by excitement and feasibility.

---

## 🔥 Idea 1: The Core Thing — CMA-ES Preset Matcher (MVP)

**What:** Drop an audio sample in, get a synth preset out. That's it.

**Stack:**
- **DawDreamer** — headless VST rendering on Linux
- **Surge XT** — free, open-source, great-sounding, Linux native, 200+ parameters to tune
- **CMA-ES** — `pip install cma`, dead simple API
- **auraloss MRSTFT** — perceptual loss

**How it works:**
```python
import cma, dawdreamer as daw, numpy as np
from auraloss.freq import MultiResolutionSTFTLoss
import torch, librosa

# Load target
target_audio, sr = librosa.load("splice_sample.wav", sr=44100)
target_tensor = torch.tensor(target_audio).unsqueeze(0).unsqueeze(0)

# Setup DawDreamer + Surge XT
engine = daw.RenderEngine(sr, 256)
synth = engine.make_plugin_processor("surge", "/path/to/SurgeXT.so")
n_params = synth.get_num_parameters()  # usually ~200+

# Loss function
mrstft = MultiResolutionSTFTLoss(fft_sizes=[512, 1024, 2048], ...)

def render_and_loss(params):
    for i, p in enumerate(params):
        synth.set_parameter(i, float(p))
    synth.clear_midi()
    synth.add_midi_note(60, 100, 0, 2.0)  # C4, 2 seconds
    engine.render(2.0)
    audio = torch.tensor(synth.get_audio()[0]).unsqueeze(0).unsqueeze(0)
    return mrstft(audio, target_tensor).item()

# CMA-ES optimization
es = cma.CMAEvolutionStrategy([0.5] * n_params, 0.3, {'maxiter': 500})
while not es.stop():
    solutions = es.ask()
    solutions_clipped = [np.clip(s, 0, 1) for s in solutions]
    fitnesses = [render_and_loss(s) for s in solutions_clipped]
    es.tell(solutions, fitnesses)

best_params = es.result.xbest
```

**Why this first:** CMA-ES is well-understood, works, and gives you a baseline to beat. No ML training needed. Pure optimization.

**Timeline:** Weekend project. Seriously.

---

## 🔥 Idea 2: Differentiable Proxy Speedup

**What:** Train a small neural net to *emulate* the VST, then optimize through the emulator with gradient descent. Much faster than CMA-ES once the emulator is trained.

**How:**
1. Generate N random patches → render them with DawDreamer → get (params, audio) pairs
2. Train a tiny MLP or CNN: `params → mel spectrogram`
3. Now you have a differentiable proxy
4. Use Adam to optimize params through the proxy
5. Transfer best params back to the real VST for final render

```python
# Training the proxy
class SynthProxy(nn.Module):
    def __init__(self, n_params, n_mel_bins=128, n_frames=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_params, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(), 
            nn.Linear(1024, n_mel_bins * n_frames)
        )
    
    def forward(self, params):
        return self.net(params).reshape(-1, 1, n_mel_bins, n_frames)

# Then optimize:
params = torch.nn.Parameter(torch.rand(1, n_params))
opt = torch.optim.Adam([params], lr=0.01)
for step in range(2000):
    pred_mel = proxy(params.sigmoid())
    loss = F.mse_loss(pred_mel, target_mel)
    opt.zero_grad(); loss.backward(); opt.step()
```

**Speedup:** 100-1000x over CMA-ES once proxy is trained. The proxy training is the upfront cost.

**Key challenge:** The proxy needs to generalize across the parameter space. With 200 params, you might need 50k-100k (param, audio) pairs. DawDreamer can render these fast with multiprocessing.

---

## 🔥 Idea 3: FAUST → JAX Pipeline (The Nuclear Option)

**What:** Write the synth in FAUST, use DawDreamer's FAUST-to-JAX transpiler, get a fully JIT-compiled, GPU-accelerated, differentiable synth. Then optimize via gradient descent.

**Why this is insane:** DawDreamer can literally transpile FAUST code to JAX, which means you can run the synth on a GPU with full autograd support. SynthAX showed you can run 90,000x faster than realtime. Optimization that would take hours with CMA-ES could take seconds.

**Relevant resources:**
- DawDreamer's [Faust-to-JAX examples](https://github.com/DBraun/DawDreamer/tree/main/examples/Faust_to_JAX)
- DawDreamer's [QDax integration](https://github.com/DBraun/DawDreamer/tree/main/examples/Faust_to_QDax) (quality-diversity search in JAX)
- SynthAX: https://github.com/PapayaResearch/synthax

**Challenge:** You'd need to implement the target synth in FAUST. This limits you to synths you build yourself or ones with open-source FAUST implementations.

**When to do this:** After proving CMA-ES works. This is the v2 speedup.

---

## 🔥 Idea 4: Multi-Note Optimization

**What:** Most sounds aren't just one note. Optimize across *multiple* MIDI notes simultaneously.

**Why:** A synth might sound exactly right at C4 but weird at C2 (different filter response, different envelope). True preset matching means it sounds right across the whole keyboard.

**How:** Change the loss to be the sum over multiple renders:
```python
notes = [48, 60, 72]  # C3, C4, C5
total_loss = sum(render_and_loss(params, note) for note in notes)
```

Also: match the *velocity response*. A good preset should react to MIDI velocity the same way the original sample would (if the original had dynamics).

---

## 💡 Idea 5: The Splice-Aware Workflow

**What:** Build the actual UX around the real use case: you downloaded a Splice sample, you want to play it melodically.

**Workflow:**
1. Drop in sample (any WAV)
2. Tool auto-detects: "This sounds like a [pad/lead/bass/pluck]"
3. Picks the best synth archetype (Surge XT preset category)
4. Runs optimization
5. Outputs: `.fxp` preset file + preview renders at C2/C3/C4/C5

**Secret sauce:** Pre-filter the parameter search space by preset category. Don't search all 200 Surge parameters for a pad — start within the "pad" neighborhood of preset space. CMA-ES from a warm start.

---

## 💡 Idea 6: Sound Type Classifier as Pre-Filter

**What:** Use a pretrained audio classifier to detect the sound type before optimization.

```python
# Using PANNs (Pretrained Audio Neural Networks) or CLAP embeddings
import laion_clap
model = laion_clap.CLAP_Module(enable_fusion=False)
embedding = model.get_audio_embedding_from_data(target_audio)
# Compute similarity to text descriptions: "warm pad", "aggressive lead", "bass pluck"
# Use top match to pre-initialize CMA-ES parameters
```

**LAION-CLAP:** https://github.com/LAION-AI/CLAP — pretrained CLAP model, maps audio and text to shared embedding space. Free, runs locally.

---

## 💡 Idea 7: DDSP as Universal Proxy

**What:** Instead of a learned emulator (Idea 2), use DDSP's harmonic + noise model as a universal proxy.

**Why this is elegant:** DDSP is a *physically motivated* synthesis model. Harmonic oscillators + filtered noise can approximate most acoustic and many electronic sounds. You can fit it to any target in ~100 gradient steps.

**Approach:**
1. Fit DDSP to target: optimize DDSP parameters via backprop
2. Look at what DDSP learned (harmonic content, noise coloring, envelope)
3. Map those characteristics to VST parameters via lookup table or another small neural net

**Reference:** realtimeDDSP repo: https://github.com/hyakuchiki/realtimeDDSP

---

## 💡 Idea 8: Batch Mode for Sample Packs

**What:** Process an entire Splice pack (100+ samples) overnight, output a folder of matched presets.

**Why:** The real producer workflow isn't one sample — it's "I want to pull the best sounds from this whole pack." Batch it.

**Implementation:** Multiprocessing with DawDreamer (already has multiprocessing support), store results as JSON + .fxp files. Maybe a little web UI to browse results.

---

## 🔭 Idea 9: Neutone Export

**What:** Take your trained differentiable proxy (from Idea 2 or 7), export it as a Neutone model, run it as a VST in Ableton.

**Why this is sick:** You've essentially built a *neural instrument* that sounds like your target. It responds to MIDI pitch and velocity, transposes perfectly, and runs in your DAW.

**Reference:** 
- Neutone SDK: https://github.com/QosmoInc/neutone_sdk
- realtimeDDSP exports to Neutone

---

## 🛠️ Immediate Next Steps (Do These This Weekend)

### Step 1: Environment Setup
```bash
# On Linux (carl works)
pip install dawdreamer pedalboard auraloss cma librosa torch torchaudio
# Download Surge XT Linux VST3: https://surge-synthesizer.github.io/
```

### Step 2: Baseline Test
```python
# Verify DawDreamer + Surge XT works headlessly
import dawdreamer as daw
engine = daw.RenderEngine(44100, 256)
synth = engine.make_plugin_processor("surge", "./SurgeXT.vst3")
synth.add_midi_note(60, 100, 0, 2.0)
engine.load_graph([(synth, [])])
engine.render(2.0)
audio = synth.get_audio()
print(f"Rendered {len(audio[0])} samples")
# Should print: Rendered 88200 samples
```

### Step 3: Loss Function Test
```python
# Make sure auraloss works with your audio shapes
from auraloss.freq import MultiResolutionSTFTLoss
import torch
mrstft = MultiResolutionSTFTLoss(fft_sizes=[512, 1024, 2048], hop_sizes=[256, 512, 1024], win_lengths=[512, 1024, 2048])
a = torch.rand(1, 1, 88200)
b = torch.rand(1, 1, 88200)
print(mrstft(a, b))  # Should print a tensor with loss value
```

### Step 4: CMA-ES Loop (the fun part)
Run the MVP from Idea 1 on a known target — first generate a random Surge patch, save the parameters, then see if CMA-ES can recover them. This validates the approach before trying to match real-world samples.

---

## Open Questions

1. **Parameter normalization:** Surge XT parameters aren't all linear. Some are log-scale (frequency), some are discrete (oscillator type). How do you handle this in CMA-ES?
   - Answer: Normalize all to [0,1], let DawDreamer translate back. Discrete params: round to nearest valid value.

2. **Note duration matching:** If your sample is a 0.5s pluck, should you optimize over 0.5s or longer? Probably match the sample length exactly.

3. **Stereo vs mono:** Most synths produce stereo. Mix to mono before computing loss, or optimize each channel?

4. **Envelope matching:** The ADSR of the target is crucial. Should you add a specific envelope loss on top of MRSTFT?

5. **Dataset for proxy training:** How many renders do you need? Start with 10k, see if loss curve flattens.

---

## Resources to Bookmark

- **DawDreamer docs:** https://dbraun.github.io/DawDreamer/
- **auraloss:** https://github.com/csteinmetz1/auraloss
- **CMA-ES Python:** https://github.com/CMA-ES/pycma
- **Surge XT:** https://surge-synthesizer.github.io/
- **torchsynth:** https://github.com/torchsynth/torchsynth
- **SynthAX:** https://github.com/PapayaResearch/synthax
- **LAION-CLAP:** https://github.com/LAION-AI/CLAP
- **Neutone SDK:** https://github.com/QosmoInc/neutone_sdk
- **FlowSynth paper:** https://arxiv.org/abs/1907.00971
- **DDSP paper:** https://arxiv.org/abs/2001.04643
