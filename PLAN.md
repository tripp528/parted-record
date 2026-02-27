# 🎛️ parted-record — Project Plan

_Last updated: 2026-02-27_

## What Is This?

You find a sick synth sound on Splice. You want to use it in your track, but transposing the sample kills the magic. What if you could reverse-engineer the sound — figure out exactly which VST knobs to turn to recreate it? Then you'd have a fully playable, transposable version.

**parted-record** = an optimizer that matches a target audio sample by tuning VST/synth parameters. Each knob is a trainable parameter. Overfit on one sound. Get the preset.

## Current State

### What Exists
- `scripts/bin/spotify-download` — download tracks via spotdl
- `scripts/bin/separate` — stem separation via Demucs
- `scripts/bin/part-record` — download + separate pipeline
- `ptrd/` — empty Python package (just `print('hello')`)

### What's Needed
Everything interesting. See phases below.

## Phases

### Phase 0: Research ✅ (in progress)
- [x] Clone and audit existing repo
- [x] Research all original README links (DawDreamer, DDSP, RAVE, etc.)
- [x] Survey 2026 AI audio landscape
- [ ] Review research findings and pick approach
- **Output:** `research/README.md`, `research/IDEAS.md`

### Phase 1: Audio Similarity Engine
Build the "ears" — a module that takes two audio clips and returns a similarity score.
- [ ] Implement multi-scale STFT loss
- [ ] Implement MFCC-based comparison
- [ ] Implement perceptual/embedding-based loss (e.g., CLAP, VGGish)
- [ ] Benchmarking: which loss function best matches human perception of "same sound"?
- **Output:** `ptrd/loss.py` with pluggable loss functions

### Phase 2: Synth Rendering (Headless)
Get a synth we can control programmatically on Linux.
- [ ] Evaluate: DawDreamer vs pyo vs SuperCollider vs FAUST vs built-in Python synth
- [ ] Build a `Synth` abstraction: `synth.set_params(dict) → synth.render(midi_note, duration) → np.array`
- [ ] Define parameter spaces (continuous knobs, discrete switches, ranges)
- **Output:** `ptrd/synth.py` with at least one working backend

### Phase 3: The Optimizer
The brain — search the parameter space to minimize audio distance.
- [ ] Start simple: random search baseline
- [ ] Evolutionary strategy (CMA-ES via `cmaes` or `nevergrad`)
- [ ] Bayesian optimization (`optuna` or `ax-platform`)
- [ ] If synth is differentiable: gradient-based (Adam on a differentiable synth)
- [ ] Logging: track optimization runs, save best presets
- **Output:** `ptrd/optimizer.py`

### Phase 4: CLI & UX
Make it usable for a musician, not just a researcher.
- [ ] `ptrd match <target_audio> --synth <synth_name>` → outputs preset
- [ ] Progress visualization (loss curve, audio comparisons at checkpoints)
- [ ] A/B listening: play target vs current best side-by-side
- [ ] Export preset in synth-native format if possible
- **Output:** `scripts/bin/match`, updated `pyproject.toml`

### Phase 5: Fun Extras
- [ ] Batch mode: match a whole folder of samples
- [ ] Web UI for drag-and-drop matching
- [ ] Integration with Splice API (if it exists)
- [ ] Multi-synth: try matching across different synths, pick best fit
- [ ] "Sound DNA" — embed sounds in a latent space, explore neighborhoods

## Architecture (Planned)

```
ptrd/
├── __init__.py
├── loss.py          # Audio similarity functions
├── synth.py         # Synth rendering abstraction
├── optimizer.py     # Parameter search algorithms
├── audio.py         # Audio I/O, resampling, normalization
├── utils.py         # Logging, config, helpers
└── cli.py           # Click/Typer CLI entry points

research/
├── README.md        # Deep dive research doc
└── IDEAS.md         # Brainstorms and future directions

scripts/bin/
├── part-record      # Original: download + separate
├── spotify-download # Original: spotdl wrapper
├── separate         # Original: demucs wrapper
└── match            # New: the main event
```

## Tech Decisions (TBD)
- **Python version:** Upgrade from 3.9.1 → 3.11+ (for speed + modern features)
- **Package manager:** Poetry (existing) or switch to uv/pixi?
- **Synth backend:** Depends on research — DawDreamer is the frontrunner
- **Optimizer:** Start with CMA-ES (proven for black-box optimization)
- **Loss function:** Multi-scale STFT + perceptual embedding hybrid likely best

## Open Questions
1. Can DawDreamer load VST3 plugins headless on Linux?
2. Should we start with a simple Python synth (full control, differentiable) before going to real VSTs?
3. What's the right audio representation for the loss function?
4. How do we handle time-varying sounds (pads, plucks with long tails)?
5. Can we leverage pre-trained audio models (CLAP, EnCodec) for perceptual loss?

## Links
- [DawDreamer](https://github.com/DBraun/DawDreamer) — headless DAW in Python
- [Nevergrad](https://github.com/facebookresearch/nevergrad) — gradient-free optimization
- [DDSP](https://github.com/magenta/ddsp) — differentiable signal processing
- [RAVE](https://github.com/acids-ircam/RAVE) — real-time audio variational autoencoder
- Full research: `research/README.md`
