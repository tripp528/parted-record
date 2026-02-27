"""
The optimizer: search parameter space to minimize audio distance.

This is the brain of parted-record. Given a target sound and a synth,
find the parameters that make the synth sound like the target.

Supports:
  - CMA-ES (proven for black-box synth optimization)
  - Two-phase: random survey → CMA-ES refinement
  - Multi-resolution: fast coarse match → fine match

The key insight from testing: 17D is hard for CMA-ES when parameters
interact heavily. The two-phase approach helps: random search finds a
decent basin, then CMA-ES refines within it.
"""

import numpy as np
import cma
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

from .synth import BaseSynth
from .loss import CompositeLoss, MRSTFTLoss, MelSpectrogramLoss
from .audio import DEFAULT_SR, pad_or_trim


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_params: np.ndarray
    best_loss: float
    n_iterations: int
    n_evaluations: int
    elapsed_seconds: float
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    method: str = ""

    def summary(self) -> str:
        return (
            f"[{self.method}] Best loss: {self.best_loss:.6f} | "
            f"Iters: {self.n_iterations} | "
            f"Evals: {self.n_evaluations} | "
            f"Time: {self.elapsed_seconds:.1f}s | "
            f"Converged: {self.converged}"
        )


class Matcher:
    """
    The core matching engine.

    Takes a target audio signal and a synth, then searches for
    the synth parameters that best reproduce the target sound.
    """

    def __init__(
        self,
        synth: BaseSynth,
        loss_fn: Optional[Any] = None,
        sr: int = DEFAULT_SR,
        midi_note: int = 60,
        duration: float = 2.0,
        verbose: bool = True,
    ):
        self.synth = synth
        self.loss_fn = loss_fn or MRSTFTLoss(sr=sr)
        self.sr = sr
        self.midi_note = midi_note
        self.duration = duration
        self.verbose = verbose
        self._eval_count = 0
        self._target_len = 0

    def _evaluate(self, params: np.ndarray, target: np.ndarray) -> float:
        """Render synth with params and compute loss against target."""
        self._eval_count += 1
        try:
            audio = self.synth.render(
                np.array(params, dtype=np.float32),
                midi_note=self.midi_note,
                duration=self.duration,
                sr=self.sr,
            )
            audio = pad_or_trim(audio, len(target))
            loss = self.loss_fn(audio, target)
            # Guard against NaN/inf
            if not np.isfinite(loss):
                return 1e6
            return loss
        except Exception:
            return 1e6

    def match(
        self,
        target: np.ndarray,
        method: str = "two_phase",
        max_iterations: int = 300,
        population_size: Optional[int] = None,
        sigma: float = 0.3,
        initial_params: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None,
        n_restarts: int = 0,
    ) -> OptimizationResult:
        """
        Find synth parameters that match the target audio.

        Methods:
            "cma" — pure CMA-ES
            "two_phase" — random survey → CMA-ES refinement (recommended)
            "random" — random search baseline
        """
        self._target_len = len(target)

        if method == "two_phase":
            return self._match_two_phase(
                target, max_iterations, population_size,
                sigma, initial_params, callback, n_restarts,
            )
        elif method == "cma":
            return self._match_cma(
                target, max_iterations, population_size,
                sigma, initial_params, callback,
            )
        elif method == "random":
            return self._match_random(target, max_iterations * 20, callback)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'two_phase', 'cma', or 'random'.")

    def _match_two_phase(
        self,
        target: np.ndarray,
        max_iterations: int,
        population_size: Optional[int],
        sigma: float,
        initial_params: Optional[np.ndarray],
        callback: Optional[Callable],
        n_restarts: int,
    ) -> OptimizationResult:
        """
        Two-phase optimization:
        1. Random survey (fast) — sample many random points to find good starting regions
        2. CMA-ES refinement — polish the best candidate

        This avoids CMA-ES getting stuck in a bad basin from the start.
        """
        start_time = time.time()
        self._eval_count = 0
        loss_history = []

        # Phase 1: Random survey
        n_survey = min(500, max(100, self.synth.n_params * 20))
        if self.verbose:
            print(f"  Phase 1: Random survey ({n_survey} samples)...")

        best_survey_params = None
        best_survey_loss = float("inf")
        top_k = []  # Keep top candidates

        for i in range(n_survey):
            params = self.synth.random_params()
            loss = self._evaluate(params, target)
            # Maintain top-10 candidates
            top_k.append((loss, params.copy()))
            if len(top_k) > 10:
                top_k.sort(key=lambda x: x[0])
                top_k = top_k[:10]

            if loss < best_survey_loss:
                best_survey_loss = loss
                best_survey_params = params.copy()

        loss_history.append(best_survey_loss)
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"    Best from survey: {best_survey_loss:.6f} ({elapsed:.1f}s)")

        # Phase 2: CMA-ES from best survey point (+ restarts from other top-k)
        x0 = initial_params if initial_params is not None else best_survey_params
        cma_iters = max_iterations - (n_survey // (population_size or 12))

        if self.verbose:
            print(f"  Phase 2: CMA-ES refinement ({cma_iters} iters, sigma={sigma:.2f})...")

        best_overall = best_survey_loss
        best_overall_params = best_survey_params.copy()

        # Run CMA-ES from best starting point
        result = self._match_cma(
            target, cma_iters, population_size,
            sigma * 0.5,  # Tighter sigma since we have a warm start
            x0, callback,
        )
        loss_history.extend(result.loss_history)

        if result.best_loss < best_overall:
            best_overall = result.best_loss
            best_overall_params = result.best_params.copy()

        # Optional restarts from other top-k candidates
        for restart_i in range(min(n_restarts, len(top_k) - 1)):
            restart_params = top_k[restart_i + 1][1]
            if self.verbose:
                print(f"  Restart {restart_i + 1}/{n_restarts} from candidate (loss={top_k[restart_i+1][0]:.4f})...")
            restart_result = self._match_cma(
                target, cma_iters // 2, population_size,
                sigma * 0.5, restart_params, callback,
            )
            loss_history.extend(restart_result.loss_history)
            if restart_result.best_loss < best_overall:
                best_overall = restart_result.best_loss
                best_overall_params = restart_result.best_params.copy()

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\n  ✅ Two-phase done! Best loss: {best_overall:.6f} in {elapsed:.1f}s "
                  f"({self._eval_count} evaluations)")

        return OptimizationResult(
            best_params=best_overall_params,
            best_loss=best_overall,
            n_iterations=len(loss_history),
            n_evaluations=self._eval_count,
            elapsed_seconds=elapsed,
            loss_history=loss_history,
            converged=result.converged,
            method="two_phase",
        )

    def _match_cma(
        self,
        target: np.ndarray,
        max_iterations: int,
        population_size: Optional[int],
        sigma: float,
        initial_params: Optional[np.ndarray],
        callback: Optional[Callable],
    ) -> OptimizationResult:
        """CMA-ES optimization."""
        n = self.synth.n_params
        x0 = initial_params if initial_params is not None else np.full(n, 0.5)
        eval_start = self._eval_count

        opts = {
            "maxiter": max_iterations,
            "bounds": [0, 1],
            "verbose": -9,
            "tolfun": 1e-6,
            "tolx": 1e-6,
        }
        if population_size:
            opts["popsize"] = population_size

        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma, opts)

        loss_history = []
        start_time = time.time()
        iteration = 0

        while not es.stop():
            solutions = es.ask()
            solutions_clipped = [np.clip(s, 0, 1) for s in solutions]
            fitnesses = [self._evaluate(s, target) for s in solutions_clipped]
            es.tell(solutions, fitnesses)

            best_loss = es.result.fbest
            loss_history.append(best_loss)
            iteration += 1

            if self.verbose and iteration % 20 == 0:
                elapsed = time.time() - start_time
                evals = self._eval_count - eval_start
                print(
                    f"    iter {iteration:4d} | "
                    f"loss {best_loss:.6f} | "
                    f"evals {evals:5d} | "
                    f"σ {es.sigma:.4f}"
                )

            if callback:
                callback(iteration, best_loss, np.clip(es.result.xbest, 0, 1))

        elapsed = time.time() - start_time
        best_params = np.clip(es.result.xbest, 0, 1).astype(np.float32)

        return OptimizationResult(
            best_params=best_params,
            best_loss=float(es.result.fbest),
            n_iterations=iteration,
            n_evaluations=self._eval_count - eval_start,
            elapsed_seconds=elapsed,
            loss_history=loss_history,
            converged="tolfun" in es.stop() or "tolx" in es.stop(),
            method="cma",
        )

    def _match_random(
        self,
        target: np.ndarray,
        n_trials: int,
        callback: Optional[Callable],
    ) -> OptimizationResult:
        """Random search baseline."""
        self._eval_count = 0
        start_time = time.time()

        best_params = None
        best_loss = float("inf")
        loss_history = []

        for i in range(n_trials):
            params = self.synth.random_params()
            loss = self._evaluate(params, target)

            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()

            if self.verbose and (i + 1) % 200 == 0:
                print(f"    trial {i+1:5d} | best loss {best_loss:.6f}")

            loss_history.append(best_loss)
            if callback:
                callback(i, best_loss, best_params)

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"\n  ✅ Random search done! Best loss: {best_loss:.6f} in {elapsed:.1f}s")

        return OptimizationResult(
            best_params=best_params,
            best_loss=best_loss,
            n_iterations=n_trials,
            n_evaluations=self._eval_count,
            elapsed_seconds=elapsed,
            loss_history=loss_history,
            converged=False,
            method="random",
        )


def match_sound(
    target: np.ndarray,
    synth_name: str = "subtractive",
    midi_note: int = 60,
    duration: float = 2.0,
    sr: int = DEFAULT_SR,
    max_iterations: int = 300,
    n_restarts: int = 2,
    verbose: bool = True,
) -> OptimizationResult:
    """
    High-level convenience function: match a target sound.

    Example:
        from ptrd.audio import load
        from ptrd.optimizer import match_sound
        from ptrd.synth import get_synth

        target, sr = load("my_sample.wav")
        result = match_sound(target, synth_name="subtractive")
        synth = get_synth("subtractive")
        matched_audio = synth.render(result.best_params)
    """
    from .synth import get_synth

    synth = get_synth(synth_name)
    loss_fn = MRSTFTLoss(sr=sr)
    matcher = Matcher(synth, loss_fn, sr=sr, midi_note=midi_note,
                      duration=duration, verbose=verbose)
    return matcher.match(target, method="two_phase",
                         max_iterations=max_iterations,
                         n_restarts=n_restarts)
