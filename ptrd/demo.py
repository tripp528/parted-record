"""
Demo / visualization tools for parted-record.

Generate comparison pages, audio previews, and optimization reports.
Outputs static HTML that can be hosted anywhere.
"""

import numpy as np
import base64
import io
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from .audio import save, audio_stats, DEFAULT_SR
from .synth import BaseSynth
from .optimizer import OptimizationResult


def audio_to_data_uri(audio: np.ndarray, sr: int = DEFAULT_SR) -> str:
    """Convert numpy audio to a base64 WAV data URI for embedding in HTML."""
    import struct

    # Convert to 16-bit PCM
    audio_16 = np.clip(audio, -1, 1)
    audio_16 = (audio_16 * 32767).astype(np.int16)

    # Build WAV in memory
    buf = io.BytesIO()
    n_samples = len(audio_16)
    data_size = n_samples * 2
    # RIFF header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    # fmt chunk
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))  # chunk size
    buf.write(struct.pack('<H', 1))   # PCM
    buf.write(struct.pack('<H', 1))   # mono
    buf.write(struct.pack('<I', sr))  # sample rate
    buf.write(struct.pack('<I', sr * 2))  # byte rate
    buf.write(struct.pack('<H', 2))   # block align
    buf.write(struct.pack('<H', 16))  # bits per sample
    # data chunk
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio_16.tobytes())

    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:audio/wav;base64,{b64}"


def generate_comparison_page(
    results: List[Dict],
    title: str = "parted-record — Match Results",
    output_path: str = "results.html",
) -> str:
    """
    Generate a static HTML page comparing target vs matched audio.

    results: list of dicts with keys:
        - name: str
        - target_audio: np.ndarray
        - matched_audio: np.ndarray
        - result: OptimizationResult
        - synth: BaseSynth (optional)
        - sr: int (optional)

    Returns the path to the generated HTML file.
    """

    cards_html = []
    for r in results:
        sr = r.get("sr", DEFAULT_SR)
        target_uri = audio_to_data_uri(r["target_audio"], sr)
        matched_uri = audio_to_data_uri(r["matched_audio"], sr)
        opt = r["result"]

        # Loss history as JSON for sparkline
        history_json = json.dumps(opt.loss_history[-50:])

        # Param table if synth provided
        param_html = ""
        if "synth" in r and "true_params" in r:
            synth = r["synth"]
            true_p = r["true_params"]
            found_p = opt.best_params
            rows = []
            for i, spec in enumerate(synth.param_specs):
                err = abs(true_p[i] - found_p[i])
                color = "#2ff7c3" if err < 0.1 else "#f7c32f" if err < 0.3 else "#f72f8e"
                rows.append(
                    f'<tr><td>{spec.name}</td>'
                    f'<td>{true_p[i]:.3f}</td>'
                    f'<td>{found_p[i]:.3f}</td>'
                    f'<td style="color:{color}">{err:.3f}</td></tr>'
                )
            param_html = f"""
            <details>
                <summary style="cursor:pointer;color:#8888a0;font-size:0.85rem;margin-top:1rem;">
                    Parameter details ▸
                </summary>
                <table style="width:100%;font-size:0.8rem;margin-top:0.5rem;border-collapse:collapse;">
                    <tr style="color:#8888a0;"><th align="left">Param</th><th>True</th><th>Found</th><th>Error</th></tr>
                    {''.join(rows)}
                </table>
            </details>"""

        card = f"""
        <div class="card">
            <h3>{r['name']}</h3>
            <div class="audio-row">
                <div class="audio-box">
                    <div class="label">🎯 Target</div>
                    <audio controls src="{target_uri}" preload="auto"></audio>
                </div>
                <div class="audio-box">
                    <div class="label">🎛️ Matched</div>
                    <audio controls src="{matched_uri}" preload="auto"></audio>
                </div>
            </div>
            <div class="stats">
                <span>Loss: <b>{opt.best_loss:.4f}</b></span>
                <span>Evals: <b>{opt.n_evaluations}</b></span>
                <span>Time: <b>{opt.elapsed_seconds:.1f}s</b></span>
                <span>Method: <b>{opt.method}</b></span>
            </div>
            <canvas class="sparkline" data-values='{history_json}' width="400" height="60"></canvas>
            {param_html}
        </div>"""
        cards_html.append(card)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600&family=JetBrains+Mono:wght@400&display=swap');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
        font-family: 'Space Grotesk', sans-serif;
        background: #0a0a0f;
        color: #e8e8f0;
        padding: 2rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }}
    h1 {{
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #e8e8f0, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .subtitle {{ color: #8888a0; margin-bottom: 2rem; font-size: 0.95rem; }}
    .card {{
        background: #16162a;
        border: 1px solid #2a2a44;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .card h3 {{ font-size: 1.1rem; margin-bottom: 1rem; }}
    .audio-row {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }}
    .audio-box {{
        flex: 1;
        min-width: 200px;
    }}
    .audio-box .label {{
        font-size: 0.8rem;
        color: #8888a0;
        margin-bottom: 0.3rem;
    }}
    audio {{
        width: 100%;
        height: 36px;
        border-radius: 8px;
    }}
    .stats {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #8888a0;
    }}
    .stats b {{ color: #2ff7c3; }}
    .sparkline {{
        width: 100%;
        height: 60px;
        margin-top: 1rem;
    }}
    table td, table th {{
        padding: 0.2rem 0.5rem;
        border-bottom: 1px solid #2a2a44;
    }}
    details summary:hover {{ color: #7b2ff7 !important; }}
</style>
</head>
<body>
<h1>🎛️ {title}</h1>
<p class="subtitle">A/B comparisons — target vs optimizer output. Listen and judge.</p>
{''.join(cards_html)}
<script>
document.querySelectorAll('.sparkline').forEach(canvas => {{
    const ctx = canvas.getContext('2d');
    const values = JSON.parse(canvas.dataset.values);
    if (!values.length) return;
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 120;
    canvas.style.height = '60px';
    ctx.scale(2, 2);
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;
    const stepX = (w/2) / (values.length - 1);
    ctx.strokeStyle = '#7b2ff7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {{
        const x = i * stepX;
        const y = 55 - ((v - min) / range) * 50;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
    ctx.fillStyle = '#8888a0';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(max.toFixed(3), 2, 10);
    ctx.fillText(min.toFixed(3), 2, 58);
}});
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    return str(Path(output_path).resolve())
