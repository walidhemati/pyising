"""Hero animation: domain coarsening after a quench from T = infinity to T < T_c."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import backends as B


def quench_gif(out: str | Path, L: int = 1024, T: float = 1.5, frames: int = 60,
               backend: str = "cupy-cuda-c", variant: str = "msc", size: int = 512,
               seed: int = 3, t_max: int = 800) -> Path:
    """Write an animated GIF of a ``size x size`` window; frames log-spaced in time up to
    ``t_max`` sweeps (domains grow like sqrt(t), so the lattice never fully orders)."""
    from PIL import Image

    cls = B.load(backend, variant) or B.load("numpy", "int8")
    sim = cls((L, L), 1.0 / T, seed=seed, init="hot")
    times = np.unique(np.round(np.geomspace(1, t_max, frames)).astype(int))
    pal = [24, 28, 48, 236, 232, 222]  # spin down: ink, spin up: paper
    imgs = []
    done = 0
    for t in times:
        sim.sweep(int(t - done))
        done = int(t)
        lat = sim.get_lattice()[:size, :size]
        im = Image.fromarray(((lat > 0).astype(np.uint8)), mode="P")
        im.putpalette(pal + [0] * (768 - len(pal)))
        imgs.append(im)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [80] * (len(imgs) - 1) + [1500]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durations, loop=0,
                 optimize=True)
    return out
