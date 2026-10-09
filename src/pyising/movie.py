"""Hero animation: domain coarsening after a quench from T = infinity to T < T_c."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import backends as B


def quench_gif(out: str | Path, L: int = 512, T: float = 1.5, frames: int = 60,
               backend: str = "cupy-cuda-c", variant: str = "msc", size: int = 384,
               seed: int = 3) -> Path:
    """Write an animated GIF; frame ``k`` is taken after ~``1.12^k`` sweeps (log time)."""
    from PIL import Image

    cls = B.load(backend, variant) or B.load("numpy", "int8")
    sim = cls((L, L), 1.0 / T, seed=seed, init="hot")
    times = np.unique(np.round(1.13 ** np.arange(frames * 2)).astype(int))[:frames]
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
