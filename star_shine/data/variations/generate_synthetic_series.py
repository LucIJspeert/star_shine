#!/usr/bin/env python3
"""Generate a synthetic positive time series with several controlled sinusoidal behaviors."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median


def smooth_step(x: float) -> float:
    """Map a scalar smoothly from 0 to 1."""
    return 0.5 * (1.0 + math.tanh(x))


def generate_series(
    duration: float = 20.0,
    n_samples: int = 4000,
):
    """Return time, flux, and the individual components."""
    time = [duration * index / n_samples for index in range(n_samples)]
    dt = duration / n_samples

    # Pure sinusoid at f1.
    f1, a1, p1 = 1.15, 0.18, 0.3
    s1 = [a1 * math.sin(math.tau * f1 * t + p1) for t in time]

    # Sinusoid at f2 with two sudden phase shifts.
    f2, a2, p2 = 2.05, 0.16, -0.8
    t_jump_1 = 0.33 * duration
    t_jump_2 = 0.68 * duration
    s2 = [
        a2
        * math.sin(
            math.tau * f2 * t
            + p2
            + (1.2 if t >= t_jump_1 else 0.0)
            + (-1.0 if t >= t_jump_2 else 0.0)
        )
        for t in time
    ]

    # Sinusoid at f3 with a gradual phase shift.
    f3, a3, p3 = 3.1, 0.14, 1.1
    s3 = []
    for t in time:
        phase_ramp = 1.4 * smooth_step((t - 0.5 * duration) / (0.06 * duration))
        s3.append(a3 * math.sin(math.tau * f3 * t + p3 + phase_ramp))

    # Sinusoid at f4 with amplitude modulation at another frequency.
    f4, a4, p4 = 4.25, 0.12, -0.2
    f_amp, amp_depth, p_amp = 0.37, 0.55, 0.6
    s4 = []
    for t in time:
        amp_mod = 1.0 + amp_depth * math.sin(math.tau * f_amp * t + p_amp)
        s4.append(a4 * amp_mod * math.sin(math.tau * f4 * t + p4))

    # Sinusoid at f5 with a small frequency wander within a band around f5.
    f5, a5, p5 = 5.3, 0.10, 0.2
    f_wander, df5, p_wander = 0.21, 0.10, -0.5
    phase5 = p5
    s5 = []
    for t in time:
        f5_inst = f5 + df5 * math.sin(math.tau * f_wander * t + p_wander)
        phase5 += math.tau * f5_inst * dt
        s5.append(a5 * math.sin(phase5))

    signal = [sum(values) for values in zip(s1, s2, s3, s4, s5)]

    # Keep the series positive and set the median to about one.
    flux = [1.0 + 0.22 * value for value in signal]
    min_flux = min(flux)
    if min_flux <= 0.0:
        flux = [value - min_flux + 0.05 for value in flux]
    med_flux = median(flux)
    flux = [value / med_flux for value in flux]

    components = {
        "pure_f1": s1,
        "phase_jump_f2": s2,
        "gradual_phase_f3": s3,
        "amp_mod_f4": s4,
        "freq_wander_f5": s5,
    }
    return time, flux, components


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("synthetic_series.csv"))
    parser.add_argument("--plot", action="store_true", help="Show a quick plot")
    args = parser.parse_args()

    time, flux, components = generate_series()
    flux_err = [0.01] * len(time)

    def write_series(path: Path, values: list[float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "flux", "flux_err"])
            writer.writerows(zip(time, values, flux_err))

    write_series(args.out, flux)
    for name, values in components.items():
        component_out = args.out.with_name(f"{args.out.stem}_{name}{args.out.suffix}")
        write_series(component_out, values)

    print(f"Saved {len(time)} samples to {args.out}")
    print(f"Median flux: {median(flux):.6f}")
    print(f"Minimum flux: {min(flux):.6f}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(time, flux, color="black", lw=1.0)
        axes[0].set_ylabel("flux")
        axes[0].set_title("Synthetic positive time series")

        for axis, (name, values) in zip(axes[1:], components.items()):
            axis.plot(time, values, lw=0.9)
            axis.set_ylabel(name)

        axes[-1].set_xlabel("time")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
