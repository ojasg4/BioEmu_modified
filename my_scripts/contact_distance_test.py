"""Does contact conditioning act as a sharp cutoff, a weak shift, or nothing?

Samples the A2A + CLR system (a2a_active_contacts.yaml) under three conditions with
the same seed, then measures every contact pair's distance in every sample:

  none      no contacts (baseline)
  yaml      the 5 contacts as written (8-9 A)
  tight4    the same 5 residue pairs, all max_distance = 4.0 A

Two distances per pair, because the token reference atom is not documented:
  CA-CA          alpha-carbon distance
  min-heavy      closest side-chain-or-backbone heavy-atom distance between the residues

Outputs (my_scripts/contact_test/): <cond>/ sample dirs, distances.npz,
hist_<pair>.png (one per pair, conditions overlaid), summary.txt, run.log.
Run: python my_scripts/contact_distance_test.py [--analyze-only]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "contact_test"
YAML = Path(__file__).parent / "a2a_active_contacts.yaml"
NUM_SAMPLES, BATCH_SIZE, NUM_STEPS, SEED = 16, 8, 100, 0
CONDITIONS = ("none", "yaml", "tight4")


def sample_all():
    from bioemu2 import Contact, load_model, load_system_and_contacts, sample_equilibrium

    system, yaml_contacts = load_system_and_contacts(YAML)
    contact_sets = {
        "none": None,
        "yaml": yaml_contacts,
        "tight4": [
            Contact(token1=c.token1, token2=c.token2, max_distance=4.0) for c in yaml_contacts
        ],
    }
    model = load_model()
    for name in CONDITIONS:
        out = OUT / name
        if (out / "samples.xtc").exists():
            print(f"{name}: already sampled, skipping", flush=True)
            continue
        print(f"{name}: sampling {NUM_SAMPLES} (batch {BATCH_SIZE}, {NUM_STEPS} steps)", flush=True)
        sample_equilibrium(
            system, model=model, contacts=contact_sets[name], num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE, num_steps=NUM_STEPS, seed=SEED, output_dir=out,
        )


def analyze():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mdtraj
    import yaml

    pairs = [(c["token1"]["res_id"], c["token2"]["res_id"], c["max_distance"])
             for c in yaml.safe_load(YAML.read_text())["contacts"]]
    data = {}  # (cond, pair index, metric) -> distances in Angstrom
    for name in CONDITIONS:
        traj = mdtraj.load(str(OUT / name / "samples.xtc"), top=str(OUT / name / "topology.cif"))
        prot = traj.topology.select("protein")
        res = {r.resSeq: r for r in traj.topology.residues if r.chain.index == 0}
        for i, (a, b, _) in enumerate(pairs):
            ra, rb = res[a], res[b]
            ca = [next(x.index for x in r.atoms if x.name == "CA") for r in (ra, rb)]
            data[name, i, "CA-CA"] = 10 * mdtraj.compute_distances(traj, [ca])[:, 0]
            ha = [x.index for x in ra.atoms if x.element.symbol != "H"]
            hb = [x.index for x in rb.atoms if x.element.symbol != "H"]
            grid = np.array([(p, q) for p in ha for q in hb])
            data[name, i, "min-heavy"] = 10 * mdtraj.compute_distances(traj, grid).min(axis=1)
        print(f"{name}: {traj.n_frames} frames, residues check "
              f"{[res[a].name + str(a) for a, _, _ in pairs]}", flush=True)
    np.savez(OUT / "distances.npz", **{f"{c}|{i}|{m}": v for (c, i, m), v in data.items()})

    colors = {"none": "#898781", "yaml": "#2a78d6", "tight4": "#eb6834"}
    lines = ["pair (script res_id) / metric / condition: mean, median, min-max, frac <= max_distance"]
    for i, (a, b, maxd) in enumerate(pairs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), squeeze=False)
        for ax, metric in zip(axes[0], ("CA-CA", "min-heavy")):
            for name in CONDITIONS:
                d = data[name, i, metric]
                ax.hist(d, bins=np.linspace(0, 40, 41), alpha=0.55, color=colors[name], label=name)
                thr = {"none": None, "yaml": maxd, "tight4": 4.0}[name]
                frac = f"{(d <= thr).mean():.2f}" if thr else "n/a"
                lines.append(f"res {a}-{b} / {metric} / {name}: mean {d.mean():.1f}, "
                             f"median {np.median(d):.1f}, {d.min():.1f}-{d.max():.1f}, frac<={thr}: {frac}")
            ax.axvline(maxd, color=colors["yaml"], ls="--", lw=1)
            ax.axvline(4.0, color=colors["tight4"], ls="--", lw=1)
            ax.set_title(f"res {a}-{b}  {metric}", fontsize=9)
            ax.set_xlabel("distance (A)")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"hist_{a}_{b}.png", dpi=120)
        plt.close(fig)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    if not parser.parse_args().analyze_only:
        OUT.mkdir(exist_ok=True)
        sample_all()
    analyze()
