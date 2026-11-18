import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Cache needs system, MSA, mode, model checkpoint, and precision to match.
#os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

import secrets
import time
from pathlib import Path

import mdtraj
import numpy as np
import torch

from bioemu2 import (
    EDMSampler,
    EmuSession,
    Ligand,
    Protein,
    System,
    check_physicality,
    join_trajectory_steps,
    load_model,
    map_cif_atom_order,
    save_frames,
)
from bioemu2.denoising.steering.potentials import get_default_potentials
from steering_log import StepLog, log_denoise_calls

sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"

system = System(
    [
        Protein(sequence=sequence, asym_ids=["A"]),
        Ligand(ccd_code="CLR", asym_ids=["L1", "L2", "L3", "L4", "L5", "L6"]),
        Ligand(ccd_code="POV", asym_ids=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]),
    ]
)

# --- Starting structure ---
# start_cif: a CIF for this exact system (protein, 6 CLR, 14 POV): a bioemu2
# topology.cif, or a frame saved from VMD as pdbx and renamed .cif (VMD
# mangles chain IDs and drops columns biotite needs, so such a file is read
# by atom order instead; see load_start below).
# start_xtc + start_frame: optionally take the coordinates of one frame of a
# bioemu2 run's samples.xtc instead of the CIF's own coordinates. Frame
# indices are 0-based, the same as VMD's. Set start_xtc = None to use the CIF.
start_cif = "./my_scripts/templates/active_6_14.cif"
start_xtc = None
start_frame = 0
# --- Propagation ---
lag_time_ns = float(os.environ.get("LAG_TIME_NS", 1000.0))  # model mostly trained on 1-1000 ns
# -----------------------------------------------

# The env overrides below exist for short pilot/timing runs, e.g.
#   NUM_TRAJECTORIES=1 NUM_TRAJECTORY_STEPS=3 OBSERVE_ENERGIES=0 OUTPUT_DIR=... python ...
output_dir = os.environ.get("OUTPUT_DIR", "./my_scripts/outputs/A2A_lipids_simulate")
log_path = Path(output_dir) / "test.txt"
num_trajectories = int(os.environ.get("NUM_TRAJECTORIES", 128))  # independent trajectories, all from the same frame
num_trajectory_steps = int(os.environ.get("NUM_TRAJECTORY_STEPS", 100))
batch_size = int(os.environ.get("BATCH_SIZE", 128))  # trajectories per denoiser call; one step = ceil(num_trajectories / batch_size) batches
num_steps = 100  # denoising steps per propagation step

if start_cif is None or lag_time_ns is None:
    raise SystemExit("Set start_cif (and optionally start_xtc/start_frame) and lag_time_ns before running.")

# Unlike bioemu2's simulate_trajectory (which keeps everything in memory and
# saves only at the end), every propagation step is written to
# output_dir/steps/step_NNNN.xtc as soon as it finishes, so a crash or rerun
# resumes from the last finished step. Step k (1-based) uses seed + k. The base
# seed is saved in run_seed.txt and reused on resume. Delete run_seed.txt and
# steps/ to start a fresh run.
seed_path = Path(output_dir) / "run_seed.txt"
step_seed_path = Path(output_dir) / "step_seeds.txt"
steps_dir = Path(output_dir) / "steps"
steps_dir.mkdir(parents=True, exist_ok=True)
if seed_path.is_file():
    seed = int(seed_path.read_text().strip())
else:
    seed = secrets.randbits(32)
    seed_path.write_text(f"{seed}\n")
print(f"Run seed: {seed}; steps 1-{num_trajectory_steps} use seeds {seed + 1}-{seed + num_trajectory_steps}")
with step_seed_path.open("w") as f:
    for k in range(1, num_trajectory_steps + 1):
        f.write(f"step {k} (t = {k * lag_time_ns:g} ns): seed {seed + k}\n")

print("Run parameters:")
print(f"  output_dir:           {output_dir}")
print(f"  start:                {start_xtc or start_cif} frame {start_frame if start_xtc else 0}")
print(f"  lag_time_ns:          {lag_time_ns}")
print(f"  num_trajectory_steps: {num_trajectory_steps} (total {num_trajectory_steps * lag_time_ns:g} ns)")
print(f"  num_trajectories:     {num_trajectories}")
print(f"  batch_size:           {batch_size}")
print(f"  num_steps:            {num_steps}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  entities: {[(type(e).__name__, len(e.asym_ids)) for e in system.entities]}", flush=True)

def load_start(cif):
    """Load a starting frame with the system's own topology, which propagation requires.

    Tries map_cif_atom_order first (matches chains by name, any atom order).
    If that fails, e.g. for a VMD-written CIF with truncated chain IDs
    ("L11" for L1, "M11" for M1 and M10-M14) and no pdbx_PDB_model_num
    column, falls back to reading atoms in file order, provided every atom's
    name and residue name match the system's atom order exactly.
    """
    try:
        return map_cif_atom_order(system, cif)[:1]
    except Exception as error:
        print(f"map_cif_atom_order failed ({type(error).__name__}: {error}); reading {cif} by atom order")
    columns, rows = [], []
    for line in open(cif):
        if line.startswith("_atom_site."):
            columns.append(line.strip().removeprefix("_atom_site."))
        elif line.startswith(("ATOM", "HETATM")):
            rows.append(line.split())
    col = {name: columns.index(name) for name in
           ("label_atom_id", "label_comp_id", "Cartn_x", "Cartn_y", "Cartn_z")}
    atoms = list(system.topology.atoms)
    if len(rows) != len(atoms):
        raise ValueError(f"{cif} has {len(rows)} atoms; the system has {len(atoms)}")
    for i, (atom, row) in enumerate(zip(atoms, rows)):
        if (row[col["label_atom_id"]], row[col["label_comp_id"]]) != (atom.name, atom.residue.name):
            raise ValueError(
                f"{cif} atom {i + 1} is {row[col['label_comp_id']]} {row[col['label_atom_id']]}; "
                f"the system expects {atom.residue.name} {atom.name} (chain {atom.residue.chain.chain_id})"
            )
    xyz = np.array([[float(r[col[c]]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")] for r in rows])
    return mdtraj.Trajectory(xyz=xyz[None] / 10.0, topology=system.topology)  # Angstrom -> nm


# bioemu2 writes samples.xtc in the system's atom order, so a chosen xtc
# frame's coordinates can be dropped straight in.
start = load_start(start_cif)
if start_xtc is not None:
    frame = mdtraj.load_frame(str(start_xtc), start_frame, top=start.topology)
    start.xyz = frame.xyz


def step_path(k):
    return steps_dir / f"step_{k:04d}.xtc"


def save_step(k, frames):
    # Write to a temp file and rename, so a crash never leaves a half-written step.
    tmp = steps_dir / f"step_{k:04d}.tmp.xtc"
    frames.save_xtc(str(tmp))
    os.replace(tmp, step_path(k))


def load_step(k):
    return mdtraj.load(str(step_path(k)), top=start.topology)


if not step_path(0).is_file():
    save_step(0, start[[0] * num_trajectories])
last_done = max(k for k in range(num_trajectory_steps + 1) if step_path(k).is_file())
current = load_step(last_done)
if current.n_frames != num_trajectories:
    raise SystemExit(
        f"{step_path(last_done)} has {current.n_frames} frames but num_trajectories={num_trajectories}; "
        "use a new output_dir to change the number of trajectories."
    )
if last_done:
    print(f"Resuming after step {last_done} (t = {last_done * lag_time_ns:g} ns)")

OBSERVE_ENERGIES = os.environ.get("OBSERVE_ENERGIES", "1") != "0"
observe = get_default_potentials() if OBSERVE_ENERGIES else None

# Each propagation step is one or more "# batch N" blocks in test.txt. Only
# steps run in this invocation are logged; resumed steps are not.
step_log = StepLog(log_path, num_steps, num_gd_steps=0)

model = load_model()
log_denoise_calls(model, step_log, observe=observe, system=system)

# The session embeds the system once and reuses it for every step.
emu = EmuSession(system, model=model, mode="equilibrium")
sampler = EDMSampler(num_steps=num_steps)

# Compute (or load from cache) the embedding now, so step 1's time is propagation only.
t0 = time.perf_counter()
emu._get_embedding()
print(f"embedding ready in {time.perf_counter() - t0:.1f} s", flush=True)

try:
    for k in range(last_done + 1, num_trajectory_steps + 1):
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        current = emu.propagate(
            current,
            lag_time_ns=lag_time_ns,
            batch_size=batch_size,
            sampler=sampler,
            seed=seed + k,
        )
        elapsed = time.perf_counter() - t0
        save_step(k, current)
        print(f"step {k}/{num_trajectory_steps} (t = {k * lag_time_ns:g} ns, seed {seed + k}): "
              f"{elapsed:.1f} s, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, saved",
              flush=True)
finally:
    # One combined set of plots (batch mean, min-max band) rather than a
    # folder per batch, since this run has many batches.
    step_log.close(per_batch=False)

# Reorganize [step][trajectory] into one time-ordered trajectory per start,
# each including the starting frame at t = 0 (MDTraj times are in ps).
trajectories = join_trajectory_steps(
    [load_step(k) for k in range(num_trajectory_steps + 1)], lag_time_ns=lag_time_ns
)
num_passed = 0
for i, trajectory in enumerate(trajectories):
    physicality = check_physicality(trajectory, system=system)
    num_passed += int(physicality.passed.sum())
    save_frames(trajectory, Path(output_dir) / "trajectories" / f"traj_{i:03d}", physicality=physicality)
print(f"Saved {len(trajectories)} trajectories of {trajectories[0].n_frames} frames to "
      f"{output_dir}/trajectories; {num_passed}/{len(trajectories) * trajectories[0].n_frames} "
      "frames passed physicality checks (all frames kept)")
