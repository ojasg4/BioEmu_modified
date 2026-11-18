import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Cache needs system, MSA, mode, model checkpoint, and precision to match.
#os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

import secrets
from pathlib import Path

import mdtraj

from bioemu2 import (
    EDMSampler,
    EmuSession,
    Ligand,
    Protein,
    System,
    check_physicality,
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
        Ligand(ccd_code="NEC", asym_ids=["N1"]),
        Ligand(ccd_code="CLR", asym_ids=["L1", "L2", "L3", "L4", "L5", "L6"]),
        Ligand(ccd_code="POV", asym_ids=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13"]),
    ]
)

x0_cif = "./my_scripts/templates/NECA_6_13_inactive_merged.cif"
lag_time_ns = 1000


output_dir = "./my_scripts/outputs/A2A_NECA_6_13_1000ns_1000step"
log_path = Path(output_dir) / "test.txt"
num_samples = 1000
batch_size = 4
num_steps = 100

if x0_cif is None or lag_time_ns is None:
    raise SystemExit("Set x0_cif and lag_time_ns at the top of this script before running.")

# Delete run_seed.txt (and partial/) to start a fresh run.
seed_path = Path(output_dir) / "run_seed.txt"
batch_seed_path = Path(output_dir) / "batch_seeds.txt"
partial_dir = Path(output_dir) / "partial"
seed_path.parent.mkdir(parents=True, exist_ok=True)
if seed_path.is_file():
    seed = int(seed_path.read_text().strip())
else:
    seed = secrets.randbits(32)
    seed_path.write_text(f"{seed}\n")
num_batches = -(-num_samples // batch_size)
batches = [
    (i, start, min(start + batch_size, num_samples))
    for i, start in enumerate(range(0, num_samples, batch_size))
]
print(f"Run seed: {seed}; batches 0-{num_batches - 1} use seeds {seed}-{seed + num_batches - 1}")
with batch_seed_path.open("w") as f:
    for i, start, end in batches:
        f.write(f"batch {i} (samples {start}-{end - 1}): seed {seed + i}\n")

print("Run parameters:")
print(f"  output_dir:  {output_dir}")
print(f"  x0_cif:      {x0_cif}")
print(f"  lag_time_ns: {lag_time_ns}")
print(f"  num_samples: {num_samples}")
print(f"  batch_size:  {batch_size}")
print(f"  num_steps:   {num_steps}")
print(f"  num_batches: {num_batches}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  entities: {[(type(e).__name__, len(e.asym_ids)) for e in system.entities]}", flush=True)

x0 = map_cif_atom_order(system, x0_cif)[:1]

OBSERVE_ENERGIES = True
observe = get_default_potentials() if OBSERVE_ENERGIES else None

step_log = StepLog(log_path, num_steps, num_gd_steps=0)

model = load_model()
log_denoise_calls(model, step_log, observe=observe, system=system)

# The session embeds the system once and reuses it for every batch.
emu = EmuSession(system, model=model, mode="equilibrium")
sampler = EDMSampler(num_steps=num_steps)

try:
    for i, start, end in batches:
        batch_dir = partial_dir / f"batch_{i:05d}"
        if (batch_dir / "samples.xtc").is_file():
            print(f"batch {i}: found in {batch_dir}, skipping")
            continue
        endpoints = emu.propagate(
            x0[[0] * (end - start)],
            lag_time_ns=lag_time_ns,
            batch_size=end - start,
            sampler=sampler,
            seed=seed + i,
        )
        save_frames(endpoints, batch_dir)
        print(f"batch {i} (samples {start}-{end - 1}, seed {seed + i}): saved to {batch_dir}", flush=True)
finally:
    # One combined set of plots (batch mean, min-max band) rather than a
    # folder per batch, since this run has many batches.
    step_log.close(per_batch=False)

# Join the batches in order and write the final outputs (all frames kept;
# physicality.json reports which pass, as in sample_equilibrium output).
# x0's topology is the system's own (map_cif_atom_order), matching every batch.
trajectory = mdtraj.join(
    [mdtraj.load(str(partial_dir / f"batch_{i:05d}" / "samples.xtc"), top=x0.topology) for i, _, _ in batches]
)
physicality = check_physicality(trajectory, system=system)
save_frames(trajectory, output_dir, physicality=physicality)
print(f"Saved {trajectory.n_frames} endpoints to {output_dir}; "
      f"{int(physicality.passed.sum())} passed physicality checks")
