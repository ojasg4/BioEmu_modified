import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Cache needs system, MSA, mode, model checkpoint, and precision to match.
#os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

import secrets
from pathlib import Path

from bioemu2 import Ligand, Protein, System, load_model, sample_equilibrium

sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"

system = System(
    [
        Protein(sequence=sequence, asym_ids=["A"]),
        Ligand(ccd_code="CLR", asym_ids=["L1", "L2", "L3", "L4", "L5", "L6"]),
        Ligand(ccd_code="POV", asym_ids=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]),
    ]
)
output_dir = "./my_scripts/outputs/A2A_lipids_90000_6_14"
num_samples = 90000
batch_size = 128

# bioemu2 only exposes per-batch seeds when a seed is passed in (batch i uses
# seed + i), so draw a random base seed here. It is saved in run_seed.txt and
# reused on resume, so batches kept in output_dir/partial stay consistent with
# the rest of the run. Delete run_seed.txt (and partial/) to start a fresh run.
seed_path = Path(output_dir) / "run_seed.txt"
batch_seed_path = Path(output_dir) / "batch_seeds.txt"
seed_path.parent.mkdir(parents=True, exist_ok=True)
if seed_path.is_file():
    seed = int(seed_path.read_text().strip())
else:
    seed = secrets.randbits(32)
    seed_path.write_text(f"{seed}\n")
num_batches = -(-num_samples // batch_size)
print(f"Run seed: {seed}; batches 0-{num_batches - 1} use seeds {seed}-{seed + num_batches - 1}")
with batch_seed_path.open("w") as f:
    for i, start in enumerate(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)
        f.write(f"batch {i} (samples {start}-{end - 1}): seed {seed + i}\n")

model = load_model()

sample_equilibrium(
    system,
    model=model,
    num_samples=num_samples,
    batch_size=batch_size,
    num_steps=100,
    seed=seed,
    output_dir=output_dir,
)
