import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Shared with my_scripts/search_batch_size_gpcr.py: a cache hit here skips the
# trunk entirely, as long as the system, MSA, mode, model checkpoint, and
# precision all match what populated the cache.
os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")
from pathlib import Path

import torch

from bioemu2 import System, load_model, sample_equilibrium
from bioemu2.denoising.steering.potentials import get_default_potentials
from steering_log import LoggedPotential, StepLog, log_denoise_calls

# NO LIPIDS. Same as sample_gpcr.py, but steered with the default physicality
# potentials (docs/potentials.md) and logged as in test_gpcr_lipid_denoising.py.

print(torch.cuda.is_available())

output_dir = "./my_scripts/outputs/A2A_potentials_10000"
log_path = Path(output_dir) / "test.txt"

sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"
system = System.from_description(proteins=[sequence])

num_samples = 256
batch_size = 128
num_steps = 200
# "edm_gd": guidance only (recommended first try); "edm_fk": guidance + resampling.
# Both use GuidedFKSampler, whose default num_gd_steps is 20. (steering_log does
# not label "edm_fkc" steps correctly.)
SAMPLER = "edm_gd"
NUM_GD_STEPS = 20  # keep equal to the sampler's num_gd_steps (default 20)

print("Run parameters:")
print(f"  output_dir:  {output_dir}")
print(f"  num_samples: {num_samples}")
print(f"  batch_size:  {batch_size}")
print(f"  num_steps:   {num_steps}")
print(f"  sampler:     {SAMPLER}")
print(f"  num_batches: {-(-num_samples // batch_size)}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

# The default physicality potentials (bond lengths/angles, VDW overlaps, local
# clashes, chirality, ...), each wrapped so its guidance energies are logged.
# Passing potentials=[...] replaces the sampler's defaults with this same set.
potentials = [LoggedPotential(p, None) for p in get_default_potentials()]

step_log = StepLog(log_path, num_steps, num_gd_steps=NUM_GD_STEPS)
for p in potentials:
    p.log = step_log

model = load_model()
log_denoise_calls(model, step_log)

try:
    sample_equilibrium(
        system,
        model=model,
        sampler=SAMPLER,
        potentials=potentials,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        output_dir=output_dir,
    )
finally:
    # One combined set of plots (batch mean, min-max band) rather than a
    # folder per batch, since this run has many batches.
    step_log.close(per_batch=False)
