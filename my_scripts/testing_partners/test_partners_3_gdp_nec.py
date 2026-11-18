import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pathlib import Path

# steering_log.py lives one directory up in my_scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bioemu2 import Ligand, Protein, System, load_model, sample_equilibrium
from bioemu2.denoising.steering.potentials import get_default_potentials
from steering_log import StepLog, log_denoise_calls

# System: A2A receptor + G protein + GDP x1, NEC x1, no lipids.
# Unguided sampling (plain edm); physicality-potential energies are logged only.
# CLR = cholesterol, POV = POPC.

print(torch.cuda.is_available())

output_dir = "./my_scripts/outputs/testing_partners/GDP_NEC"
log_path = Path(output_dir) / "test.txt"

receptor = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"
g_protein = "GIEKQLQKDKQVYRATHRLLLLGADNSGKSTIVKQMRIYHGGSGGSGGTSGIFETKFQVDKVNFHMFDVGGQRDERRKWIQCFNDVTAIIFVVDSSDYNRLQEALNDFKSIWNNRWLRTISVILFLNKQDLLAEKVLAGKSKIEDYFPEFARYTTPEDATPEPGEDPRVTRAKYFIRDEFLRISTASGDGRHYCYPHFTCAVDTENARRIFNDCRDIIQRMHLRQYELL"

system = System(
    [
        Protein(sequence=receptor, asym_ids=["A"]),
        Protein(sequence=g_protein, asym_ids=["B"]),
        Ligand(ccd_code="GDP", asym_ids=["G1"]),
        Ligand(ccd_code="NEC", asym_ids=["N1"]),
    ]
)

num_samples = 128
batch_size = 128  # one batch
num_steps = 200
SAMPLER = "edm"  # no guidance

print("Run parameters:")
print(f"  output_dir:  {output_dir}")
print(f"  num_samples: {num_samples}")
print(f"  batch_size:  {batch_size}")
print(f"  num_steps:   {num_steps}")
print(f"  sampler:     {SAMPLER}")
print(f"  num_batches: {-(-num_samples // batch_size)}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

# Observed only: energies are evaluated on x0_hat and logged, never applied.
potentials = get_default_potentials()
step_log = StepLog(log_path, num_steps, num_gd_steps=1)

model = load_model()
log_denoise_calls(model, step_log, observe=potentials, system=system)

try:
    sample_equilibrium(
        system,
        model=model,
        sampler=SAMPLER,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        output_dir=output_dir,
    )
finally:
    step_log.close(per_batch=False)
