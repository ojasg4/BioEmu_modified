import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Cache needs system, MSA, mode, model checkpoint, and precision to match.
#os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

from pathlib import Path

from bioemu2 import Ligand, Protein, System, load_model, sample_equilibrium
from bioemu2.denoising.steering.potentials import get_default_potentials
from steering_log import StepLog, log_denoise_calls

sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"

system = System(
    [
        Protein(sequence=sequence, asym_ids=["A"]),
        Ligand(ccd_code="CLR", asym_ids=["L1", ]), #"L2", "L3", "L4", "L5", "L6"]),
        #Ligand(ccd_code="POV", asym_ids=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]),
    ]
)
# # PIP2 (A1L26, chain P1) contact conditioning, from Song et al., Structure 2019
# # (doi:10.1016/j.str.2018.10.024): a lipid-residue contact starts at < 0.55 nm.
# # The paper's A2aR numbering is offset by 5 from this construct (S6 -> residue 1).
# # Default site is TM6/TM7 (R291, R293), the PIP2 site that tightens on activation.
# # Other sites: TM1/TM2/TM4 R107,R111; TM3/ICL2/TM4 R120,K122; TM3/TM5 R205,R206.
# PIP2_CONTACT_DISTANCE = 5.5  # Angstrom
# PIP2_HEADGROUP_ATOM = "P60"  # inositol phosphate of A1L26
# PIP2_SITE_RESIDUES = [291 - 5, 293 - 5]
# contacts = [
#     Contact(
#         token1={"asym_id": "A", "res_id": res_id},
#         token2={"asym_id": "P1", "atom_name": PIP2_HEADGROUP_ATOM},
#         max_distance=PIP2_CONTACT_DISTANCE,
#     )
#     for res_id in PIP2_SITE_RESIDUES
# ]

# Plain run: no contacts and the default "edm" sampler (no steering), so
# test.txt holds one score line per step ("no guidance").
output_dir = "./my_scripts/outputs/A2A_200_1_0"
log_path = Path(output_dir) / "test.txt"
num_samples = 4
batch_size = 2
num_steps = 200

print("Run parameters:")
print(f"  output_dir:  {output_dir}")
print(f"  num_samples: {num_samples}")
print(f"  batch_size:  {batch_size}")
print(f"  num_steps:   {num_steps}")
print(f"  num_batches: {-(-num_samples // batch_size)}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  entities: {[(type(e).__name__, len(e.asym_ids)) for e in system.entities]}", flush=True)

# True: also log the default potentials' energies (bonds, angles, clashes,
# chirality, ...) on x0_hat every step. Observed only, never applied, so samples
# match a plain edm run. False: score lines only.
OBSERVE_ENERGIES = True
observe = get_default_potentials() if OBSERVE_ENERGIES else None

step_log = StepLog(log_path, num_steps, num_gd_steps=0)

model = load_model()
log_denoise_calls(model, step_log, observe=observe, system=system)

try:
    sample_equilibrium(
        system,
        model=model,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        # contacts=contacts,
        output_dir=output_dir,
    )
finally:
    step_log.close()
