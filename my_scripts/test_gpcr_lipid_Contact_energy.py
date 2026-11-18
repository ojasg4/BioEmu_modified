import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Cache needs system, MSA, mode, model checkpoint, and precision to match.
#os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

from pathlib import Path

import torch

from bioemu2 import Contact, Ligand, Protein, System, load_model, sample_equilibrium
from bioemu2.denoising.steering.potentials import ContactPotential, get_default_potentials
from bioemu2.structure.all_atom_structure import (
    AllAtomStructure,
    decode_atom_name,
    get_asym_id_to_chain_metadata,
)
from steering_log import LoggedPotential, StepLog, log_denoise_calls

sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"

system = System(
    [
        Protein(sequence=sequence, asym_ids=["A"]),
        Ligand(ccd_code="CLR", asym_ids=["L1", "L2", "L3", "L4", "L5", "L6"]),
        Ligand(ccd_code="POV", asym_ids=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14"]),
    ]
)
# PIP2 (A1L26, chain P1) contact conditioning, from Song et al., Structure 2019
# (doi:10.1016/j.str.2018.10.024): a lipid-residue contact starts at < 0.55 nm.
# The paper's A2aR numbering is offset by 5 from this construct (S6 -> residue 1).
# Default site is TM6/TM7 (R291, R293), the PIP2 site that tightens on activation.
# Other sites: TM1/TM2/TM4 R107,R111; TM3/ICL2/TM4 R120,K122; TM3/TM5 R205,R206.
PIP2_CONTACT_DISTANCE = 5.5  # Angstrom
PIP2_HEADGROUP_ATOM = "P60"  # inositol phosphate of A1L26
PIP2_SITE_RESIDUES = [291 - 5, 293 - 5]
contacts = [
    Contact(
        token1={"asym_id": "A", "res_id": res_id},
        token2={"asym_id": "P1", "atom_name": PIP2_HEADGROUP_ATOM},
        max_distance=PIP2_CONTACT_DISTANCE,
    )
    for res_id in PIP2_SITE_RESIDUES
]

# Energy steering for the same contacts. A Contact only conditions the trunk;
# ContactPotential holds the flat-bottom energy relu(d - max_distance) but nothing
# in bioemu2 fills its atom pairs, and the sampler re-runs prepare() before
# sampling, so the pairs are added inside prepare(). Residue contacts need one
# atom: CZ (Arg guanidinium carbon) is used as the salt-bridge partner of P60.
PIP2_SITE_ATOM = "CZ"


class PIP2ContactPotential(ContactPotential):
    """ContactPotential over (asym_id, res_id, atom_name) endpoint pairs."""

    def __init__(self, pairs, thresholds, parameters):
        super().__init__(parameters=parameters)
        self.pairs = pairs
        self.thresholds = thresholds

    def prepare(self, system):
        super().prepare(system)
        structure = AllAtomStructure.from_system(system, mode="structure")
        chain_index = {a: i for i, a in enumerate(get_asym_id_to_chain_metadata(system))}
        atom_asym = structure.asym_id[structure.token_idx_at_atom]
        # residue_idx increases across chains; make it chain-local and 0-based.
        res = structure.residue_idx.clone()
        for chain in atom_asym.unique():
            res[atom_asym == chain] -= res[atom_asym == chain].min()
        names = [decode_atom_name(n.tolist()) for n in structure.name_encoding]

        def atom_index(asym_id, res_id, atom_name):
            hits = [
                i
                for i in range(structure.num_nodes)
                if atom_asym[i] == chain_index[asym_id]
                and res[i] == res_id - 1
                and names[i] == atom_name
            ]
            if len(hits) != 1:
                raise ValueError(f"{asym_id}:{res_id}:{atom_name} matched {len(hits)} atoms")
            return hits[0]

        index = [(atom_index(*a), atom_index(*b)) for a, b in self.pairs]
        print(f"PIP2ContactPotential atom pairs: {index}", flush=True)
        self.feats["contact_pair_index"] = torch.tensor(index).T[None]  # [1, 2, K]
        self.feats["contact_thresholds"] = torch.tensor(
            self.thresholds, dtype=torch.float32
        )[None]  # [1, K]


contact_potential = PIP2ContactPotential(
    pairs=[(("A", res_id, PIP2_SITE_ATOM), ("P1", 1, PIP2_HEADGROUP_ATOM)) for res_id in PIP2_SITE_RESIDUES],
    thresholds=[PIP2_CONTACT_DISTANCE] * len(PIP2_SITE_RESIDUES),
    parameters={"guidance_weight": 0.1, "guidance_interval": 1, "resampling_weight": 1.0},
)

output_dir = "./my_scripts/outputs/A2A_200_6_13_PIP2_c1_steered"
log_path = Path(output_dir) / "test.txt"
NUM_GD_STEPS = 20  # GuidedFKSampler default inner guidance steps
num_samples = 100
batch_size = 128
num_steps = 200

print("Run parameters:")
print(f"  output_dir:  {output_dir}")
print(f"  num_samples: {num_samples}")
print(f"  batch_size:  {batch_size}")
print(f"  num_steps:   {num_steps}")
print(f"  num_batches: {-(-num_samples // batch_size)}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  entities: {[(type(e).__name__, len(e.asym_ids)) for e in system.entities]}", flush=True)


step_log = StepLog(log_path, num_steps, NUM_GD_STEPS)
potentials = [LoggedPotential(p, step_log) for p in (*get_default_potentials(), contact_potential)]

model = load_model()
log_denoise_calls(model, step_log)

try:
    sample_equilibrium(
        system,
        model=model,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        contacts=contacts,
        sampler="edm_fk",
        potentials=potentials,
        output_dir=output_dir,
    )
finally:
    step_log.close()
