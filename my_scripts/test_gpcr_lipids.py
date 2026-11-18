import gc
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Shared with my_scripts/sample_gpcr_lipids.py: populates the trunk-embedding
# cache for this exact system, so the 10000-sample production run can skip
# straight to denoising if this test passes.
os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

import torch

from bioemu2 import Ligand, Protein, System, load_model, sample_equilibrium

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lipid-count probe: batch_size=2 and num_samples=2 stay fixed the whole
# sweep, so any OOM comes from token count (more lipids), not batch size.
# Keeps this system's 3 CLR : 7 POPC ratio and multiplies both up each
# round to find how many lipid copies fit before running out of memory.
# Each round gets its own output_dir so the resulting frames are kept
# per lipid count instead of overwriting each other.
sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"

model = load_model()

for multiplier in (1, 2, 3, 4, 5, 6, 8, 10):
    n_clr = 3 * multiplier
    n_popc = 7 * multiplier
    system = System(
        [
            Protein(sequence=sequence, asym_ids=["A"]),
            Ligand(ccd_code="CLR", asym_ids=[f"L{i}" for i in range(1, n_clr + 1)]),
            Ligand(ccd_code="POV", asym_ids=[f"M{i}" for i in range(1, n_popc + 1)]),
        ]
    )
    output_dir = f"./my_scripts/outputs/test_lipids_{n_clr}CLR_{n_popc}POPC"
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        sample_equilibrium(
            system,
            model=model,
            num_samples=2,
            batch_size=2,
            progress=False,
            output_dir=output_dir,
        )
        peak_gb = torch.cuda.max_memory_reserved() / 1e9
        logger.info("CLR=%d POPC=%d OK peak_mem=%.2fGB", n_clr, n_popc, peak_gb)
    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_reserved() / 1e9
        logger.info(
            "CLR=%d POPC=%d CUDA OOM at peak_mem=%.2fGB -- stopping sweep", n_clr, n_popc, peak_gb
        )
        gc.collect()
        torch.cuda.empty_cache()
        break
