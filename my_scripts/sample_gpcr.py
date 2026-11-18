import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Shared with my_scripts/search_batch_size_gpcr.py: a cache hit here skips the
# trunk entirely, as long as the system, MSA, mode, model checkpoint, and
# precision all match what populated the cache.
os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")
import torch

# NO LIPIDS

print(torch.cuda.is_available())

from bioemu2 import System, load_model, sample_equilibrium

output_dir = "./my_scripts/outputs/A2A_nolipids_steering"

sequence = "SSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKNHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRS"
system = System.from_description(proteins=[sequence])
model = load_model()

sample_equilibrium(
    system,
    model=model,
    num_samples=256,
    batch_size=128,
    output_dir=output_dir,
)
