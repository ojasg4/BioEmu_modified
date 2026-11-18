import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Cache needs system, MSA, mode, model checkpoint, and precision to match.
#os.environ.setdefault("BIOEMU2_EMBEDS_CACHE", "my_scripts/outputs/embeddings_cache")

from pathlib import Path

from bioemu2 import load_model, load_system_and_contacts, sample_equilibrium
from bioemu2.denoising.steering.potentials import get_default_potentials
from steering_log import StepLog, log_denoise_calls

# System and active-state microswitch contacts (D'Amore et al., Chem 2024) in the YAML.
system, contacts = load_system_and_contacts(Path(__file__).parent / "a2a_active_contacts.yaml")
system.validate_contacts(contacts)

output_dir = "./my_scripts/outputs/A2A_100_MS_yaml_6_14_5k_2"
log_path = Path(output_dir) / "test.txt"
num_samples = 5000
batch_size = 128
num_steps = 200

print("Run parameters:")
print(f"  output_dir:  {output_dir}")
print(f"  num_samples: {num_samples}")
print(f"  batch_size:  {batch_size}")
print(f"  num_steps:   {num_steps}")
print(f"  num_batches: {-(-num_samples // batch_size)}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"  num_contacts: {len(contacts)}")
print(f"  entities: {[(type(e).__name__, len(e.asym_ids)) for e in system.entities]}", flush=True)

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
        contacts=contacts,
        output_dir=output_dir,
    )
finally:
    # One combined set of plots (batch mean, min-max band) rather than a
    # folder per batch, since this run has many batches.
    step_log.close(per_batch=False)
