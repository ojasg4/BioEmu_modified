"""Per-step logging of the EDM score and steering energies.

Writes one line per (denoising step, inner guidance step): the network's score
paired with each potential's guidance energy. Each sampling batch starts with "# batch N" and
restarts step numbering at 1 (detected by sigma rising back to its maximum).

Works with edm, edm_fk, and edm_gd samplers

Observe mode (log_denoise_calls(..., observe=potentials, system=system)) adds the
unweighted energy of each listed potential on the network's x0_hat to every step
line, computed without gradients and never applied. Use it for an unsteered
baseline in plain edm runs, where the sampler evaluates no potentials. It costs
one no-grad evaluation of each potential per step.

Does not work with (runs fine, but step labels are wrong):
    sampler="edm_fkc"  Up to 2 denoise calls per step (counted as 2 steps), and
                       in weight_update="guidance" mode potential calls through
                       the denoiser are logged as inner step 1.
    denoise_micro_batch_size set on a steered sampler: each particle chunk is a
                       separate denoise call, so one step is split into several.
    Context-parallel sampling (bioemu2.cp): not supported or tested. It uses the
                       CP model wrapper, and every rank would write the same file.

The "/N" inner-step label is the num_gd_steps passed to StepLog; keep it equal to
the sampler's num_gd_steps (GuidedFKSampler default 20).

Plots: StepLog.close() writes PNGs to <test.txt folder>/plots/ (close(plot=False)
to skip), including after a crash. Which plots depends on what the run logged:
    every run          score.png               sigma, |x0_hat - x_t|, score per step
    observe mode       observed_energies.png   each observed potential on x0_hat
    edm_fk / edm_gd    guidance_energies.png   each potential before vs after the
                                               inner guidance steps
                       guidance_vs_score.png   guidance shift vs the denoiser's own
                                               displacement, their ratio, and the
                                               weighted total energy being descended
With several batches, each batch gets its own set in plots/batch_001/,
plots/batch_002/, ... (one PNG set per batch, so runs with hundreds of batches
write hundreds of folders). close(per_batch=False) or --combined instead writes
one set where each line is the batch mean and the band spans min to max.
Re-plot any existing log (also logs written before batch markers existed):
    python my_scripts/steering_log.py path/to/test.txt [out_dir] [--combined]
Plotting needs matplotlib (not a bioemu2 dependency); logging does not.

Usage:
    step_log = StepLog(path, num_steps, num_gd_steps=20)
    potentials = [LoggedPotential(p, step_log) for p in potentials]  # steered only
    log_denoise_calls(model, step_log)  # plain edm + energies: add observe=..., system=...
    try:
        sample_equilibrium(..., sampler="edm_fk", potentials=potentials)
    finally:
        step_log.close()
"""

import contextlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bioemu2.denoising.steering import potentials as potentials_module
from bioemu2.denoising.steering.potentials import Potential

# Every potential is a sum of hinge violations with an implicit slope k = 1, so its
# "energy" carries the unit of the quantity it constrains: Angstrom for distances,
# radian for angles and dihedrals. These are penalties, not physical energies.
UNITS = {
    "BondLengthPotential": "A",
    "BondAnglePotential": "rad",
    "InterChainVDWOverlapPotential": "A",
    "IntraChainVDWOverlapPotential": "A",
    "PolymerLocalClashPotential": "A",
    "SymmetricChainCOMPotential": "A",
    "PoseBustersPotential": "A",
    "ChiralAtomPotential": "rad",
    "StereoBondPotential": "rad",
    "PlanarBondPotential": "rad",
}


class StepLog:
    """Writes one line per (denoising step, inner guidance step) to test.txt.

    Score: the network's EDM score (x0_hat - x_t) / sigma^2, mean per-atom norm
    over real atoms and particles, plus the denoiser displacement |x0_hat - x_t|.
    Guidance: each active potential's unweighted energy (particle mean) at that
    inner step, the weighted total sum(w_g * E) that is differentiated, and the
    cumulative guidance shift |x0_hat_guided - x0_hat| applied so far.

    With path=None, lines are printed to stdout instead (e.g. in a notebook).
    """

    def __init__(self, path, num_steps, num_gd_steps):
        self.path = path
        if path is None:
            self.file = sys.stdout
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w")
        self.num_steps = num_steps
        self.num_gd_steps = num_gd_steps
        self.batch = 0
        self.step = 0
        self.last_sigma = None
        self.header = None
        self.x0_hat = None
        self.mask = None
        self.rows = {}
        self.atom_label = str  # atom index -> readable label; set by log_denoise_calls
        self.detail_file = None

    def on_denoise(self, sigma, coords, x0_hat, mask):
        self.flush()
        # sigma only decreases within a batch, so a rise means the next batch started.
        if self.last_sigma is None or sigma > self.last_sigma:
            self.batch += 1
            self.step = 0
            self.file.write(f"# batch {self.batch}\n")
        self.last_sigma = sigma
        self.step += 1
        self.x0_hat, self.mask = x0_hat.detach().float(), mask
        real = mask.sum()
        disp = ((x0_hat - coords).float().norm(dim=-1) * mask).sum() / real
        self.header = (
            f"step {self.step:03d}/{self.num_steps} | sigma={sigma:9.4f} A | "
            f"score mean|grad log p|={disp.item() / sigma**2:.4e} 1/A, "
            f"mean|x0_hat - x_t|={disp.item():8.3f} A"
        )

    def on_observe(self, energies):
        terms = ", ".join(f"{n}={e:.4f}" for n, e in energies.items())
        self.header += f" | observed E on x0_hat (not applied): {terms}"

    def on_detail(self, name, records):
        """Write per-term violation statistics for batch 1 to detail_batch1.txt.

        One line per potential per step: how many terms there are, how many are
        violated (mean over particles), each particle's summed energy, and the single
        worst term with the atoms involved. ``records`` holds (terms [B, P], index
        [k, P] or None) captured while the potential was evaluated.
        """
        if self.detail_file is None:
            self.detail_file = sys.stdout if self.path is None else (self.path.parent / "detail_batch1.txt").open("w")
        unit = UNITS.get(name, "?")
        prefix = f"step {self.step:03d}/{self.num_steps} | sigma={self.last_sigma:9.4f} A | {name} [{unit}]"
        if not records:
            self.detail_file.write(f"{prefix} | no terms\n")
            return
        for terms, index in records:
            terms = terms.float()
            per_particle = terms.sum(dim=-1)
            violating = (terms > 0).sum(dim=-1).float().mean().item()
            worst_value, flat = terms.flatten().max(dim=0)
            particle, term = divmod(flat.item(), terms.shape[-1])
            atoms = ""
            if index is not None:
                atoms = " atoms " + "-".join(self.atom_label(int(i)) for i in index[:, term].tolist())
            self.detail_file.write(
                f"{prefix} | terms={terms.shape[-1]} violating={violating:.1f} | "
                f"sum per particle=[{', '.join(f'{v:.3f}' for v in per_particle.tolist())}] | "
                f"worst={worst_value.item():.4f} {unit} (particle {particle}{atoms})\n"
            )
        self.detail_file.flush()

    def on_energy(self, name, inner, energy, weight, coords):
        row = self.rows.setdefault(inner, {"energies": {}, "shift": None})
        row["energies"][name] = (energy, weight)
        if row["shift"] is None and self.x0_hat is not None:
            shift = (coords.detach().float() - self.x0_hat).norm(dim=-1) * self.mask
            row["shift"] = (shift.sum() / self.mask.sum()).item(), shift.max().item()

    def flush(self):
        if self.header is None:
            return
        if not self.rows:
            self.file.write(f"{self.header} | inner --/{self.num_gd_steps} | no guidance this step\n")
        for inner in sorted(self.rows):
            row = self.rows[inner]
            total = sum(e * w for e, w in row["energies"].values())
            terms = ", ".join(f"{n}={e:.4f}(w={w:g})" for n, (e, w) in row["energies"].items())
            mean_shift, max_shift = row["shift"]
            self.file.write(
                f"{self.header} | inner {inner + 1:02d}/{self.num_gd_steps} | "
                f"guidance weighted E={total:.4f}, shift mean={mean_shift:.4f} A max={max_shift:.4f} A"
                f" | {terms}\n"
            )
        self.file.flush()
        self.rows = {}
        self.header = None

    def close(self, plot=True, per_batch=True):
        self.flush()
        if self.detail_file not in (None, sys.stdout):
            self.detail_file.close()
        if self.file is sys.stdout:
            return
        self.file.close()
        # Plot on close, including after a crash (close() sits in a finally);
        # a plotting error only warns, so it never hides the sampling result.
        if plot:
            try:
                written = plot_steering_log(self.path, per_batch=per_batch)
                print(f"steering_log: wrote {len(written)} plots to {self.path.parent / 'plots'}")
            except Exception as error:  # noqa: BLE001
                print(f"steering_log: plotting failed ({error!r}); test.txt is intact", file=sys.stderr)


class LoggedPotential(Potential):
    """Passes through to a potential and logs its guidance-pass energies.

    The sampler evaluates potentials twice: under no_grad for FK weights and
    with requires_grad coords inside the 20 inner guidance steps. Only the
    latter is logged. A potential with guidance_interval I runs on inner steps
    0, I, 2I, ..., so its k-th call in a denoising step is inner step k * I.
    """

    def __init__(self, inner, log):
        self.inner = inner
        self.log = log
        self.name = type(inner).__name__
        self.calls = 0
        self.call_step = None

    def prepare(self, system):
        self.inner.prepare(system)

    def compute_parameters(self, t):
        return self.inner.compute_parameters(t)

    def __call__(self, coords_angstrom, *, t):
        energy = self.inner(coords_angstrom, t=t)
        if coords_angstrom.requires_grad:
            if self.call_step != (self.log.batch, self.log.step):
                self.call_step, self.calls = (self.log.batch, self.log.step), 0
            params = self.compute_parameters(t) or {}
            inner_step = self.calls * params.get("guidance_interval", 1)
            self.calls += 1
            self.log.on_energy(
                self.name,
                inner_step,
                energy.detach().float().mean().item(),
                params.get("guidance_weight", 0.0),
                coords_angstrom,
            )
        return energy


@contextlib.contextmanager
def _record_terms(records):
    """Capture every per-term violation tensor while a potential is evaluated.

    Temporarily wraps the flat-bottom hinge, the steric-overlap penalty and the
    index hand-off, appending (terms [B, P], atom index [k, P]) to ``records``.
    Energies and gradients are unchanged.
    """
    flat_bottom = potentials_module.FlatBottomPotential.compute_function
    from_args = potentials_module.FeatsPotential._compute_from_args
    steric = potentials_module._steric_overlap_penalty
    state = {"index": None}

    def compute_from_args(self, coords, index, args, com_args):
        state["index"] = index
        return from_args(self, coords, index, args, com_args)

    def compute_function(self, value, *args):
        terms = flat_bottom(self, value, *args)
        records.append((terms.detach(), state["index"]))
        return terms

    def steric_overlap(coords, index, minimum_distance):
        i, j = index
        delta = coords.detach().float().index_select(-2, i) - coords.detach().float().index_select(-2, j)
        distance = torch.sqrt(delta.square().sum(dim=-1) + 1e-12)
        records.append((torch.relu(minimum_distance - distance), index))
        return steric(coords, index, minimum_distance)

    potentials_module.FeatsPotential._compute_from_args = compute_from_args
    potentials_module.FlatBottomPotential.compute_function = compute_function
    potentials_module._steric_overlap_penalty = steric_overlap
    try:
        yield
    finally:
        potentials_module.FeatsPotential._compute_from_args = from_args
        potentials_module.FlatBottomPotential.compute_function = flat_bottom
        potentials_module._steric_overlap_penalty = steric


def _atom_labeller(system):
    """Return atom index -> "idx(RES<n>.NAME)"; plain indices if labels are unavailable."""
    try:
        from bioemu2.structure import AllAtomStructure, decode_atom_name

        structure = AllAtomStructure.from_system(system, mode="structure")
        labels = [
            f"{i}({structure.residue_labels[i]}"
            f"{int(structure.residue_idx[i])}.{decode_atom_name(structure.name_encoding[i])})"
            for i in range(structure.num_nodes)
        ]
        return lambda i: labels[i] if i < len(labels) else str(i)
    except Exception:  # noqa: BLE001
        return str


def log_denoise_calls(model, step_log, observe=None, system=None):
    """Wrap ``model.denoise`` on this instance so each call logs the score.

    With ``observe`` (a list of potentials) and ``system``, each step also logs
    those potentials' unweighted energies on the network's x0_hat, computed
    without gradients and never applied, so sampling is unchanged.
    """
    denoise = model.denoise
    observe = list(observe or ())
    if observe:
        if system is None:
            raise ValueError("observe requires system, to prepare the potentials")
        for potential in observe:
            potential.prepare(system)
        step_log.atom_label = _atom_labeller(system)

    def logged_denoise(*, context, trunk_embeddings, coords, sigmas):
        out = denoise(context=context, trunk_embeddings=trunk_embeddings, coords=coords, sigmas=sigmas)
        mask = context.atom_mask.float()
        if mask.shape[0] != coords.shape[0]:
            mask = mask.repeat_interleave(coords.shape[0] // mask.shape[0], 0)
        step_log.on_denoise(sigmas[0].item(), coords, out.denoised_coords, mask)
        if observe:
            # Same normalized time as the steered sampler: t = 1 at peak noise.
            t = 1.0 - step_log.step / step_log.num_steps
            x0_hat = out.denoised_coords.detach().float()
            detail = step_log.batch == 1  # extra per-term logging, first batch only
            energies = {}
            with torch.no_grad():
                for p in observe:
                    name = type(p).__name__
                    records = []
                    with _record_terms(records) if detail else contextlib.nullcontext():
                        energies[name] = p(x0_hat, t=t).float().mean().item()
                    if detail:
                        step_log.on_detail(name, records)
            step_log.on_observe(energies)
        return out

    model.denoise = logged_denoise


# Plotting: parse test.txt and draw whichever metrics the run logged.

_STEP = re.compile(r"step (\d+)/(\d+)")
_SIGMA = re.compile(r"sigma=\s*([-+\d.eE]+)")
_SCORE = re.compile(r"mean\|grad log p\|=([-+\d.eE]+) 1/A, mean\|x0_hat - x_t\|=\s*([-+\d.eE]+)")
_INNER = re.compile(r"inner (\d+|--)/(\d+)")
_GUIDANCE = re.compile(r"guidance weighted E=([-+\d.eE]+), shift mean=([-+\d.eE]+) A max=([-+\d.eE]+) A")
_TERM = re.compile(r"(\w+)=([-+\d.eE]+|nan|inf)(?:\(w=([-+\d.eE]+)\))?")

# Light chart chrome and a validated two-series pair (blue, orange).
_SURFACE, _INK, _INK_2, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS = "#e1e0d9", "#c3c2b7"
_BLUE, _ORANGE = "#2a78d6", "#eb6834"


def parse_steering_log(path):
    """Parse test.txt into one record per (batch, step).

    Each record has batch, step, num_steps, sigma, score, disp, observed
    ({name: E}) and inner (a list of {inner, total, shift_mean, shift_max,
    energies: {name: (E, w)}}). Logs written before "# batch" markers existed
    start a new batch whenever the step number goes back down.
    """
    records = {}
    batch, last_step = 1, 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("# batch"):
            batch, last_step = int(line.split()[-1]), 0
            continue
        if not line.startswith("step "):
            continue
        segments = line.split(" | ")
        step, num_steps = map(int, _STEP.match(segments[0]).groups())
        if step < last_step:
            batch += 1
        last_step = step
        record = records.get((batch, step))
        if record is None:
            score, disp = map(float, _SCORE.search(segments[2]).groups())
            record = records[(batch, step)] = {
                "batch": batch,
                "step": step,
                "num_steps": num_steps,
                "sigma": float(_SIGMA.search(segments[1]).group(1)),
                "score": score,
                "disp": disp,
                "observed": {},
                "inner": [],
            }
        for index, segment in enumerate(segments):
            if segment.startswith("observed E"):
                terms = segment.split(":", 1)[1]
                record["observed"] = {name: float(value) for name, value, _ in _TERM.findall(terms)}
            match = _INNER.fullmatch(segment.strip())
            if match and match.group(1) != "--":
                total, shift_mean, shift_max = map(float, _GUIDANCE.search(segments[index + 1]).groups())
                terms = segments[index + 2] if index + 2 < len(segments) else ""
                record["inner"].append(
                    {
                        "inner": int(match.group(1)),
                        "total": total,
                        "shift_mean": shift_mean,
                        "shift_max": shift_max,
                        "energies": {n: (float(e), float(w)) for n, e, w in _TERM.findall(terms) if w},
                    }
                )
    return list(records.values())


def _by_step(records, value, num_steps):
    """Mean, min and max of value(record) over batches at every step; NaN where absent."""
    groups = defaultdict(list)
    for record in records:
        result = value(record)
        if result is not None:
            groups[record["step"]].append(result)
    steps = np.arange(1, num_steps + 1)
    stats = np.full((3, num_steps), np.nan)
    for step, values in groups.items():
        stats[:, step - 1] = np.mean(values), np.min(values), np.max(values)
    return steps, stats


def _draw(ax, records, value, num_steps, color, label=None, positive=False):
    """Mean line, plus a min-max band when there is more than one batch."""
    steps, (mean, low, high) = _by_step(records, value, num_steps)
    if positive:  # log axes cannot show zeros
        mean, low, high = (np.where(a > 0, a, np.nan) for a in (mean, low, high))
    if np.all(np.isnan(mean)):
        return False
    ax.plot(steps, mean, color=color, linewidth=1.6, label=label)
    if len({r["batch"] for r in records}) > 1:
        ax.fill_between(steps, low, high, color=color, alpha=0.15, linewidth=0)
    return True


def _style(ax, title, ylabel, scale):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_2, labelsize=8)
    ax.set_title(title, loc="left", color=_INK, fontsize=10)
    ax.set_ylabel(ylabel, color=_INK_2, fontsize=8)
    # Data inside one decade gets no labelled log ticks, so plot it linearly.
    values = np.concatenate([line.get_ydata() for line in ax.lines] or [np.array([])])
    values = values[np.isfinite(values)]
    if len(values) and values.min() > 0 and values.max() / values.min() < 10:
        scale = None
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-2)
    elif scale:
        ax.set_yscale(scale)


def _figure(plt, rows, cols, title, subtitle):
    # Single-column figures stay wide enough for their titles and legends.
    fig, axes = plt.subplots(
        rows, cols, figsize=(max(3.8 * cols, 9.0), 2.6 * rows + 0.9), sharex=True, squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / height, title, color=_INK, fontsize=12, va="top")
    fig.text(0.01, 1 - 0.42 / height, subtitle, color=_INK_2, fontsize=8.5, va="top")
    return fig, axes.ravel()


def _finish(fig, axes, used, path, legend=False):
    """Hide unused panels, label the bottom row, add a shared legend, save."""
    cols = axes[0].get_gridspec().ncols
    for ax in axes[used:]:
        ax.set_visible(False)
    for index, ax in enumerate(axes[:used]):
        ax.xaxis.set_tick_params(labelbottom=True)
        if index + cols >= used:  # nothing visible below this panel
            ax.set_xlabel("denoising step  (noise decreases →)", color=_INK_2, fontsize=8)
    height = fig.get_figheight()
    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.62 / height),
                   frameon=False, fontsize=8.5, ncols=len(labels), labelcolor=_INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.95 if legend else 0.7) / height))
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return path


def _panel_legend(ax):
    ax.legend(frameon=False, fontsize=8, labelcolor=_INK_2, loc="best")


def plot_steering_log(path, out_dir=None, per_batch=True):
    """Plot the metrics present in a test.txt and return the written PNG paths.

    Always: score.png (sigma, |x0_hat - x_t|, score). Observe mode:
    observed_energies.png (each potential on x0_hat). Steered runs:
    guidance_energies.png (each potential before vs after the inner guidance
    steps) and guidance_vs_score.png (guidance shift against the denoiser's
    own displacement).

    per_batch=True (default) writes one set of plots per batch, to
    out_dir/batch_001/, out_dir/batch_002/, ... (straight into out_dir when the
    log has a single batch). per_batch=False writes one combined set where each
    line is the batch mean and the band spans min to max.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    records = parse_steering_log(path)
    if not records:
        raise ValueError(f"{path} has no step lines")
    out_dir = Path(out_dir) if out_dir is not None else path.parent / "plots"
    num_steps = max(r["num_steps"] for r in records)
    batches = sorted({r["batch"] for r in records})

    if len(batches) == 1:
        return _plot_records(plt, records, out_dir, f"{path.parent.name}  ·  1 batch, {num_steps} steps")
    if not per_batch:
        subtitle = (
            f"{path.parent.name}  ·  {len(batches)} batches, {num_steps} steps  ·  "
            "line = mean over batches, band = min–max"
        )
        return _plot_records(plt, records, out_dir, subtitle)
    written = []
    for batch in batches:
        subtitle = f"{path.parent.name}  ·  batch {batch} of {len(batches)}, {num_steps} steps"
        batch_records = [r for r in records if r["batch"] == batch]
        written += _plot_records(plt, batch_records, out_dir / f"batch_{batch:03d}", subtitle)
    return written


def _plot_records(plt, records, out_dir, subtitle):
    """Write the plots for one set of records (one batch, or all batches combined)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = max(r["num_steps"] for r in records)
    written = []

    fig, axes = _figure(plt, 3, 1, "Network score per denoising step", subtitle)
    panels = [
        ("Noise level σ (after churn)", "σ (Å)", lambda r: r["sigma"]),
        ("Denoiser displacement |x̂₀ − x_t|", "Å, mean per atom", lambda r: r["disp"]),
        ("Score |∇log p| = |x̂₀ − x_t| / σ²", "1/Å, mean per atom", lambda r: r["score"]),
    ]
    for ax, (title, ylabel, value) in zip(axes, panels):
        _draw(ax, records, value, num_steps, _BLUE, positive=True)
        _style(ax, title, ylabel, "log")
    written.append(_finish(fig, axes, len(panels), out_dir / "score.png"))
    plt.close(fig)

    observed = sorted({name for r in records for name in r["observed"]})
    if observed:
        cols = min(3, len(observed))
        rows = -(-len(observed) // cols)
        fig, axes = _figure(
            plt, rows, cols, "Observed energies on x̂₀ (logged, not applied)",
            subtitle + "  ·  unweighted, mean over particles",
        )
        for ax, name in zip(axes, observed):
            _draw(ax, records, lambda r, n=name: r["observed"].get(n), num_steps, _BLUE)
            values = [r["observed"][name] for r in records if name in r["observed"]]
            note = "  (0 at every step)" if not any(values) else ""
            unit = UNITS.get(name, "?")
            _style(ax, name.removesuffix("Potential") + note, f"Σ violation ({unit}, summed over terms)", "symlog")
        written.append(_finish(fig, axes, len(observed), out_dir / "observed_energies.png"))
        plt.close(fig)

    steered = sorted({n for r in records for row in r["inner"] for n in row["energies"]})
    if steered:

        def first(record, name):
            rows = [row for row in record["inner"] if name in row["energies"]]
            return rows[0]["energies"][name][0] if rows else None

        def last(record, name):
            rows = [row for row in record["inner"] if name in row["energies"]]
            return rows[-1]["energies"][name][0] if rows else None

        cols = min(3, len(steered))
        rows = -(-len(steered) // cols)
        fig, axes = _figure(
            plt, rows, cols, "Guidance energies per denoising step",
            subtitle + "  ·  unweighted, mean over particles; gaps = potential inactive",
        )
        for ax, name in zip(axes, steered):
            _draw(ax, records, lambda r, n=name: first(r, n), num_steps, _BLUE, "before guidance (first inner step)")
            _draw(ax, records, lambda r, n=name: last(r, n), num_steps, _ORANGE, "after guidance (last inner step)")
            values = [first(r, name) for r in records] + [last(r, name) for r in records]
            note = "  (0 at every step)" if not any(v for v in values if v is not None) else ""
            _style(ax, name.removesuffix("Potential") + note, "energy", "symlog")
        written.append(_finish(fig, axes, len(steered), out_dir / "guidance_energies.png", legend=True))
        plt.close(fig)

        def shift(record):
            return record["inner"][-1]["shift_mean"] if record["inner"] else None

        fig, axes = _figure(plt, 3, 1, "Guidance vs the network's own update", subtitle)
        _draw(axes[0], records, lambda r: r["disp"], num_steps, _BLUE, "denoiser |x̂₀ − x_t|", positive=True)
        _draw(axes[0], records, shift, num_steps, _ORANGE, "guidance shift (after last inner step)", positive=True)
        _style(axes[0], "Displacement per step", "Å, mean per atom", "log")
        _panel_legend(axes[0])
        _draw(
            axes[1], records,
            lambda r: shift(r) / r["disp"] if r["inner"] and r["disp"] > 0 else None,
            num_steps, _BLUE, positive=True,
        )
        _style(axes[1], "Guidance share: shift / |x̂₀ − x_t|", "ratio", "log")
        _draw(axes[2], records, lambda r: r["inner"][0]["total"] if r["inner"] else None, num_steps,
              _BLUE, "before guidance (first inner step)")
        _draw(axes[2], records, lambda r: r["inner"][-1]["total"] if r["inner"] else None, num_steps,
              _ORANGE, "after guidance (last inner step)")
        _style(axes[2], "Weighted total Σ w·E being descended", "energy", "symlog")
        _panel_legend(axes[2])
        written.append(_finish(fig, axes, 3, out_dir / "guidance_vs_score.png"))
        plt.close(fig)

    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a steering_log test.txt.")
    parser.add_argument("log", help="path to test.txt")
    parser.add_argument("out_dir", nargs="?", help="default: <log folder>/plots")
    parser.add_argument(
        "--combined", action="store_true", help="one set of plots, mean over batches, instead of per batch"
    )
    args = parser.parse_args()
    for written_path in plot_steering_log(args.log, args.out_dir, per_batch=not args.combined):
        print(written_path)
