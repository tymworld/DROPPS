"""User-facing command catalog, selection guidance, and CLI examples."""

from __future__ import annotations


TASK_GROUPS = (
    (
        "Build a simulation system",
        "Create molecular models, pack a box, and compile the run input.",
        (
            "pdb2dps",
            "modifyres",
            "genelastic",
            "addangle",
            "genmesh",
            "editconf",
            "grompp",
        ),
    ),
    (
        "Run or recompute a simulation",
        "Integrate a trajectory, resume a run, or recompute saved observables.",
        (
            "mdrun",
            "rerun",
        ),
    ),
    (
        "Inspect and convert files",
        "Check inputs, create selections, process trajectories, and convert formats.",
        (
            "check",
            "energy",
            "make_ndx",
            "trjconv",
            "gsd2xtc",
            "convert-tpr",
            "pdb2bond",
        ),
    ),
    (
        "Analyze structure and contacts",
        "Measure conformations, contacts, correlations, and assemblies.",
        (
            "gyrate",
            "angle",
            "idist",
            "odist",
            "contact",
            "cstat",
            "rmsd",
            "assembly",
        ),
    ),
    (
        "Analyze phase behavior and transport",
        "Measure density, molecular exchange, and motion.",
        (
            "density",
            "exchange",
            "msd",
        ),
    ),
)


START_HERE = (
    (
        "New system from a protein sequence",
        "pdb2dps -> genmesh -> grompp -> mdrun",
    ),
    (
        "Prepared structure and topology",
        "grompp -> mdrun",
    ),
    (
        "Before trajectory analysis",
        "check -> make_ndx (when custom groups are needed) -> analysis command",
    ),
)


COMMAND_CHOICES = (
    (
        "mdrun or rerun",
        "Use mdrun to integrate or resume dynamics. Use rerun to recompute "
        "configurational observables from an existing XTC trajectory.",
    ),
    (
        "rmsd or gyrate",
        "Use rmsd for conformational change with optional per-molecule or "
        "whole-selection fitting. Use gyrate for molecular size and shape.",
    ),
    (
        "idist or odist",
        "Use idist for bead-pair distances within equivalent chains. Use "
        "odist for corresponding beads between two chain groups.",
    ),
    (
        "contact or cstat",
        "Use contact to create contact maps and cstat to summarize an already "
        "computed contact map.",
    ),
)


COMMAND_EXAMPLES = {
    "pdb2dps": (
        "dps pdb2dps -s MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW "
        "-oc chain.pdb -op chain.itp",
    ),
    "modifyres": (
        "dps modifyres -ip chain.itp -if chain.pdb -op modified.itp "
        "-of modified.pdb -m S129SMP",
    ),
    "genelastic": (
        "dps genelastic -f chain.pdb -p chain.itp -o elastic.itp "
        "-er elastic-groups.dat",
    ),
    "addangle": ("dps addangle -ip chain.itp -op chain-angles.itp -al angles.dat",),
    "genmesh": (
        "dps genmesh -f chain.pdb -p chain.itp -n 100 -oc system.pdb -op system.top",
    ),
    "editconf": ("dps editconf -f system.pdb -o resized.pdb -mx 2 -my 2 -mz 2",),
    "grompp": ("dps grompp -f system.pdb -p system.top -m md.mdp -o run.tpr",),
    "mdrun": (
        "dps mdrun -s run.tpr -o run",
        "dps mdrun -s run.tpr -o run -cpi",
    ),
    "rerun": ("dps rerun -s run.tpr -f run.xtc -o rerun.edr",),
    "check": ("dps check -s run.tpr -f run.xtc -n index.ndx",),
    "energy": (
        "dps energy -f run.edr --list",
        "dps energy -f run.edr -o thermo.xvg --terms temperature volume",
    ),
    "make_ndx": ("dps make_ndx -s run.tpr -o index.ndx",),
    "trjconv": (
        "dps trjconv -s run.tpr -f run.xtc -o frame.pdb -b 100 -e 100 -sel 'group 0'",
    ),
    "gsd2xtc": ("dps gsd2xtc -f run.gsd -o run.xtc",),
    "convert-tpr": ("dps convert-tpr -s legacy.tpr -o portable.tpr",),
    "pdb2bond": ("dps pdb2bond -s run.tpr -f frame.pdb -o bonded.pdb",),
    "gyrate": ("dps gyrate -s run.tpr -f run.xtc -oa rg-average.xvg",),
    "angle": ("dps angle -s run.tpr -f run.xtc -or angles-by-residue.xvg",),
    "idist": ("dps idist -s run.tpr -f run.xtc -op intrachain-distances.xvg",),
    "odist": ("dps odist -s run.tpr -f run.xtc -oa interchain-distances.xvg",),
    "contact": (
        "dps contact -s run.tpr -f run.xtc -ref 0 -sel 1 -cs global -ors contacts.dat",
    ),
    "cstat": ("dps cstat -m contacts.dat -s run.tpr -o contact-statistics.xlsx",),
    "assembly": (
        "dps assembly -s run.tpr -f run.xtc -cn clusters.xvg -cs largest.xvg",
    ),
    "density": ("dps density -s run.tpr -f run.xtc -o density.xvg -x z",),
    "exchange": ("dps exchange -s run.tpr -f run.xtc -o exchange --axis z",),
    "msd": ("dps msd -s run.tpr -f run.xtc -o msd.xvg -t xyz",),
    "rmsd": (
        "dps rmsd -s run.tpr -f run.xtc -o rmsd.xvg "
        '--select "(mol PROT) & (resid 20-80)"',
        "dps rmsd -s run.tpr -f run.xtc -o rmsd-system.xvg -sel 1 "
        "--fit-mode selection --output-mode selection",
    ),
}


def format_examples(command_name: str) -> str | None:
    """Return a ready-to-render Examples section for one command."""

    examples = COMMAND_EXAMPLES.get(command_name)
    if examples is None:
        return None
    return "Examples:\n" + "\n".join(f"  {example}" for example in examples)
