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
            "trjcat",
            "eneconv",
            "convert-tpr",
            "report-methods",
            "pdb2bond",
        ),
    ),
    (
        "Analyze structure and contacts",
        "Measure conformations, contacts, correlations, and assemblies.",
        (
            "gyrate",
            "polystat",
            "ss",
            "angle",
            "distance",
            "rdf",
            "contact",
            "contact.ng",
            "timecorr",
            "cstat",
            "rmsd",
            "rmsf",
            "sasa",
            "assembly",
            "pbcontact",
        ),
    ),
    (
        "Analyze phase behavior and transport",
        "Measure slab phases, molecular motion, interfaces, and viscosity.",
        (
            "density",
            "densmap",
            "coexistence",
            "exchange",
            "phase-msd",
            "msd",
            "surftension",
            "viscosity-gk",
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
        "contact or contact.ng",
        "Use contact for the standard contact-map workflow. Use contact.ng "
        "for large trajectories when memory-efficient aggregation is needed.",
    ),
    (
        "msd or phase-msd",
        "Use msd for ordinary displacement of one selected group. Use "
        "phase-msd for dense/interface/dilute motion in a planar slab.",
    ),
    (
        "rmsd, rmsf, gyrate, or polystat",
        "Use rmsd for conformational change against one reference, rmsf for "
        "time-averaged positional flexibility, gyrate for legacy radius-of-"
        "gyration workflows, and polystat for size and shape across many "
        "protein copies.",
    ),
    (
        "density, densmap, or coexistence",
        "Use density for a one-dimensional profile, densmap for a planar 2D "
        "map of droplets or laterally heterogeneous interfaces, and coexistence "
        "for fitted slab densities and interface widths.",
    ),
    (
        "distance scopes",
        "Use distance --scope intra-chain or inter-chain after selecting each "
        "region once across all molecule copies.",
    ),
    (
        "contact, timecorr, or cstat",
        "Use contact to create contact maps, timecorr for contact lifetimes, "
        "and cstat to summarize an already computed contact map.",
    ),
)


COMMAND_EXAMPLES = {
    "pdb2dps": (
        "dps pdb2dps -s MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW "
        "-o chain.pdb -op chain.itp",
    ),
    "modifyres": (
        "dps modifyres -p chain.itp -f chain.pdb "
        "-op modified.itp -o modified.pdb "
        "-m S129SMP",
    ),
    "genelastic": (
        "dps genelastic -f chain.pdb -p chain.itp -o elastic.itp "
        "-r elastic-groups.dat",
    ),
    "addangle": (
        "dps addangle -p chain.itp -op chain-angles.itp -a angles.dat",
    ),
    "genmesh": (
        "dps genmesh -f chain.pdb -p chain.itp -n 100 "
        "-o system.pdb -op system.top",
    ),
    "editconf": (
        "dps editconf -f system.pdb -o resized.pdb "
        "--multiply-x-axis 2 --multiply-y-axis 2 --multiply-z-axis 2",
    ),
    "grompp": (
        "dps grompp -f system.pdb -p system.top -m md.mdp -o run.tpr",
    ),
    "mdrun": (
        "dps mdrun -s run.tpr -o run",
        "dps mdrun -s run.tpr -o run --checkpoint",
    ),
    "rerun": (
        "dps rerun -s run.tpr -f run.xtc -o rerun.edr",
    ),
    "check": (
        "dps check -s run.tpr -f run.xtc -n index.ndx",
    ),
    "energy": (
        "dps energy -f run.edr --list",
        "dps energy -f run.edr -o thermo.xvg --terms temperature volume",
    ),
    "make_ndx": (
        "dps make_ndx -s run.tpr -o index.ndx",
    ),
    "trjconv": (
        "dps trjconv -s run.tpr -f run.xtc -o frame.pdb "
        "-b 100 -e 100 --select 'group 0'",
        "dps trjconv -s run.tpr -f run.xtc -o centered.xtc "
        "-se 'group 0' -p mol -c mass -ca z",
    ),
    "gsd2xtc": (
        "dps gsd2xtc -f run.gsd -o run.xtc",
    ),
    "trjcat": (
        "dps trjcat -f run.xtc run.part0002.xtc -o complete.xtc",
    ),
    "eneconv": (
        "dps eneconv -f run.edr run.part0002.edr -o complete.edr",
    ),
    "convert-tpr": (
        "dps convert-tpr -s legacy.tpr -o portable.tpr",
    ),
    "report-methods": (
        "dps report-methods -s run.tpr -o methods.out -m methods.tex",
    ),
    "pdb2bond": (
        "dps pdb2bond -s run.tpr -f frame.pdb -o bonded.pdb",
    ),
    "gyrate": (
        "dps gyrate -s run.tpr -f run.xtc --output-average rg-average.xvg",
        "dps gyrate -s slab.tpr -f slab.xtc -n groups.ndx "
        "--output-phase phase-rg.xvg --reference-group 0 "
        "--selection-calculate 1 --axis z",
    ),
    "polystat": (
        'dps polystat -s run.tpr -f run.xtc -o polystat.xvg '
        '--select "mol PROT"',
        "dps polystat -s run.tpr -f run.xtc -o polystat.xvg "
        "--selection-groups 1 "
        "-i internal-distance.csv -p persistence.xvg",
    ),
    "ss": (
        'dps ss -s run.tpr -f run.xtc -o ss-state.csv --select "mol PROT" '
        '--map-output ss-map.xpm --number-output ss-count.xvg '
        '--occupancy-output ss-occupancy.xvg',
        'dps ss -s run.tpr -f run.xtc -o ss-paired.csv --select "mol PROT" '
        '--strand-pairing interchain --strand-pair-output strand-pairs.csv',
    ),
    "angle": (
        "dps angle -s run.tpr -f run.xtc "
        "--output-residue angles-by-residue.xvg",
    ),
    "distance": (
        "dps distance -s run.tpr -f run.xtc --mode pair "
        "--scope intra-chain --reference 1 --selection 2 "
        "--output-statistics intra.csv",
        "dps distance -s run.tpr -f run.xtc --mode pair "
        "--scope inter-chain --reference 1 --selection 2 "
        "--output-statistics inter.csv",
    ),
    "rdf": (
        "dps rdf -s run.tpr -f run.xtc --reference 1 --selection 2 -o rdf.xvg",
        "dps rdf -s run.tpr -f run.xtc --reference 1 --selection 1 "
        "-o rdf-com.xvg "
        "--position com --grouping chain --exclude same-molecule",
    ),
    "contact": (
        "dps contact -s run.tpr -f run.xtc --reference-group 0 "
        "--selection-group 1 --cutoff-scheme global "
        "--output-inter-reference-selection contacts.dat",
    ),
    "contact.ng": (
        "dps contact.ng -s run.tpr -f run.xtc --reference-group 0 "
        "--selection-group 1 --cutoff-scheme global "
        "--output-inter-reference-selection contacts.dat --search-mode stream",
    ),
    "timecorr": (
        "dps timecorr -s run.tpr -f run.xtc --reference-group 0 "
        "--selection-group 1 --cutoff-scheme global "
        "--output-corr-inter-reference-selection contact-correlation.dat",
    ),
    "cstat": (
        "dps cstat -m contacts.dat -s run.tpr -o contact-statistics.xlsx",
    ),
    "assembly": (
        "dps assembly -s run.tpr -f run.xtc "
        "--cluster-number clusters.xvg --cluster-size largest.xvg",
    ),
    "pbcontact": (
        "dps pbcontact -s run.tpr -f run.xtc -o periodic-images",
    ),
    "density": (
        "dps density -s run.tpr -f run.xtc -o density.xvg -x z",
    ),
    "densmap": (
        "dps densmap -s run.tpr -f run.xtc -o densmap.dat "
        "--selection 0 --average-axis z --type number",
        "dps densmap -s run.tpr -f run.xtc -o droplet.dat "
        "--selection 1 2 --type mass --center-mode frame --selection-fit 1",
    ),
    "coexistence": (
        "dps coexistence -s run.tpr -f run.xtc -o coexistence.xvg --axis z",
    ),
    "exchange": (
        "dps exchange -s run.tpr -f run.xtc -o exchange --axis z",
    ),
    "phase-msd": (
        "dps phase-msd -s run.tpr -f run.xtc -o phase-msd.xvg --axis z",
    ),
    "msd": (
        "dps msd -s run.tpr -f run.xtc -o msd.xvg -t xyz",
    ),
    "rmsd": (
        'dps rmsd -s run.tpr -f run.xtc -o rmsd.xvg '
        '--select "(mol PROT) & (resid 20-80)"',
        "dps rmsd -s run.tpr -f run.xtc -o rmsd-system.xvg "
        "--selection-groups 1 "
        "--fit-mode selection --output-mode selection",
    ),
    "rmsf": (
        'dps rmsf -s run.tpr -f run.xtc -o rmsf.xvg '
        '--select "(mol PROT) & (resid 20-80)" --residue',
        "dps rmsf -s run.tpr -f run.xtc -o rmsf.xvg "
        "--selection-groups 1 --output-deviation reference-deviation.xvg "
        "--output-average average.pdb",
    ),
    "sasa": (
        "dps sasa -s run.tpr -f run.xtc -o sasa.xvg "
        "--surface 0 --output-selection 1 2",
    ),
    "surftension": (
        "dps surftension -s run.tpr -f run.xtc -o surface-tension.xvg --axis z",
    ),
    "viscosity-gk": (
        "dps viscosity-gk -f run.stress.xvg -o viscosity.xvg",
    ),
}


def format_examples(command_name: str) -> str | None:
    """Return a ready-to-render Examples section for one command."""

    examples = COMMAND_EXAMPLES.get(command_name)
    if examples is None:
        return None
    return "Examples:\n" + "\n".join(f"  {example}" for example in examples)
