"""Practical command notes shared by terminal help and the complete reference."""

from __future__ import annotations


COMMAND_NOTES = {
    "help": (
        "Use 'dps help commands' for the complete task catalog, and 'dps help COMMAND' for inputs, outputs, defaults, choices, and examples.",
        "Long options are the stable interface for scripts. Group IDs and DROPPS NDX atom indices start at zero; PDB and ITP bead IDs start at one.",
    ),
    "pdb2dps": (
        "Provide a one-letter sequence and at least one of --output-conformation (PDB) or --output-topology (ITP). Set --forcefield to avoid the interactive prompt.",
        "An optional all-atom PDB supplies the first model's C-alpha positions; its C-alpha count must match the sequence. Otherwise --seed controls random chain generation and --max-attempts bounds retries.",
        "--radius is a construction confinement radius in nm, not a measured radius of gyration. Modification numbers refer to the residue numbering set by --residue-index.",
    ),
    "modifyres": (
        "Read a matched PDB/ITP pair and write both modified files. Select a force field that defines the requested modification, for example S129SMP.",
        "The current modification implementation addresses 1-based bead positions and cannot modify terminal residues. Regenerate the system TOP/TPR after changing topology.",
    ),
    "genelastic": (
        "The elastic-residue file contains one whitespace-separated group of 1-based bead indices per line. Bonds are added between eligible bead pairs within each group.",
        "Distances and cutoffs are in nm; the elastic force constant is in kJ mol^-1 nm^-2. The output is an ITP with the added bonds; retain the matching PDB.",
    ),
    "addangle": (
        "Each angle-list row is BEAD_ID ANGLE_DEG FORCE_CONSTANT. BEAD_ID is the 1-based central bead of a consecutive three-bead angle.",
        "Angles are supplied in degrees; force constants use kJ mol^-1 rad^-2. This command edits the ITP; 'angle' measures angles in a trajectory.",
    ),
    "genmesh": (
        "Supply equally long --structure, --topology, and --number lists. Each PDB must match its ITP bead count and molecule definition.",
        "--mesh sets NX NY NZ slots, --gap sets the intermolecular gap in nm, and --box-type chooses anisotropy, xy-equal, or cubic box lengths. --shuffle uses --seed.",
        "The output PDB and TOP describe the same molecule ordering. ITP paths in the TOP are relative to the TOP directory; keep those files together when moving a system.",
    ),
    "editconf": (
        "Read a periodic PDB and write an enlarged box with translated coordinates. Use either an absolute length in nm or a multiplier for each axis.",
        "Expanded axes are unwrapped automatically. Unchanged axes are unwrapped only when requested. A changed box requires a new 'grompp' run before simulation.",
    ),
    "grompp": (
        "Compile a matching coarse-grained PDB, system TOP, referenced ITP files, and MDP settings into a checksummed portable TPR v2.",
        "TOP sections are [ itp files ] and [ system ]; the latter lists MOLECULE_NAME COUNT in coordinate order. ITP paths are resolved relative to the TOP.",
        "Box lengths and cutoffs use nm. Each nonbonded cutoff must be strictly less than half the shortest box length. Use the bundled md.mdp template and MDP reference for supported settings.",
    ),
    "convert-tpr": (
        "Convert a trusted legacy pickle TPR into ZIP/XML/JSON/NumPy TPR v2. Legacy pickle loading can execute code, so use files from a trusted source.",
        "For an old OpenMM checkpoint, first export a portable State using its original environment. Converting the TPR alone does not convert the native checkpoint.",
    ),
    "mdrun": (
        "A new Langevin run minimizes/warms as configured, then integrates to the total production nsteps. --nsteps sets the total target, including steps already completed on resume.",
        "--checkpoint resumes from the output prefix or an explicit restart path. Append mode rolls outputs back to the checkpoint; --no-append writes a new part. Native checkpoints have priority over portable State fallback.",
        "The output prefix produces log, DROPPS CSV EDR, XTC, checkpoint, State XML, restart JSON, and final PDB. Optional stress output contains the pressure tensor for viscosity analysis.",
        "Set --platform CUDA when GPU execution is required. integrator=steep runs only OpenMM energy minimization and writes the final structure/restart without a dynamics trajectory.",
    ),
    "rerun": (
        "Re-evaluate saved XTC coordinates using the system stored in TPR; this does not integrate new dynamics. Output is a DROPPS EDR table.",
        "Configurational pressure terms omit kinetic and constraint virials. Use runtime stress output, rather than these terms, for Green-Kubo viscosity.",
    ),
    "check": (
        "Inspect TPR, XTC, and optional NDX to confirm atom counts, topology, frame times, box geometry, and available groups before analysis.",
        "TPR and EDR are DROPPS formats; files produced by GROMACS with the same extensions are not interchangeable.",
    ),
    "make_ndx": (
        "Build an index interactively from TPR. Expressions support mol, name, abbr, res, resid, chainid, index, group, parentheses, & and |.",
        "DROPPS atom and group indices are zero-based. Enter an expression to add a group, 'name NUMBER LABEL' to rename, 'splitch NUMBER' to split by molecule, and 'q' to save.",
    ),
    "energy": (
        "Read the versioned CSV table written by DROPPS mdrun/rerun. --list displays available terms; --terms selects the columns to export as XVG.",
        "The output header records units. Potential/kinetic/total energies use kJ/mol, temperature K, box lengths nm, volume nm^3, and pressure bar where available.",
    ),
    "trjconv": (
        "Process XTC using the TPR topology and write XTC or PDB. --select chooses output atoms; centering and fitting may use separate selections.",
        "Time bounds use --time-unit (ns by default). Equal start/end selects the nearest saved frame. PBC modes include whole molecules, wrapping, and temporal no-jump tracking.",
        "No-jump and progressive fitting visit intermediate frames before output subsampling. A subset trajectory requires a matching subset topology for later analysis.",
    ),
    "gsd2xtc": (
        "Convert a GSD trajectory whose coordinates are expressed in nm to XTC. The GSD box is used unless --box overrides its lengths in nm.",
        "--time-step is the spacing between saved frames in ns; output starts at time zero. Confirm the units used by the GSD producer before conversion.",
    ),
    "trjcat": (
        "Concatenate XTC segments with identical atom counts. Inputs are ordered by their first timestamp by default; overlapping timestamps prefer the later segment.",
        "Use --keep-duplicates only when repeated timestamps are intended. Time selection uses --time-unit, and the destination is replaced atomically after successful writing.",
    ),
    "eneconv": (
        "Merge DROPPS CSV EDR segments with matching columns and units. This command does not read binary GROMACS EDR.",
        "Overlapping times prefer the later segment unless duplicate retention is enabled. Check chronology and step numbers before combining unrelated runs.",
    ),
    "report-methods": (
        "Generate text and optional LaTeX methods from TPR settings and provenance. The report does not read restart metadata or record the actual runtime platform.",
        "The report describes stored inputs; supply run overrides and subsequent analysis choices when preparing a complete experimental record.",
    ),
    "pdb2bond": (
        "Add PDB CONECT records from either a TPR or TOP/ITP topology. The input PDB bead order must match the supplied topology.",
        "Bond records improve structure display in molecular viewers. They do not change coordinates or simulation forces.",
    ),
    "density": (
        "Calculate one-dimensional number, mass, or charge profiles along the selected axis. Supply --selection-fit and --selection-calculate to avoid interactive selection.",
        "Per-frame box lengths and volumes determine density normalization. Frame/block/none centering controls alignment of a drifting slab; the output header specifies coordinates and density units.",
    ),
    "densmap": (
        "Average along one axis and histogram the remaining plane into a two-dimensional density map. Choose number, mass, or charge and one or more selection groups.",
        "The text outputs include grid coordinates, dimensions, units, and normalization. PNG output requires the plot extra. Centering can track a moving droplet before averaging.",
    ),
    "coexistence": (
        "Fit a centered planar-slab mass-density profile with a double-tanh interface model to obtain dense/dilute densities and interface widths.",
        "Use a reference group that identifies the slab, inspect the profile and fit, and choose contiguous blocks for uncertainty estimation. This geometry assumes a planar slab, not a spherical droplet.",
    ),
    "exchange": (
        "Classify molecular centers as dense/interface/dilute using fitted slab boundaries, then calculate phase transitions, survival, and residence statistics.",
        "The output prefix produces the population/transition and residence data plus the reference profile. Time windows are in ns; trajectories need at least two frames and adequate temporal resolution for crossings.",
    ),
    "phase-msd": (
        "Calculate molecular displacement by slab phase and residence-conditioning rule. Select which molecule groups to track and how to remove phase translation.",
        "Output includes lag-time MSD curves in nm^2, detailed statistics, and the fitted reference density profile. Uniformly spaced frames are required for lag-based analysis.",
    ),
    "gyrate": (
        "Measure mass-weighted radius of gyration in nm for selected groups, with time series, distributions, and per-chain details controlled by output options.",
        "--output-phase adds per-molecule dense/interface/dilute Rg, block summaries, occupancy observations, and slab-fit profiles. Use polystat for an explicit many-copy size/shape cohort analysis.",
    ),
    "polystat": (
        "Split selected molecules into compatible cohorts and measure per-chain Rg, end-to-end length, gyration-tensor shape, and optional internal-distance/persistence statistics.",
        "The compact XVG reports cohort summaries; optional CSV/XVG files retain per-molecule observations, tangent correlations, or internal distances. Lengths use nm; inspect censored or invalid persistence estimates.",
    ),
    "ss": (
        "Assign coarse-grained backbone geometry states and optional strand pairings. States describe the bead geometry; they are not all-atom hydrogen-bond assignments.",
        "The main CSV contains state observations; optional XPM, state counts, occupancies, and pairing tables describe residue/time patterns. Use compatible sequential backbone selections.",
    ),
    "angle": (
        "Measure consecutive intrachain bead angles for a selected group. Chain-terminal beads have no centered triplet and are excluded.",
        "Choose at least one output for the time series, residue profile, standard deviation, or verbose per-chain values. Output angles are in degrees and trajectory time is in ns.",
    ),
    "distance": (
        "Select each region once across all molecule copies, then choose intra-chain or inter-chain scope and pair or center-based measurement.",
        "Distances use nm. Select output statistics, time series, or pair details as needed; PBC and pair matching are explicit options. Hidden intra-distance/inter-distance and idist/odist aliases retain the older interfaces.",
    ),
    "rdf": (
        "Measure radial distributions between atoms or grouped centers with 3D spherical or xy planar geometry, optional exclusions, and cumulative coordination numbers.",
        "RDF normalization is dimensionless; number-density and raw-count modes have different meanings. Grouping, surface selection, and exclusion rules affect the denominator and must match the scientific question.",
    ),
    "contact": (
        "Compute residue-level intra/interchain maps or contact-count time series. Select global nm cutoffs or residue-sigma cutoffs and choose at least one output.",
        "Choose reference and selection groups with consistent chain layouts for residue averaging. Enable periodic-boundary distances when analyzing wrapped molecules.",
    ),
    "contact.ng": (
        "Compute the contact workflow with neighbor search and streaming aggregation for larger systems. Cutoff definitions and output choices are shared with contact.",
        "--search-mode chooses the computation strategy; the map remains a residue-level aggregate. Use time-series outputs when temporal changes are the quantity of interest.",
    ),
    "timecorr": (
        "Calculate contact time correlations/lifetimes from a uniformly sampled trajectory with the chosen reference/selection groups and contact definition.",
        "Specify the required cutoff scheme and correlation output. The maximum lag and saved-frame interval determine the time range that can be resolved.",
    ),
    "cstat": (
        "Summarize a precomputed contact map using the matching TPR and reference/selection group layout. Output is an Excel workbook.",
        "Use the same group order and averaging convention as the command that generated the map; a matrix of the right size alone does not establish matching residue identities.",
    ),
    "msd": (
        "Compute the time-origin-averaged atomic mean-square displacement in the requested Cartesian directions. Output is lag time in ns versus MSD in nm^2.",
        "All saved frames are unwrapped before applying time-window/stride selection. Frames must be uniformly sampled, and motion between saved frames must be small enough to determine the periodic image.",
        "This is atomic MSD without center-of-mass drift subtraction. Use phase-msd for molecular motion conditioned on slab phase.",
    ),
    "rmsd": (
        "Compare selected coordinates with an explicit reference frame/time, optionally after translation/rotation fitting and molecular PBC reconstruction.",
        "Fit and measurement selections can differ. Per-molecule and whole-selection outputs answer different questions; all RMSD distances are in nm.",
    ),
    "rmsf": (
        "Measure time-averaged positional fluctuations after the chosen PBC/fitting procedure, either per particle/residue or pooled across compatible molecules.",
        "RMSF uses nm; optional outputs include the mean structure and deviation from a reference. Translation/rotation left in the coordinates contributes to the fluctuation.",
    ),
    "sasa": (
        "Estimate coarse-grained solvent-accessible surface area by sampling expanded bead spheres. Choose the surface group, output groups, probe radius, and sphere resolution.",
        "Areas use nm^2. Occlusion uses the full selected surface even when reporting a smaller output group; force-field radii and probe size determine the physical interpretation.",
    ),
    "assembly": (
        "Build a graph of molecular groups connected by enough bead contacts, then report cluster count, size, composition, and optional shape measures.",
        "Cutoffs are in nm. Enable PBC for periodic contacts; shape reconstruction uses contact-connected images. Specify output options explicitly and inspect percolating clusters before interpreting finite-cluster shape.",
    ),
    "pbcontact": (
        "Detect contacts between molecules and periodic images to assess finite-box self-interaction. The output prefix produces detailed and summarized interaction data.",
        "Interpret the results together with chain extension and box size. This diagnostic complements ordinary minimum-image contact maps.",
    ),
    "surftension": (
        "Estimate planar interfacial tension by finite-strain configurational energy evaluation along the interface normal. Choose the number of interfaces and strain values consistently with the slab geometry.",
        "Output is in mN/m with block uncertainty and strain-convergence information. Inspect convergence and the pressure/geometry assumptions before comparing state points.",
    ),
    "viscosity-gk": (
        "Integrate stress autocorrelations from the full runtime pressure tensor to obtain Green-Kubo shear viscosity. Use uniformly sampled equilibrated stress data.",
        "The running integral and plateau estimate are in mPa s. Select a plateau window and blocks; output from configurational rerun lacks the full tensor required here.",
    ),
}


def format_notes(command_name: str) -> str | None:
    notes = COMMAND_NOTES.get(command_name)
    if notes is None:
        return None
    return "Notes:\n" + "\n\n".join("  " + note for note in notes)
