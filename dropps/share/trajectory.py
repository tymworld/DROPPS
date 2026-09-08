import MDAnalysis as mda
import numpy as np
from openmm.unit import nanometer, picosecond
import os

from dropps.fileio.tpr_reader import read_tpr
from dropps.share.indexing import indexGroups
from dropps.share.time_selection import select_time_indices

from os.path import splitext


def printcolor(text, success):
    if success:
        print(f"\033[32m{text}\033[0m")
    else:
        print(f"\033[5;34;46m{text}\033[0m")


class trajectory_class:
    def __init__(self, tpr_path, index_path=None, trajectory_path=None):
        self.tpr = read_tpr(tpr_path)
        self.topology = self.tpr.mdtopology

        print(f"## Loaded topology from {tpr_path}.")

        positions_array = np.array(
            [
                [coor.value_in_unit(nanometer) for coor in atom]
                for atom in self.tpr.positions
            ]
        )

        self.Universe = mda.Universe(self.tpr.mdtopology, positions_array)

        # We generate charge lists

        charges = [atom.charge for itp in self.tpr.itp_list for atom in itp.atoms]
        masses = [atom.mass for itp in self.tpr.itp_list for atom in itp.atoms]

        if self.tpr.parameters["vdwtype"] == "pLJ":
            sigmas = [
                itp.type2sigma[atom.abbr]
                for itp in self.tpr.itp_list
                for atom in itp.atoms
            ]
            mylambdas = [
                itp.type2mylambda[atom.abbr]
                for itp in self.tpr.itp_list
                for atom in itp.atoms
            ]

            self.mylambdas = mylambdas
            self.sigmas = sigmas

        self.Universe.add_TopologyAttr("charges", charges)
        self.Universe.add_TopologyAttr("masses", masses)

        print(
            f"## Generating Universe from topology and coordinates loaded from {tpr_path}."
        )

        self.index = indexGroups(self.tpr.mdtopology, self.tpr.itp_list)
        print(f"## Index initilized by {tpr_path}.")

        if index_path is not None:
            self.index.load_ndx(index_path)
            print(f"## Known index entries loaded from {index_path}.")

        if trajectory_path is not None:
            trajectory_path = os.path.abspath(os.path.expanduser(trajectory_path))
            if not os.path.isfile(trajectory_path):
                raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

            file_extention = splitext(trajectory_path)[1].lower()

            if file_extention in {".xtc", ".pdb"}:
                print(
                    f"Loading {file_extention[1:].upper()} trajectory file {trajectory_path} ..."
                )
                try:
                    self.Universe.load_new(trajectory_path, in_memory=False)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to read trajectory '{trajectory_path}'. "
                        "Please check whether the file is complete/corrupted and "
                        "compatible with the provided topology."
                    ) from exc
                print(f"## Loaded trajectory file {trajectory_path} into Universe.")

            else:
                raise ValueError(
                    f"Unsupported trajectory extension '{file_extention}'; "
                    "DROPPS trjconv accepts .xtc and .pdb input."
                )

        # We generate information for each atoms.
        self.id2charge = charges
        self.id2masses = masses

        chain_length_list = [len(itp.atoms) for itp in self.tpr.itp_list]
        self.id2chainID = [
            i for i, length in enumerate(chain_length_list) for _ in range(length)
        ]
        self.id2resID = [
            atom.residueid for itp in self.tpr.itp_list for atom in itp.atoms
        ]
        print(
            "## The next line contains the first trajectory frame for a sanity check."
        )
        print(f"## {self.Universe.trajectory[0]}")

    def get_chainID(self, index):
        if index < 0 or index >= self.num_atoms():
            print(
                f"ERROR: System of {self.num_atoms()} atoms doesn't contains atom index {index}."
            )
            quit()
        return self.id2chainID[index]

    def get_resID(self, index):
        if index < 0 or index >= self.num_atoms():
            print(
                f"ERROR: System of {self.num_atoms()} atoms doesn't contains atom index {index}."
            )
            quit()
        return self.id2resID[index]

    def is_terminal(self, index):
        return (
            index == 0
            or index == self.num_atoms() - 1
            or self.get_chainID(index - 1) != self.get_chainID(index)
            or self.get_chainID(index + 1) != self.get_chainID(index)
        )

    def num_frames(self):
        return len(self.Universe.trajectory)

    def num_atoms(self):
        return len(self.Universe.atoms)

    def num_chains(self):
        return self.topology.getNumChains()

    def num_bonds(self):
        return len(self.Universe.bonds)

    def time_init(self):
        return self.Universe.trajectory[0].time * picosecond

    def time_end(self):
        return self.Universe.trajectory[-1].time * picosecond

    def time_step(self):
        return self.Universe.trajectory.dt * picosecond

    def time2frame(self, time_start=None, time_end=None, time_step=None):
        """Convert a physical-time window to an inclusive constant-stride slice.

        Bounds are selected from the trajectory's actual timestamps.  The
        analysis API represents sampling as one frame stride, so an explicit
        physical interval additionally requires uniformly spaced timestamps in
        the selected window.
        """

        frame_times_ps = np.asarray(
            [float(timestep.time) for timestep in self.Universe.trajectory],
            dtype=np.float64,
        )
        selection = select_time_indices(
            frame_times_ps,
            start_time=time_start,
            end_time=time_end,
            time_unit="ns",
        )
        start_frame = int(selection.indices[0])
        end_frame = int(selection.indices[-1])
        interval_frame = 1

        if time_step is not None:
            interval_ps = float(time_step) * 1000.0
            if not np.isfinite(interval_ps) or interval_ps <= 0.0:
                raise ValueError("Analysis time interval must be greater than zero.")

            selected_times = frame_times_ps[start_frame : end_frame + 1]
            if selected_times.size > 1:
                frame_intervals = np.diff(selected_times)
                if np.any(frame_intervals <= 0.0):
                    raise ValueError(
                        "A fixed analysis interval requires strictly increasing "
                        "trajectory timestamps."
                    )
                trajectory_interval = float(np.median(frame_intervals))
                tolerance = max(
                    np.finfo(np.float32).eps
                    * max(float(np.max(np.abs(selected_times))), 1.0)
                    * 8.0,
                    abs(trajectory_interval) * 1.0e-6,
                )
                if not np.allclose(
                    frame_intervals,
                    trajectory_interval,
                    rtol=1.0e-6,
                    atol=tolerance,
                ):
                    raise ValueError(
                        "A fixed analysis interval cannot be represented by one "
                        "frame stride because trajectory timestamps are nonuniform."
                    )
                interval_frame = max(
                    1,
                    int(np.floor(interval_ps / trajectory_interval + 0.5)),
                )
            interval_time = interval_ps * picosecond
        else:
            interval_time = "every saved frame"

        actual_start = frame_times_ps[start_frame] * picosecond
        actual_end = frame_times_ps[end_frame] * picosecond
        print(
            f"## Selected actual trajectory time {actual_start} to {actual_end} "
            f"with interval {interval_time}."
        )
        print(
            f"## Will use frame {start_frame} to {end_frame} with interval "
            f"of {interval_frame}."
        )

        return start_frame, end_frame, interval_frame

    def getSelection(self, selection_string, help="analyzing group"):
        if selection_string.isdigit():
            indices = self.index.phraseSelection("group " + selection_string)
            selection_name = self.index.index_groups[int(selection_string.strip())].name
        else:
            indices = self.index.phraseSelection(selection_string)
            selection_name = selection_string.replace(" ", "")

        selection = self.Universe.atoms[indices]
        print(f"## Selected {len(selection)} atoms for {help}.\n")

        return selection, selection_name

    def getSelection_interactive(self, help="analyzing group"):
        print(f"> Please enter one selection for {help}.")
        selection_string = input("> Selection: ").strip()
        return self.getSelection(selection_string)

    def getSelection_interactive_multiple(self, help="analyzing group"):
        selections = []
        selection_names = []
        print(
            f"> Please enter selections for {help}. Type 'q' or press Enter to finish."
        )

        while True:
            selection_string = input("> Selection: ").strip()
            if selection_string.lower() == "q" or selection_string == "":
                break
            try:
                selection, name = self.getSelection(selection_string)
                selections.append(selection)
                selection_names.append(name)
            except Exception as e:
                print(f"Error processing selection: {e}")

        print(f"## Get {len(selections)} selections for {help}.")

        return selections, selection_names
