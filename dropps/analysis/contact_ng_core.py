"""Memory-bounded accumulation helpers for :mod:`dropps.analysis.contact_ng`.

This module deliberately depends only on NumPy.  Keeping the aggregation logic
separate from trajectory I/O makes it possible to check it against the legacy
implementation without loading a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INTER_RS = "inter_rs"
INTER_RR = "inter_rr"
INTER_SS = "inter_ss"
INTRA_R = "intra_r"
INTRA_S = "intra_s"

CONTACT_CATEGORIES = (INTER_RS, INTER_RR, INTER_SS, INTRA_R, INTRA_S)
SELECTION_CATEGORIES = frozenset((INTER_RS, INTER_SS, INTRA_S))


@dataclass(frozen=True)
class ContactLayout:
    """Chain and in-chain position lookups in neighbor-search index space."""

    ref_chain: np.ndarray
    ref_pos: np.ndarray
    sel_chain: np.ndarray | None
    sel_pos: np.ndarray | None
    n_ref: int
    n_sel: int
    length_ref: int
    length_sel: int

    @property
    def n_atoms(self):
        return self.ref_chain.size


def _fixed_chain_length(chains, label):
    if not chains:
        return 0

    chain_length = len(chains[0])
    if any(len(chain) != chain_length for chain in chains):
        raise ValueError(f"{label} chains are not fixed-length within type")
    return chain_length


def _build_membership(search_atom_indices, chains, label):
    """Map global atom indices in ``chains`` to the sorted search atom array."""

    chain_lookup = np.full(search_atom_indices.size, -1, dtype=np.int32)
    position_lookup = np.full(search_atom_indices.size, -1, dtype=np.int32)

    for chain_id, chain in enumerate(chains):
        atom_indices = np.asarray(chain, dtype=np.int64)
        if atom_indices.ndim != 1:
            raise ValueError(f"{label} chain indices must be one-dimensional")
        if atom_indices.size == 0:
            continue

        local_indices = np.searchsorted(search_atom_indices, atom_indices)
        valid = local_indices < search_atom_indices.size
        if not np.all(valid):
            missing = atom_indices[~valid][0]
            raise ValueError(f"Atom index {missing} from {label} chains is not in the search group")

        found = search_atom_indices[local_indices]
        if not np.array_equal(found, atom_indices):
            missing = atom_indices[found != atom_indices][0]
            raise ValueError(f"Atom index {missing} from {label} chains is not in the search group")

        # This intentionally mirrors the old lookup behavior: if malformed
        # input assigns an atom more than once, the last chain wins.
        chain_lookup[local_indices] = chain_id
        position_lookup[local_indices] = np.arange(atom_indices.size, dtype=np.int32)

    return chain_lookup, position_lookup


def build_contact_layout(search_atom_indices, reference_chains, selection_chains=None):
    """Build compact O(N) lookups once, instead of rebuilding them each frame."""

    search_atom_indices = np.asarray(search_atom_indices, dtype=np.int64)
    if search_atom_indices.ndim != 1:
        raise ValueError("search_atom_indices must be one-dimensional")
    if search_atom_indices.size and np.any(search_atom_indices[1:] <= search_atom_indices[:-1]):
        raise ValueError("search_atom_indices must be sorted and unique")

    length_ref = _fixed_chain_length(reference_chains, "Reference")
    if not reference_chains or length_ref == 0:
        raise ValueError("At least one non-empty reference chain is required")

    ref_chain, ref_pos = _build_membership(
        search_atom_indices, reference_chains, "reference"
    )

    if selection_chains is None:
        sel_chain = None
        sel_pos = None
        n_sel = 0
        length_sel = 0
    else:
        length_sel = _fixed_chain_length(selection_chains, "Selection")
        if not selection_chains or length_sel == 0:
            raise ValueError("Selection chains must be non-empty when provided")
        sel_chain, sel_pos = _build_membership(
            search_atom_indices, selection_chains, "selection"
        )
        n_sel = len(selection_chains)

    return ContactLayout(
        ref_chain=ref_chain,
        ref_pos=ref_pos,
        sel_chain=sel_chain,
        sel_pos=sel_pos,
        n_ref=len(reference_chains),
        n_sel=n_sel,
        length_ref=length_ref,
        length_sel=length_sel,
    )


class StreamingContactAccumulator:
    """Accumulate contacts directly into final 2-D maps and frame scalars.

    Only maps explicitly listed in ``map_categories`` are allocated.  Categories
    requested only as time series use scalar counters and therefore consume no
    O(L^2) memory.
    """

    def __init__(
        self,
        layout,
        map_categories=(),
        time_categories=(),
        intra_average=False,
        remove_diagonal=2,
    ):
        self.layout = layout
        self.map_categories = frozenset(map_categories)
        self.time_categories = frozenset(time_categories)
        self.categories = self.map_categories | self.time_categories
        self.intra_average = intra_average
        self.remove_diagonal = remove_diagonal
        self.n_frames = 0
        self._frame_active = False
        self._frame_counts = {}

        unknown = self.categories.difference(CONTACT_CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown contact categories: {sorted(unknown)}")
        if layout.sel_chain is None and self.categories.intersection(SELECTION_CATEGORIES):
            raise ValueError("Selection contact output requested without selection chains")

        shapes = {
            INTER_RS: (layout.length_ref, layout.length_sel),
            INTER_RR: (layout.length_ref, layout.length_ref),
            INTER_SS: (layout.length_sel, layout.length_sel),
            INTRA_R: (layout.length_ref, layout.length_ref),
            INTRA_S: (layout.length_sel, layout.length_sel),
        }
        # float64 matches the old global map accumulators and supports the
        # fractional contribution used by --intra-average.
        self.maps = {
            category: np.zeros(shapes[category], dtype=np.float64)
            for category in self.map_categories
        }

    def begin_frame(self):
        if self._frame_active:
            raise RuntimeError("A contact frame is already active")
        self._frame_counts = {
            category: 0.0 for category in self.time_categories
        }
        self._frame_active = True

    def _record_symmetric(self, category, mask, pos_a, pos_b, weight=1.0):
        if category in self.time_categories:
            self._frame_counts[category] += float(np.count_nonzero(mask)) * weight

        matrix = self.maps.get(category)
        if matrix is None or not np.any(mask):
            return

        row = pos_a[mask]
        column = pos_b[mask]
        np.add.at(matrix, (row, column), weight)
        np.add.at(matrix, (column, row), weight)

    def _record_directed(self, category, mask, row_pos, column_pos):
        if category in self.time_categories:
            self._frame_counts[category] += float(np.count_nonzero(mask))

        matrix = self.maps.get(category)
        if matrix is None or not np.any(mask):
            return

        np.add.at(matrix, (row_pos[mask], column_pos[mask]), 1.0)

    def accumulate(self, atom_a, atom_b):
        """Consume unique unordered neighbor pairs in search-local indexing."""

        if not self._frame_active:
            raise RuntimeError("begin_frame() must be called before accumulate()")

        atom_a = np.asarray(atom_a)
        atom_b = np.asarray(atom_b)
        if atom_a.ndim != 1 or atom_b.ndim != 1 or atom_a.shape != atom_b.shape:
            raise ValueError("atom_a and atom_b must be same-length one-dimensional arrays")
        if atom_a.size == 0:
            return
        if (
            np.min(atom_a) < 0
            or np.min(atom_b) < 0
            or np.max(atom_a) >= self.layout.n_atoms
            or np.max(atom_b) >= self.layout.n_atoms
        ):
            raise IndexError("Neighbor pair index is outside the search atom group")

        nonself = atom_a != atom_b
        if not np.all(nonself):
            atom_a = atom_a[nonself]
            atom_b = atom_b[nonself]
            if atom_a.size == 0:
                return

        needs_reference = bool(
            self.categories.intersection((INTER_RS, INTER_RR, INTRA_R))
        )
        needs_selection = bool(
            self.categories.intersection((INTER_RS, INTER_SS, INTRA_S))
        )

        if needs_reference:
            ref_chain_a = self.layout.ref_chain[atom_a]
            ref_chain_b = self.layout.ref_chain[atom_b]
            ref_pos_a = self.layout.ref_pos[atom_a]
            ref_pos_b = self.layout.ref_pos[atom_b]

            if INTRA_R in self.categories:
                mask = (
                    (ref_chain_a >= 0)
                    & (ref_chain_b >= 0)
                    & (ref_chain_a == ref_chain_b)
                )
                if self.remove_diagonal >= 0:
                    mask &= np.abs(ref_pos_a - ref_pos_b) > self.remove_diagonal
                weight = 1.0 / self.layout.n_ref if self.intra_average else 1.0
                self._record_symmetric(INTRA_R, mask, ref_pos_a, ref_pos_b, weight)

            if INTER_RR in self.categories:
                mask = (
                    (ref_chain_a >= 0)
                    & (ref_chain_b >= 0)
                    & (ref_chain_a != ref_chain_b)
                )
                self._record_symmetric(INTER_RR, mask, ref_pos_a, ref_pos_b)

        if needs_selection:
            sel_chain_a = self.layout.sel_chain[atom_a]
            sel_chain_b = self.layout.sel_chain[atom_b]
            sel_pos_a = self.layout.sel_pos[atom_a]
            sel_pos_b = self.layout.sel_pos[atom_b]

            if INTRA_S in self.categories:
                mask = (
                    (sel_chain_a >= 0)
                    & (sel_chain_b >= 0)
                    & (sel_chain_a == sel_chain_b)
                )
                if self.remove_diagonal >= 0:
                    mask &= np.abs(sel_pos_a - sel_pos_b) > self.remove_diagonal
                weight = 1.0 / self.layout.n_sel if self.intra_average else 1.0
                self._record_symmetric(INTRA_S, mask, sel_pos_a, sel_pos_b, weight)

            if INTER_SS in self.categories:
                mask = (
                    (sel_chain_a >= 0)
                    & (sel_chain_b >= 0)
                    & (sel_chain_a != sel_chain_b)
                )
                self._record_symmetric(INTER_SS, mask, sel_pos_a, sel_pos_b)

        if INTER_RS in self.categories:
            # Preserve both cases from contact.py.  This matters when reference
            # and selection overlap or are identical.
            mask = (ref_chain_a >= 0) & (sel_chain_b >= 0)
            self._record_directed(INTER_RS, mask, ref_pos_a, sel_pos_b)

            mask = (sel_chain_a >= 0) & (ref_chain_b >= 0)
            self._record_directed(INTER_RS, mask, ref_pos_b, sel_pos_a)

    def finish_frame(self):
        if not self._frame_active:
            raise RuntimeError("No active contact frame")
        self._frame_active = False
        self.n_frames += 1
        return self._frame_counts.copy()

    def averaged_maps(self):
        """Normalize accumulated maps in place and return them."""

        if self._frame_active:
            raise RuntimeError("Cannot normalize maps while a frame is active")
        if self.n_frames == 0:
            raise RuntimeError("No trajectory frames were processed")

        for matrix in self.maps.values():
            matrix /= self.n_frames
        return self.maps
