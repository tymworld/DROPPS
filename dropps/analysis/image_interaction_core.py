"""Core algorithms for periodic-image self-interaction analysis.

The molecular contact graph is a *voltage graph*: every contact edge stores
the integer lattice translation of the contacted molecule image.  A molecule
is connected to one of its own periodic images exactly when its connected
component contains a closed walk whose accumulated lattice translation is
non-zero.  Weighted union-find detects that condition without limiting the
number of intermediate molecules in the walk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnwrapPlan:
    """Static topology information used to unwrap selected molecules."""

    atom_indices: np.ndarray
    roots: np.ndarray
    levels: tuple[tuple[np.ndarray, np.ndarray], ...]
    selected_local_indices: np.ndarray
    selected_molecule_ids: np.ndarray


@dataclass(frozen=True)
class ImageInteractionResult:
    """Periodic-image connectivity result for one trajectory frame."""

    affected: np.ndarray
    direct: np.ndarray
    component_labels: np.ndarray
    wrapped_components: int
    winding_conflicts: int


def build_unwrap_plan(molecule_atom_indices, selected_atom_indices, bonds):
    """Build a breadth-first molecular unwrapping plan from topology bonds.

    ``molecule_atom_indices`` must contain the complete atom list for every
    selected molecule.  ``selected_atom_indices`` may be a subset of those
    atoms; only that subset is later used to define contacts.
    """

    molecule_groups = [
        np.asarray(indices, dtype=np.int64) for indices in molecule_atom_indices
    ]
    if not molecule_groups or any(
        group.ndim != 1 or group.size == 0 for group in molecule_groups
    ):
        raise ValueError(
            "Molecule atom groups must be non-empty one-dimensional arrays."
        )

    selected = np.asarray(selected_atom_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("The selected atom indices must be a non-empty 1D array.")
    if np.unique(selected).size != selected.size:
        raise ValueError("The selected atom indices contain duplicates.")

    atom_indices = np.concatenate(molecule_groups)
    if np.unique(atom_indices).size != atom_indices.size:
        raise ValueError("Molecule atom groups overlap.")

    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(atom_indices)
    }
    owner = np.empty(atom_indices.size, dtype=np.int64)
    offset = 0
    for molecule_id, group in enumerate(molecule_groups):
        owner[offset : offset + group.size] = molecule_id
        offset += group.size

    adjacency = [[] for _ in range(atom_indices.size)]
    bond_array = np.asarray(bonds, dtype=np.int64)
    if bond_array.size == 0:
        bond_array = np.empty((0, 2), dtype=np.int64)
    if bond_array.ndim != 2 or bond_array.shape[1] != 2:
        raise ValueError("Bonds must have shape (n_bonds, 2).")

    for global_a, global_b in bond_array:
        local_a = global_to_local.get(int(global_a))
        local_b = global_to_local.get(int(global_b))
        if local_a is None or local_b is None or owner[local_a] != owner[local_b]:
            continue
        adjacency[local_a].append(local_b)
        adjacency[local_b].append(local_a)

    roots = []
    edges_by_depth: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for molecule_id, group in enumerate(molecule_groups):
        molecule_locals = np.arange(offset, offset + group.size, dtype=np.int64)
        root = int(molecule_locals[0])
        roots.append(root)
        visited = {root}
        frontier = [root]
        depth = 0
        while frontier:
            next_frontier = []
            for parent in frontier:
                for child in adjacency[parent]:
                    if child in visited:
                        continue
                    visited.add(child)
                    next_frontier.append(child)
                    edges_by_depth.setdefault(depth, []).append((parent, child))
            frontier = next_frontier
            depth += 1

        if len(visited) != group.size:
            disconnected = group.size - len(visited)
            raise ValueError(
                f"Molecule {molecule_id} has {disconnected} atom(s) disconnected "
                "from its first atom; bonded connectivity is required for "
                "unambiguous periodic unwrapping."
            )
        offset += group.size

    selected_local_indices = np.empty(selected.size, dtype=np.int64)
    selected_molecule_ids = np.empty(selected.size, dtype=np.int64)
    for selected_id, global_index in enumerate(selected):
        local_index = global_to_local.get(int(global_index))
        if local_index is None:
            raise ValueError(
                f"Selected atom {global_index} is absent from the selected molecules."
            )
        selected_local_indices[selected_id] = local_index
        selected_molecule_ids[selected_id] = owner[local_index]

    represented = np.unique(selected_molecule_ids)
    if represented.size != len(molecule_groups):
        raise ValueError(
            "Every molecule must contain at least one selected contact atom."
        )

    levels = []
    for depth in sorted(edges_by_depth):
        edge_array = np.asarray(edges_by_depth[depth], dtype=np.int64)
        levels.append((edge_array[:, 0], edge_array[:, 1]))

    return UnwrapPlan(
        atom_indices=atom_indices,
        roots=np.asarray(roots, dtype=np.int64),
        levels=tuple(levels),
        selected_local_indices=selected_local_indices,
        selected_molecule_ids=selected_molecule_ids,
    )


def canonicalize_image_contacts(
    molecule_a,
    molecule_b,
    image_shifts,
    minimum_contacts=1,
):
    """Aggregate atom contacts into unique undirected molecule-image edges.

    An edge row is ``(molecule_a, molecule_b, sx, sy, sz)`` and means that the
    base image of ``molecule_a`` contacts image ``(sx, sy, sz)`` of
    ``molecule_b``.  Reverse orientations, including ``+shift``/``-shift``
    self-image contacts, are canonicalized before applying the contact count.
    Ordinary zero-shift intramolecular contacts are discarded.
    """

    molecule_a = np.asarray(molecule_a, dtype=np.int64)
    molecule_b = np.asarray(molecule_b, dtype=np.int64)
    shifts = np.asarray(image_shifts)
    if molecule_a.ndim != 1 or molecule_b.shape != molecule_a.shape:
        raise ValueError("Molecule index arrays must be matching 1D arrays.")
    if shifts.shape != (molecule_a.size, 3):
        raise ValueError("Image shifts must have shape (n_contacts, 3).")
    if minimum_contacts <= 0:
        raise ValueError("minimum_contacts must be greater than zero.")
    if shifts.size and not np.all(np.isfinite(shifts)):
        raise ValueError("Image shifts contain non-finite values.")
    rounded_shifts = np.rint(shifts).astype(np.int64)
    if shifts.size and not np.allclose(shifts, rounded_shifts, atol=1e-7, rtol=0.0):
        raise ValueError("Image shifts must be integer lattice translations.")
    shifts = rounded_shifts

    keep = (molecule_a != molecule_b) | np.any(shifts != 0, axis=1)
    molecule_a = molecule_a[keep].copy()
    molecule_b = molecule_b[keep].copy()
    shifts = shifts[keep].copy()
    if molecule_a.size == 0:
        return np.empty((0, 5), dtype=np.int64), np.empty(0, dtype=np.int64)

    swap = molecule_a > molecule_b
    if np.any(swap):
        old_a = molecule_a[swap].copy()
        molecule_a[swap] = molecule_b[swap]
        molecule_b[swap] = old_a
        shifts[swap] *= -1

    self_edges = molecule_a == molecule_b
    if np.any(self_edges):
        self_rows = np.flatnonzero(self_edges)
        self_shifts = shifts[self_edges]
        first_nonzero = np.argmax(self_shifts != 0, axis=1)
        reverse = self_shifts[np.arange(self_shifts.shape[0]), first_nonzero] < 0
        shifts[self_rows[reverse]] *= -1

    rows = np.column_stack((molecule_a, molecule_b, shifts))
    unique_rows, counts = np.unique(rows, axis=0, return_counts=True)
    accepted = counts >= int(minimum_contacts)
    return unique_rows[accepted], counts[accepted]


class _LatticeUnionFind:
    """Union-find with integer vector potentials between graph vertices."""

    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)
        # potential[x] = lattice_position[x] - lattice_position[parent[x]]
        self.potential = np.zeros((size, 3), dtype=np.int64)
        self.wrapped = np.zeros(size, dtype=bool)

    def find(self, node):
        parent = int(self.parent[node])
        if parent == node:
            return node, self.potential[node].copy()
        root, parent_potential = self.find(parent)
        self.potential[node] += parent_potential
        self.parent[node] = root
        return root, self.potential[node].copy()

    def add_constraint(self, node_a, node_b, shift):
        """Add ``position[b] = position[a] + shift``.

        Returns ``True`` only when the edge proves a non-zero winding cycle.
        """

        root_a, potential_a = self.find(node_a)
        root_b, potential_b = self.find(node_b)
        shift = np.asarray(shift, dtype=np.int64)

        if root_a == root_b:
            winding = potential_a + shift - potential_b
            if np.any(winding != 0):
                self.wrapped[root_a] = True
                return True
            return False

        if self.rank[root_a] < self.rank[root_b]:
            # position[root_a] - position[root_b]
            self.parent[root_a] = root_b
            self.potential[root_a] = potential_b - potential_a - shift
            self.wrapped[root_b] |= self.wrapped[root_a]
        else:
            # position[root_b] - position[root_a]
            self.parent[root_b] = root_a
            self.potential[root_b] = potential_a + shift - potential_b
            self.wrapped[root_a] |= self.wrapped[root_b]
            if self.rank[root_a] == self.rank[root_b]:
                self.rank[root_a] += 1
        return False


def analyze_image_edges(n_molecules, edges):
    """Find molecules connected directly or indirectly to their own images."""

    if n_molecules <= 0:
        raise ValueError("n_molecules must be greater than zero.")
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        edge_array = np.empty((0, 5), dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[1] != 5:
        raise ValueError("Edges must have shape (n_edges, 5).")
    if edge_array.size and (
        np.min(edge_array[:, :2]) < 0 or np.max(edge_array[:, :2]) >= n_molecules
    ):
        raise IndexError("An edge contains a molecule index outside the graph.")

    union_find = _LatticeUnionFind(n_molecules)
    direct = np.zeros(n_molecules, dtype=bool)
    winding_conflicts = 0
    for node_a, node_b, shift_x, shift_y, shift_z in edge_array:
        shift = np.array((shift_x, shift_y, shift_z), dtype=np.int64)
        if node_a == node_b and np.any(shift != 0):
            direct[node_a] = True
        if union_find.add_constraint(int(node_a), int(node_b), shift):
            winding_conflicts += 1

    roots = np.empty(n_molecules, dtype=np.int64)
    for molecule_id in range(n_molecules):
        roots[molecule_id] = union_find.find(molecule_id)[0]

    unique_roots = np.unique(roots)
    root_to_label = {int(root): label for label, root in enumerate(unique_roots)}
    component_labels = np.asarray(
        [root_to_label[int(root)] for root in roots], dtype=np.int64
    )
    affected = np.asarray([union_find.wrapped[int(root)] for root in roots], dtype=bool)
    wrapped_components = int(
        sum(bool(union_find.wrapped[int(root)]) for root in unique_roots)
    )
    return ImageInteractionResult(
        affected=affected,
        direct=direct,
        component_labels=component_labels,
        wrapped_components=wrapped_components,
        winding_conflicts=winding_conflicts,
    )
