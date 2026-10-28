"""Lightweight construction of topology-defined non-bonded exclusions."""

from __future__ import annotations

from collections import defaultdict


def build_exclusion_pairs(topology_list, addition_per_molecule=None):
    """Return sorted unique global 1-2/1-3 exclusion pairs.

    DROPPS currently supports ITP ``nrexcl`` values 1 and 2.  ``topology_list``
    contains expanded molecule instances; ``addition_per_molecule`` gives the
    global atom offset for each instance.  When offsets are omitted they are
    derived from the instance atom counts.
    """

    topologies = list(topology_list)
    if addition_per_molecule is None:
        offsets = []
        offset = 0
        for topology in topologies:
            offsets.append(offset)
            offset += len(topology.atoms)
    else:
        offsets = [int(value) for value in addition_per_molecule]
        if len(offsets) != len(topologies):
            raise ValueError(
                "Topology instances and molecule atom offsets must have the "
                "same length."
            )

    exclusion_pairs = set()
    for topology, addition in zip(topologies, offsets):
        nrexcl = int(topology.nrexcl)
        if nrexcl not in (1, 2):
            raise ValueError(
                "Only 1-2 and 1-3 exclusions are supported; "
                f"molecule {topology.molecule_name} has nrexcl={nrexcl}."
            )

        neighbors = defaultdict(set)
        for bond in topology.bonds or []:
            atom_1 = int(bond.a1) + addition
            atom_2 = int(bond.a2) + addition
            exclusion_pairs.add(tuple(sorted((atom_1, atom_2))))
            neighbors[atom_1].add(atom_2)
            neighbors[atom_2].add(atom_1)

        if nrexcl == 2:
            for atom_1, direct_neighbors in neighbors.items():
                for middle_atom in direct_neighbors:
                    for atom_3 in neighbors[middle_atom]:
                        if atom_1 != atom_3:
                            exclusion_pairs.add(tuple(sorted((atom_1, atom_3))))

    return [list(pair) for pair in sorted(exclusion_pairs)]
