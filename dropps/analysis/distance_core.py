"""Numerical planning and streaming kernels for unified distance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors


ANGSTROM_PER_NM = 10.0


@dataclass(frozen=True)
class Entity:
    """One ordered atom entity emitted by endpoint grouping."""

    atom_indices: np.ndarray
    atom_chain_ids: np.ndarray
    label: str

    @property
    def chain_ids(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(int(value) for value in self.atom_chain_ids))


@dataclass(frozen=True)
class EndpointPlan:
    """Static selection, grouping, and point-construction definition."""

    label: str
    entities: tuple[Entity, ...]
    position_type: str


@dataclass(frozen=True)
class DistanceJobPlan:
    """Static relationship between two endpoint plans."""

    label: str
    reference: EndpointPlan
    selection: EndpointPlan
    scope: str
    mode: str
    exclusion: str
    unique_entity_pairs: bool
    entity_pair_count: int
    channel_count: int
    corresponding_pairs: np.ndarray | None = None


@dataclass(frozen=True)
class ChannelDescriptor:
    """Stable identity for one scalar distance channel."""

    index: int
    label: str
    reference_entity: str
    selection_entity: str


@dataclass(frozen=True)
class DistanceBlock:
    """One bounded, contiguous slice of frame results."""

    start: int
    values: np.ndarray
    vectors: np.ndarray


def validate_dimensions(dimensions):
    """Validate an MDAnalysis six-value periodic box."""

    if dimensions is None:
        raise ValueError("Trajectory frame does not contain a periodic box.")
    box = np.asarray(dimensions, dtype=np.float64)
    if box.shape != (6,) or not np.all(np.isfinite(box)):
        raise ValueError("Periodic box dimensions must be a finite six-vector.")
    cell = np.asarray(triclinic_vectors(box, dtype=np.float64))
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 0.0:
        raise ValueError("Periodic box must have positive finite volume.")
    return box


def dimensions_angstrom_to_nm(dimensions):
    box = validate_dimensions(dimensions).copy()
    box[:3] /= ANGSTROM_PER_NM
    return box


def _minimum_image(vectors, dimensions, pbc):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1:] != (3,) or not np.all(np.isfinite(vectors)):
        raise ValueError("Distance vectors must be finite Cartesian 3-vectors.")
    if not pbc:
        return vectors
    return minimize_vectors(vectors, validate_dimensions(dimensions))


def _ordered_groups(indices, keys):
    groups: dict[object, list[int]] = {}
    for index, key in zip(indices, keys):
        groups.setdefault(key, []).append(int(index))
    return list(groups.items())


def build_endpoint_plan(
    selection_indices,
    label,
    grouping,
    position_type,
    atom_chain_ids,
    atom_residue_ids,
):
    """Resolve one ordered atom selection into static entities."""

    indices = np.asarray(selection_indices, dtype=np.int64)
    chain_ids = np.asarray(atom_chain_ids, dtype=np.int64)
    residue_ids = np.asarray(atom_residue_ids, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"Distance selection '{label}' is empty or invalid.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Distance selection '{label}' contains duplicate atoms.")
    if chain_ids.ndim != 1 or residue_ids.shape != chain_ids.shape:
        raise ValueError("Atom chain and residue metadata must be matching vectors.")
    if np.min(indices) < 0 or np.max(indices) >= chain_ids.size:
        raise IndexError(f"Distance selection '{label}' contains an invalid atom.")
    if grouping not in {"all", "chain", "residue", "atom"}:
        raise ValueError(f"Unknown distance grouping: {grouping}")
    if position_type not in {"atom", "cog", "com"}:
        raise ValueError(f"Unknown distance position type: {position_type}")

    if grouping == "all":
        grouped = [("all", [int(index) for index in indices])]
    elif grouping == "chain":
        grouped = _ordered_groups(indices, chain_ids[indices])
    elif grouping == "residue":
        keys = [
            (int(chain_ids[index]), int(residue_ids[index]))
            for index in indices
        ]
        grouped = _ordered_groups(indices, keys)
    else:
        grouped = [(int(index), [int(index)]) for index in indices]

    entities = []
    for key, group_indices in grouped:
        atoms = np.asarray(group_indices, dtype=np.int64)
        if grouping == "all":
            entity_label = str(label)
        elif grouping == "chain":
            entity_label = f"{label}[chain={int(key)}]"
        elif grouping == "residue":
            chain_id, residue_id = key
            entity_label = (
                f"{label}[chain={int(chain_id)},resid={int(residue_id)}]"
            )
        else:
            entity_label = f"{label}[atom={int(key)}]"
        entities.append(
            Entity(
                atom_indices=atoms,
                atom_chain_ids=chain_ids[atoms].copy(),
                label=entity_label,
            )
        )
    return EndpointPlan(str(label), tuple(entities), position_type)


def _point_count(endpoint, entity_index):
    entity = endpoint.entities[int(entity_index)]
    return int(entity.atom_indices.size) if endpoint.position_type == "atom" else 1


def _point_atom_set(endpoint, entity_index, point_index):
    entity = endpoint.entities[int(entity_index)]
    if endpoint.position_type == "atom":
        return frozenset((int(entity.atom_indices[int(point_index)]),))
    return frozenset(int(index) for index in entity.atom_indices)


def _point_chain_set(endpoint, entity_index, point_index):
    entity = endpoint.entities[int(entity_index)]
    if endpoint.position_type == "atom":
        return frozenset((int(entity.atom_chain_ids[int(point_index)]),))
    return frozenset(entity.chain_ids)


def _point_allowed(job, ref_entity, sel_entity, ref_point, sel_point):
    if job.exclusion == "none":
        return True
    if job.exclusion == "same-atom":
        reference = _point_atom_set(job.reference, ref_entity, ref_point)
        selection = _point_atom_set(job.selection, sel_entity, sel_point)
    elif job.exclusion == "same-molecule":
        reference = _point_chain_set(job.reference, ref_entity, ref_point)
        selection = _point_chain_set(job.selection, sel_entity, sel_point)
    else:
        raise ValueError(f"Unknown distance exclusion: {job.exclusion}")
    return reference.isdisjoint(selection)


def _entity_chain(endpoint, entity_index):
    chain_ids = endpoint.entities[int(entity_index)].chain_ids
    if len(chain_ids) != 1:
        raise ValueError(
            "Intra-chain and inter-chain scopes require every entity to belong "
            "to exactly one topology chain."
        )
    return chain_ids[0]


def _corresponding_pairs(reference, selection):
    ref_count = len(reference.entities)
    sel_count = len(selection.entities)
    if ref_count == sel_count:
        return np.column_stack(
            (
                np.arange(ref_count, dtype=np.int64),
                np.arange(sel_count, dtype=np.int64),
            )
        )

    if (
        ref_count == 1
        and _point_count(reference, 0) == 1
        and all(_point_count(selection, index) == 1 for index in range(sel_count))
    ):
        return np.column_stack(
            (
                np.zeros(sel_count, dtype=np.int64),
                np.arange(sel_count, dtype=np.int64),
            )
        )
    if (
        sel_count == 1
        and _point_count(selection, 0) == 1
        and all(_point_count(reference, index) == 1 for index in range(ref_count))
    ):
        return np.column_stack(
            (
                np.arange(ref_count, dtype=np.int64),
                np.zeros(ref_count, dtype=np.int64),
            )
        )
    raise ValueError(
        "Corresponding distance scope requires matching entity counts, except "
        "that one single-position entity may be broadcast."
    )


def _iter_raw_entity_pair_blocks(job, block_size):
    if block_size <= 0:
        raise ValueError("Distance block size must be greater than zero.")
    if job.corresponding_pairs is not None:
        for start in range(0, len(job.corresponding_pairs), block_size):
            yield job.corresponding_pairs[start : start + block_size]
        return

    ref_count = len(job.reference.entities)
    sel_count = len(job.selection.entities)
    ref_chains = np.asarray(
        [_entity_chain(job.reference, index) for index in range(ref_count)],
        dtype=np.int64,
    ) if job.scope in {"intra-chain", "inter-chain"} else None
    sel_chains = np.asarray(
        [_entity_chain(job.selection, index) for index in range(sel_count)],
        dtype=np.int64,
    ) if job.scope in {"intra-chain", "inter-chain"} else None

    if job.scope == "intra-chain":
        reference_by_chain: dict[int, list[int]] = {}
        for ref_index, chain_id in enumerate(ref_chains):
            reference_by_chain.setdefault(int(chain_id), []).append(ref_index)
        pending = []
        for sel_index, chain_id in enumerate(sel_chains):
            pending.extend(
                (ref_index, sel_index)
                for ref_index in reference_by_chain.get(int(chain_id), ())
            )
            while len(pending) >= block_size:
                yield np.asarray(pending[:block_size], dtype=np.int64)
                pending = pending[block_size:]
        if pending:
            yield np.asarray(pending, dtype=np.int64)
        return

    total = ref_count * sel_count
    for start in range(0, total, block_size):
        linear = np.arange(start, min(start + block_size, total), dtype=np.int64)
        ref_indices = linear % ref_count
        sel_indices = linear // ref_count
        keep = np.ones(linear.size, dtype=bool)
        if job.scope == "inter-chain":
            keep &= ref_chains[ref_indices] != sel_chains[sel_indices]
        if job.unique_entity_pairs:
            keep &= ref_indices < sel_indices
        if np.any(keep):
            yield np.column_stack((ref_indices[keep], sel_indices[keep]))


def _pair_channel_count(job, ref_entity, sel_entity):
    ref_points = _point_count(job.reference, ref_entity)
    sel_points = _point_count(job.selection, sel_entity)
    if job.mode == "pair":
        if ref_points != sel_points:
            raise ValueError(
                f"Cannot pair {job.reference.entities[ref_entity].label} "
                f"({ref_points} positions) with "
                f"{job.selection.entities[sel_entity].label} "
                f"({sel_points} positions)."
            )
        return sum(
            _point_allowed(job, ref_entity, sel_entity, index, index)
            for index in range(ref_points)
        )

    for sel_point in range(sel_points):
        for ref_point in range(ref_points):
            if _point_allowed(
                job,
                ref_entity,
                sel_entity,
                ref_point,
                sel_point,
            ):
                return 1
    return 0


def _single_point_job(job):
    return all(
        _point_count(job.reference, index) == 1
        for index in range(len(job.reference.entities))
    ) and all(
        _point_count(job.selection, index) == 1
        for index in range(len(job.selection.entities))
    )


def build_distance_job(
    label,
    reference,
    selection,
    scope,
    mode,
    exclusion="none",
    unique_entity_pairs=False,
    block_size=65536,
):
    """Create and validate a distance job without materializing frame data."""

    if scope not in {"corresponding", "intra-chain", "inter-chain", "all"}:
        raise ValueError(f"Unknown distance scope: {scope}")
    if mode not in {"pair", "min", "max"}:
        raise ValueError(f"Unknown distance mode: {mode}")
    if exclusion not in {"none", "same-atom", "same-molecule"}:
        raise ValueError(f"Unknown distance exclusion: {exclusion}")

    corresponding = None
    if scope == "corresponding":
        if unique_entity_pairs:
            raise ValueError(
                "--unique-entity-pairs requires all or inter-chain scope."
            )
        corresponding = _corresponding_pairs(reference, selection)
    elif scope in {"intra-chain", "inter-chain"}:
        for endpoint in (reference, selection):
            for entity_index in range(len(endpoint.entities)):
                _entity_chain(endpoint, entity_index)

    if unique_entity_pairs:
        if scope not in {"all", "inter-chain"}:
            raise ValueError(
                "Unique entity pairs require all or inter-chain scope."
            )
        if (
            reference.label != selection.label
            or reference.position_type != selection.position_type
            or len(reference.entities) != len(selection.entities)
            or any(
                not np.array_equal(ref.atom_indices, sel.atom_indices)
                for ref, sel in zip(reference.entities, selection.entities)
            )
        ):
            raise ValueError(
                "Unique entity pairs require identical endpoint definitions."
            )

    provisional = DistanceJobPlan(
        label=str(label),
        reference=reference,
        selection=selection,
        scope=scope,
        mode=mode,
        exclusion=exclusion,
        unique_entity_pairs=bool(unique_entity_pairs),
        entity_pair_count=0,
        channel_count=0,
        corresponding_pairs=corresponding,
    )
    entity_pair_count = 0
    channel_count = 0
    direct_single_points = exclusion == "none" and _single_point_job(provisional)
    for pairs in _iter_raw_entity_pair_blocks(provisional, block_size):
        if direct_single_points:
            entity_pair_count += int(pairs.shape[0])
            channel_count += int(pairs.shape[0])
            continue
        for ref_entity, sel_entity in pairs:
            count = _pair_channel_count(
                provisional,
                int(ref_entity),
                int(sel_entity),
            )
            if count:
                entity_pair_count += 1
                channel_count += count
    if channel_count == 0:
        raise ValueError(f"Distance job '{label}' contains no valid channels.")
    return DistanceJobPlan(
        label=provisional.label,
        reference=reference,
        selection=selection,
        scope=scope,
        mode=mode,
        exclusion=exclusion,
        unique_entity_pairs=bool(unique_entity_pairs),
        entity_pair_count=entity_pair_count,
        channel_count=channel_count,
        corresponding_pairs=corresponding,
    )


def iter_active_entity_pair_blocks(job, block_size=4096):
    """Yield stable entity-pair blocks with at least one output channel."""

    if job.exclusion == "none" and _single_point_job(job):
        yield from _iter_raw_entity_pair_blocks(job, block_size)
        return
    pending = []
    for pairs in _iter_raw_entity_pair_blocks(job, max(block_size, 1)):
        for ref_entity, sel_entity in pairs:
            ref_entity = int(ref_entity)
            sel_entity = int(sel_entity)
            if _pair_channel_count(job, ref_entity, sel_entity):
                pending.append((ref_entity, sel_entity))
                if len(pending) == block_size:
                    yield np.asarray(pending, dtype=np.int64)
                    pending = []
    if pending:
        yield np.asarray(pending, dtype=np.int64)


def _point_label(endpoint, entity_index, point_index):
    entity = endpoint.entities[int(entity_index)]
    if endpoint.position_type == "atom":
        return f"{entity.label}:atom={int(entity.atom_indices[int(point_index)])}"
    return f"{entity.label}:{endpoint.position_type}"


def iter_channel_descriptors(job):
    channel_index = 0
    for pairs in iter_active_entity_pair_blocks(job):
        for ref_entity, sel_entity in pairs:
            ref_entity = int(ref_entity)
            sel_entity = int(sel_entity)
            ref_label = job.reference.entities[ref_entity].label
            sel_label = job.selection.entities[sel_entity].label
            if job.mode == "pair":
                count = _point_count(job.reference, ref_entity)
                for point_index in range(count):
                    if not _point_allowed(
                        job,
                        ref_entity,
                        sel_entity,
                        point_index,
                        point_index,
                    ):
                        continue
                    point_ref = _point_label(job.reference, ref_entity, point_index)
                    point_sel = _point_label(job.selection, sel_entity, point_index)
                    label = f"{job.label}: {point_ref} -> {point_sel}"
                    yield ChannelDescriptor(
                        channel_index,
                        label,
                        ref_label,
                        sel_label,
                    )
                    channel_index += 1
            else:
                yield ChannelDescriptor(
                    channel_index,
                    f"{job.label}: {ref_label} -> {sel_label} {job.mode}",
                    ref_label,
                    sel_label,
                )
                channel_index += 1
    if channel_index != job.channel_count:
        raise RuntimeError("Distance channel planning produced an unstable count.")


def project_endpoint(endpoint, positions, masses, dimensions=None, pbc=True):
    """Construct atom, COG, or COM positions for every endpoint entity."""

    coordinates = np.asarray(positions, dtype=np.float64)
    atom_masses = np.asarray(masses, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Positions must have shape (n_atoms, 3).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Positions contain non-finite values.")
    if atom_masses.shape != (coordinates.shape[0],):
        raise ValueError("Atom masses must match the coordinate count.")
    if pbc:
        validate_dimensions(dimensions)

    projected = []
    for entity in endpoint.entities:
        entity_positions = coordinates[entity.atom_indices].copy()
        if endpoint.position_type == "atom":
            projected.append(entity_positions)
            continue

        if pbc and len(entity.chain_ids) > 1:
            first_chain = entity.chain_ids[0]
            first_mask = entity.atom_chain_ids == first_chain
            anchor = entity_positions[first_mask].mean(axis=0)
            for chain_id in entity.chain_ids[1:]:
                mask = entity.atom_chain_ids == chain_id
                center = entity_positions[mask].mean(axis=0)
                displacement = _minimum_image(
                    (center - anchor)[np.newaxis, :],
                    dimensions,
                    True,
                )[0]
                entity_positions[mask] += anchor + displacement - center

        if endpoint.position_type == "cog":
            center = entity_positions.mean(axis=0)
        else:
            weights = atom_masses[entity.atom_indices]
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError(f"Entity '{entity.label}' has invalid atom masses.")
            total = float(weights.sum())
            if total <= 0.0:
                raise ValueError(f"Entity '{entity.label}' has no positive mass.")
            center = np.einsum("ai,a->i", entity_positions, weights) / total
        projected.append(center[np.newaxis, :])
    return tuple(projected)


def _allowed_pair_mask(job, ref_entity, sel_entity, point_count):
    if job.exclusion == "none":
        return np.ones(point_count, dtype=bool)
    return np.asarray(
        [
            _point_allowed(job, ref_entity, sel_entity, index, index)
            for index in range(point_count)
        ],
        dtype=bool,
    )


def _extreme_vector(
    job,
    ref_entity,
    sel_entity,
    reference_points,
    selection_points,
    dimensions,
    pbc,
    candidate_block_size,
):
    ref_count = reference_points.shape[0]
    sel_count = selection_points.shape[0]
    total = ref_count * sel_count
    best_squared = np.inf if job.mode == "min" else -np.inf
    best_vector = None
    for start in range(0, total, candidate_block_size):
        linear = np.arange(
            start,
            min(start + candidate_block_size, total),
            dtype=np.int64,
        )
        ref_indices = linear % ref_count
        sel_indices = linear // ref_count
        if job.exclusion != "none":
            keep = np.asarray(
                [
                    _point_allowed(
                        job,
                        ref_entity,
                        sel_entity,
                        int(ref_point),
                        int(sel_point),
                    )
                    for ref_point, sel_point in zip(ref_indices, sel_indices)
                ],
                dtype=bool,
            )
            ref_indices = ref_indices[keep]
            sel_indices = sel_indices[keep]
        if ref_indices.size == 0:
            continue
        vectors = selection_points[sel_indices] - reference_points[ref_indices]
        vectors = _minimum_image(vectors, dimensions, pbc)
        squared = np.einsum("ai,ai->a", vectors, vectors)
        local_index = int(np.argmin(squared) if job.mode == "min" else np.argmax(squared))
        local_value = float(squared[local_index])
        better = local_value < best_squared if job.mode == "min" else local_value > best_squared
        if better:
            best_squared = local_value
            best_vector = vectors[local_index].copy()
    if best_vector is None:
        raise RuntimeError("An active entity pair produced no distance candidate.")
    return best_vector


def iter_distance_blocks(
    job,
    reference_positions,
    selection_positions,
    dimensions=None,
    pbc=True,
    entity_block_size=4096,
    candidate_block_size=65536,
) -> Iterator[DistanceBlock]:
    """Evaluate one frame as stable bounded channel blocks."""

    if pbc:
        validate_dimensions(dimensions)
    channel_start = 0
    for entity_pairs in iter_active_entity_pair_blocks(job, entity_block_size):
        # The overwhelmingly common many-chain case has one point per entity.
        one_point = all(
            reference_positions[int(ref_entity)].shape == (1, 3)
            and selection_positions[int(sel_entity)].shape == (1, 3)
            for ref_entity, sel_entity in entity_pairs
        )
        if one_point:
            ref_indices = entity_pairs[:, 0]
            sel_indices = entity_pairs[:, 1]
            ref_points = np.vstack(
                [reference_positions[int(index)][0] for index in ref_indices]
            )
            sel_points = np.vstack(
                [selection_positions[int(index)][0] for index in sel_indices]
            )
            vectors = _minimum_image(sel_points - ref_points, dimensions, pbc)
            values = np.linalg.norm(vectors, axis=1)
            yield DistanceBlock(channel_start, values, vectors)
            channel_start += int(values.size)
            continue

        values = []
        vectors = []
        for ref_entity, sel_entity in entity_pairs:
            ref_entity = int(ref_entity)
            sel_entity = int(sel_entity)
            ref_points = reference_positions[ref_entity]
            sel_points = selection_positions[sel_entity]
            if job.mode == "pair":
                pair_vectors = _minimum_image(sel_points - ref_points, dimensions, pbc)
                keep = _allowed_pair_mask(
                    job,
                    ref_entity,
                    sel_entity,
                    pair_vectors.shape[0],
                )
                pair_vectors = pair_vectors[keep]
                vectors.extend(pair_vectors)
                values.extend(np.linalg.norm(pair_vectors, axis=1))
            else:
                vector = _extreme_vector(
                    job,
                    ref_entity,
                    sel_entity,
                    ref_points,
                    sel_points,
                    dimensions,
                    pbc,
                    candidate_block_size,
                )
                vectors.append(vector)
                values.append(float(np.linalg.norm(vector)))
        value_array = np.asarray(values, dtype=np.float64)
        vector_array = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        if value_array.size:
            yield DistanceBlock(channel_start, value_array, vector_array)
            channel_start += int(value_array.size)
    if channel_start != job.channel_count:
        raise RuntimeError("Distance evaluation produced an unstable channel count.")


class RunningStatistics:
    """Stable scalar streaming statistics with batch updates."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        if not np.all(np.isfinite(array)):
            raise ValueError("Distance statistics received non-finite values.")
        batch_count = int(array.size)
        batch_mean = float(array.mean())
        batch_m2 = float(np.sum((array - batch_mean) ** 2))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    @property
    def std(self):
        return float(np.sqrt(self.m2 / self.count)) if self.count else np.nan


class ChannelStatistics:
    """One Welford accumulator per stable distance channel."""

    def __init__(self, channel_count):
        if channel_count <= 0:
            raise ValueError("Channel statistics require at least one channel.")
        self.count = np.zeros(channel_count, dtype=np.int64)
        self.mean = np.zeros(channel_count, dtype=np.float64)
        self.m2 = np.zeros(channel_count, dtype=np.float64)
        self.minimum = np.full(channel_count, np.inf, dtype=np.float64)
        self.maximum = np.full(channel_count, -np.inf, dtype=np.float64)

    def update(self, start, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        stop = int(start) + array.size
        if start < 0 or stop > self.count.size:
            raise IndexError("Channel statistic block is outside the channel range.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Channel statistics received non-finite values.")
        target = slice(int(start), stop)
        counts = self.count[target] + 1
        delta = array - self.mean[target]
        self.mean[target] += delta / counts
        delta2 = array - self.mean[target]
        self.m2[target] += delta * delta2
        self.count[target] = counts
        self.minimum[target] = np.minimum(self.minimum[target], array)
        self.maximum[target] = np.maximum(self.maximum[target], array)

    @property
    def std(self):
        result = np.full(self.count.size, np.nan, dtype=np.float64)
        valid = self.count > 0
        result[valid] = np.sqrt(self.m2[valid] / self.count[valid])
        return result


class HistogramAccumulator:
    """Streaming common-edge histograms for jobs or channels."""

    def __init__(self, edges, series_count):
        edges = np.asarray(edges, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("Histogram edges must be finite and increasing.")
        if series_count <= 0:
            raise ValueError("Histogram requires at least one series.")
        self.edges = edges
        self.counts = np.zeros((series_count, edges.size - 1), dtype=np.int64)

    def update_series(self, series, values):
        series = int(series)
        if not 0 <= series < self.counts.shape[0]:
            raise IndexError("Histogram series index is out of range.")
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        bins = np.searchsorted(self.edges, array, side="right") - 1
        bins[array == self.edges[-1]] = self.counts.shape[1] - 1
        valid = (bins >= 0) & (bins < self.counts.shape[1])
        np.add.at(self.counts[series], bins[valid], 1)

    def update_channels(self, start, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        rows = np.arange(int(start), int(start) + array.size, dtype=np.int64)
        if rows.size and (rows[0] < 0 or rows[-1] >= self.counts.shape[0]):
            raise IndexError("Histogram channel block is outside the channel range.")
        bins = np.searchsorted(self.edges, array, side="right") - 1
        bins[array == self.edges[-1]] = self.counts.shape[1] - 1
        valid = (bins >= 0) & (bins < self.counts.shape[1])
        np.add.at(self.counts, (rows[valid], bins[valid]), 1)

    def normalized(self, normalization):
        counts = self.counts.astype(np.float64)
        if normalization == "count":
            return counts
        totals = counts.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts),
            where=totals > 0.0,
        )
        if normalization == "probability":
            return probabilities
        if normalization == "density":
            return probabilities / np.diff(self.edges)[np.newaxis, :]
        raise ValueError(f"Unknown histogram normalization: {normalization}")
