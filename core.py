"""Topology and arc-length helpers for the Ring Slide prototype."""

from __future__ import annotations

from array import array
from bisect import bisect_right
from dataclasses import dataclass
import math

import bmesh
from mathutils import Vector


class RingSlideError(RuntimeError):
    """Raised when the selected face cannot produce a usable closed loop."""


@dataclass(frozen=True)
class EdgeLoop:
    """One opposite-edge walk through a strip of quads."""

    edge_indices: tuple[int, ...]
    closed: bool
    vertex_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class RingSnapshot:
    """Immutable geometry captured when the mouse button is pressed."""

    vertex_indices: tuple[int, ...]
    coordinates: tuple[Vector, ...]
    cumulative_arc: tuple[float, ...]
    total_length: float

    @property
    def count(self) -> int:
        return len(self.vertex_indices)


def _face_edges(mesh, face_index: int) -> tuple[int, ...]:
    polygon = mesh.polygons[face_index]
    return tuple(mesh.loops[loop_index].edge_index for loop_index in polygon.loop_indices)


def _edge_faces(mesh) -> list[list[int]]:
    result = [[] for _ in range(len(mesh.edges))]
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            edge_index = mesh.loops[loop_index].edge_index
            faces = result[edge_index]
            if not faces or faces[-1] != polygon.index:
                faces.append(polygon.index)
    return result


def trace_opposite_edge_loop(mesh, face_index: int, start_edge: int):
    """Walk Blender's topological edge-loop cycle from one edge.

    BMesh's radial/next-link walk follows connected edges through quad
    vertices. It does not inspect coordinates or nearest neighbors.
    """

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        start = bm.edges[start_edge]
        start_loops = [
            loop
            for loop in start.link_loops
            if loop.face is not None and loop.face.index == face_index
        ]
        if not start_loops:
            return EdgeLoop((), False)

        current_loop = start_loops[0]
        start_vertex = current_loop.vert
        edges: list[int] = []
        vertices: list[int] = []
        visited: set[int] = set()

        for _step in range(len(bm.edges) + 1):
            if current_loop.index in visited:
                return EdgeLoop(tuple(edges), False, tuple(vertices))
            visited.add(current_loop.index)
            edges.append(current_loop.edge.index)
            vertices.append(current_loop.vert.index)

            next_corner = current_loop.link_loop_next
            radial = next_corner.link_loop_radial_next
            if radial == next_corner or radial.face is None:
                return EdgeLoop(tuple(edges), False, tuple(vertices))
            if len(current_loop.face.verts) != 4 or len(radial.face.verts) != 4:
                return EdgeLoop(tuple(edges), False, tuple(vertices))
            current_loop = radial.link_loop_next

            if current_loop.edge.index == start_edge:
                closed = current_loop.vert == start_vertex and len(edges) >= 3
                return EdgeLoop(tuple(edges), closed, tuple(vertices))

        return EdgeLoop(tuple(edges), False, tuple(vertices))
    finally:
        bm.free()


def loop_vertex_order(mesh, loop: EdgeLoop) -> tuple[int, ...]:
    """Return the topology-ordered vertex cycle captured by the walk."""

    if not loop.closed or len(loop.vertex_indices) < 3:
        return ()
    if len(set(loop.vertex_indices)) != len(loop.vertex_indices):
        return ()
    return loop.vertex_indices


def opposite_loop_candidates(mesh, face_index: int):
    """Return the two topological edge loops crossing a selected quad."""

    face_edges = _face_edges(mesh, face_index)
    if len(face_edges) != 4:
        raise RingSlideError("点击的面不是四边形")

    return (
        trace_opposite_edge_loop(mesh, face_index, face_edges[0]),
        trace_opposite_edge_loop(mesh, face_index, face_edges[1]),
    )


def choose_ring(mesh, face_index: int):
    """Choose the closed opposite-edge group, preferring the shorter loop."""

    candidates = opposite_loop_candidates(mesh, face_index)
    closed = []
    for candidate in candidates:
        vertices = loop_vertex_order(mesh, candidate)
        if vertices:
            closed.append((candidate, vertices))

    if not closed:
        raise RingSlideError("四边形两组对边都没有形成闭合拓扑环")

    if len(closed) == 2:
        closed.sort(key=lambda item: len(item[1]))
        print(
            "[Ring Slide] 两组对边都闭合；"
            f"选择点数较少的一组 ({len(closed[0][1])} vs {len(closed[1][1])})。"
        )

    candidate, vertices = closed[0]
    print(
        f"[Ring Slide] 拓扑闭环: {len(vertices)} 点，"
        f"{len(candidate.edge_indices)} 条边。"
    )
    return candidate, vertices


def create_ring_snapshot(mesh, vertex_indices: tuple[int, ...]) -> RingSnapshot:
    if len(vertex_indices) < 3:
        raise RingSlideError("闭环点数不足")
    if len(set(vertex_indices)) != len(vertex_indices):
        raise RingSlideError("闭环顶点顺序包含重复点")

    coordinates = tuple(Vector(mesh.vertices[index].co).copy() for index in vertex_indices)
    cumulative = [0.0]
    for index in range(len(coordinates) - 1):
        cumulative.append(cumulative[-1] + (coordinates[index + 1] - coordinates[index]).length)
    total_length = cumulative[-1] + (coordinates[0] - coordinates[-1]).length
    if total_length <= 1.0e-12:
        raise RingSlideError("闭环总周长过小")
    cumulative.append(total_length)

    return RingSnapshot(
        vertex_indices=tuple(vertex_indices),
        coordinates=coordinates,
        cumulative_arc=tuple(cumulative),
        total_length=total_length,
    )


def point_at_arc_length(snapshot: RingSnapshot, arc_length: float) -> Vector:
    """Linearly interpolate a point on the captured closed polyline."""

    wrapped = arc_length % snapshot.total_length
    total = snapshot.count
    segment = bisect_right(snapshot.cumulative_arc, wrapped) - 1
    segment = max(0, min(segment, total - 1))

    start_s = snapshot.cumulative_arc[segment]
    end_s = snapshot.cumulative_arc[segment + 1]
    segment_length = end_s - start_s
    if segment_length <= 1.0e-12:
        return snapshot.coordinates[segment].copy()

    factor = (wrapped - start_s) / segment_length
    end_index = (segment + 1) % total
    return snapshot.coordinates[segment].lerp(snapshot.coordinates[end_index], factor)


def offset_ring_coordinates(snapshot: RingSnapshot, distance: float) -> tuple[Vector, ...]:
    """Return all ring points at original arc position plus one common offset."""

    return tuple(
        point_at_arc_length(snapshot, start + distance)
        for start in snapshot.cumulative_arc[:-1]
    )


def closest_point_on_ring(snapshot: RingSnapshot, point_world: Vector, matrix_world):
    """Return the closest point and ordered tangent on the captured ring."""

    best_distance_sq = math.inf
    best_point = None
    best_tangent = None
    total = snapshot.count

    for index in range(total):
        start = matrix_world @ snapshot.coordinates[index]
        end = matrix_world @ snapshot.coordinates[(index + 1) % total]
        edge = end - start
        edge_length_sq = edge.length_squared
        if edge_length_sq <= 1.0e-20:
            continue

        factor = (point_world - start).dot(edge) / edge_length_sq
        factor = max(0.0, min(1.0, factor))
        closest = start + edge * factor
        distance_sq = (point_world - closest).length_squared
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_point = closest
            best_tangent = edge.normalized()

    if best_point is None:
        raise RingSlideError("无法在闭环上定位点击位置")
    return best_point, best_tangent


def read_vertex_coordinates(mesh) -> array:
    values = array("f", [0.0]) * (len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", values)
    return values


def write_vertex_coordinates(mesh, coordinates: array) -> None:
    if len(coordinates) != len(mesh.vertices) * 3:
        raise ValueError("坐标缓冲区大小与网格不一致")
    mesh.vertices.foreach_set("co", coordinates)
    mesh.update()


def apply_offset_to_mesh(mesh, snapshot: RingSnapshot, distance: float, original_all: array) -> None:
    """Write the offset ring while preserving every non-ring vertex exactly."""

    working = array("f", original_all)
    targets = offset_ring_coordinates(snapshot, distance)
    for vertex_index, coordinate in zip(snapshot.vertex_indices, targets):
        offset = vertex_index * 3
        working[offset : offset + 3] = array(
            "f", (coordinate.x, coordinate.y, coordinate.z)
        )
    write_vertex_coordinates(mesh, working)


def distance_to_closed_polyline(point: Vector, coordinates: tuple[Vector, ...]) -> float:
    """Distance from a point to any segment of a closed polyline."""

    distance, _arc_position = project_to_closed_polyline(point, coordinates)
    return distance


def project_to_closed_polyline(point: Vector, coordinates: tuple[Vector, ...]):
    """Return distance and arc position of the closest polyline segment."""

    best_sq = math.inf
    best_arc = 0.0
    cumulative = 0.0
    for index, start in enumerate(coordinates):
        end = coordinates[(index + 1) % len(coordinates)]
        edge = end - start
        edge_length_sq = edge.length_squared
        edge_length = math.sqrt(edge_length_sq)
        if edge_length_sq <= 1.0e-20:
            closest = start
            factor = 0.0
        else:
            factor = (point - start).dot(edge) / edge_length_sq
            factor = max(0.0, min(1.0, factor))
            closest = start + edge * factor
        distance_sq = (point - closest).length_squared
        if distance_sq < best_sq:
            best_sq = distance_sq
            best_arc = cumulative + factor * edge_length
        cumulative += edge_length
    return math.sqrt(best_sq), best_arc


def arc_offset_from_mouse(
    mouse_xy,
    press_xy,
    tangent_screen,
    pixels_per_world_unit: float,
) -> float:
    """Project mouse travel onto a screen tangent and convert to world length."""

    if pixels_per_world_unit <= 1.0e-12:
        raise RingSlideError("闭环切线在当前视角下无法投影")
    tangent = Vector((tangent_screen[0], tangent_screen[1]))
    tangent_length = tangent.length
    if tangent_length <= 1.0e-12:
        raise RingSlideError("闭环切线在当前视角下长度为 0")
    tangent.normalize()

    mouse_delta = Vector((mouse_xy[0] - press_xy[0], mouse_xy[1] - press_xy[1]))
    pixels_along_tangent = mouse_delta.dot(tangent)
    return pixels_along_tangent / pixels_per_world_unit

