"""Ring Slide brush module for HairBrush Library."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import bpy
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import core


BRUSH_ID = "ring_slide"
BRUSH_NAME = "Ring Slide"
DISPLAY_NAME = BRUSH_NAME
BRUSH_MARKER = "hairbrush_library:ring_slide"
LEGACY_BRUSH_MARKERS = ("ring_slide_brush",)
BRUSH_OPERATOR = "hairbrush_library.ring_slide_modal"
HOVER_OPERATOR = "hairbrush_library.ring_slide_hover_watch"
BRUSH_VERSION = 3
HOVER_UPDATE_INTERVAL = 1.0 / 45.0
PRESSURE_DEADZONE = 0.02


@dataclass
class _DragReference:
    press_xy: tuple[float, float]
    tangent_screen: Vector
    pixels_per_world_unit: float


def _is_ring_slide_brush(brush) -> bool:
    if brush is None:
        return False
    if bool(brush.get(BRUSH_MARKER, False)):
        return True
    return any(bool(brush.get(marker, False)) for marker in LEGACY_BRUSH_MARKERS)


def _find_ring_slide_brush():
    brushes = getattr(bpy.data, "brushes", None)
    if brushes is None:
        return None
    for brush in brushes:
        if bool(brush.get(BRUSH_MARKER, False)):
            return brush
    return next((brush for brush in brushes if _is_ring_slide_brush(brush)), None)


def _brush_preview_pixels(size: int = 64):
    """Render a faceted topology ring with an orange target cross section."""

    def mix(start, end, factor):
        factor = max(0.0, min(1.0, factor))
        return tuple(
            int(start[index] + (end[index] - start[index]) * factor)
            for index in range(3)
        ) + (255,)

    def inside_rounded_panel(x, y):
        inset = size * 0.045
        radius = size * 0.16
        left, right = inset, size - inset
        top, bottom = inset, size - inset
        dx = max(left + radius - x, 0.0, x - (right - radius))
        dy = max(top + radius - y, 0.0, y - (bottom - radius))
        return left <= x <= right and top <= y <= bottom and (
            math.hypot(dx, dy) <= radius
        )

    def sample(x, y):
        if not inside_rounded_panel(x, y):
            return (0, 0, 0, 0)

        panel_low = (19, 28, 39)
        panel_high = (33, 47, 62)
        color = mix(panel_low, panel_high, y / size)

        inset = size * 0.045
        border = min(x - inset, y - inset, size - inset - x, size - inset - y)
        if border < 1.15:
            color = mix(color, (104, 129, 151), 0.65)

        center_x = size * 0.50
        center_y = size * 0.52
        rx = size * 0.355
        ry = size * 0.185
        dx = (x - center_x) / rx
        dy = (y - center_y) / ry
        radial = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx)

        if 0.66 < radial < 1.06:
            edge_distance = min(radial - 0.66, 1.06 - radial)
            center_light = min(edge_distance / 0.18, 1.0)
            front_light = 0.55 + 0.30 * (0.5 + 0.5 * math.cos(angle + 0.65))
            surface = mix(
                (47, 65, 82),
                (106, 132, 154),
                front_light * (0.55 + 0.45 * center_light),
            )

            segment_phase = (angle + math.pi) * 8.0 / (2.0 * math.pi)
            if abs(segment_phase - round(segment_phase)) < 0.055:
                surface = mix(surface, (33, 48, 63), 0.70)
            if edge_distance < 0.055:
                surface = mix(surface, (24, 35, 47), 0.82)
            color = surface

        # A front-facing cross section makes the brush's target loop readable.
        ring_x = (x - size * 0.50) / (size * 0.066)
        ring_y = (y - size * 0.51) / (size * 0.225)
        ring_distance = abs(math.sqrt(ring_x * ring_x + ring_y * ring_y) - 1.0)
        if ring_distance < 0.105:
            orange = (242, 158, 90, 255)
            if ring_distance < 0.045:
                color = orange
            else:
                color = mix((104, 67, 40), orange, 0.62)

        return color

    samples = 3
    float_pixels = []
    int_pixels = []
    sample_step = 1.0 / samples
    for y in range(size):
        for x in range(size):
            total = [0.0, 0.0, 0.0, 0.0]
            for sy in range(samples):
                for sx in range(samples):
                    rgba = sample(
                        x + (sx + 0.5) * sample_step,
                        y + (sy + 0.5) * sample_step,
                    )
                    alpha = rgba[3] / 255.0
                    total[0] += rgba[0] * alpha
                    total[1] += rgba[1] * alpha
                    total[2] += rgba[2] * alpha
                    total[3] += rgba[3]

            count = samples * samples
            alpha_byte = max(0, min(255, round(total[3] / count)))
            if alpha_byte:
                alpha = alpha_byte / 255.0
                red = max(0, min(255, round(total[0] / count / alpha)))
                green = max(0, min(255, round(total[1] / count / alpha)))
                blue = max(0, min(255, round(total[2] / count / alpha)))
            else:
                red = green = blue = 0

            float_pixels.extend((red / 255.0, green / 255.0, blue / 255.0, alpha_byte / 255.0))
            packed = (alpha_byte << 24) | (blue << 16) | (green << 8) | red
            int_pixels.append(packed - (1 << 32) if packed >= (1 << 31) else packed)

    return int_pixels, float_pixels


def _set_brush_preview(brush) -> None:
    try:
        preview = brush.preview_ensure()
        size = 64
        preview.image_size = (size, size)
        int_pixels, float_pixels = _brush_preview_pixels(size)
        if len(preview.image_pixels_float) == len(float_pixels):
            preview.image_pixels_float = float_pixels
            preview.is_image_custom = True
        elif len(preview.image_pixels) == len(int_pixels):
            preview.image_pixels = int_pixels
            preview.is_image_custom = True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # A missing thumbnail must never prevent the brush from being usable.
        pass


def _ensure_ring_slide_brush():
    """Create or repair the brush data-block used by the asset shelf."""

    brush = _find_ring_slide_brush()
    if brush is None:
        name = BRUSH_NAME
        suffix = 1
        while bpy.data.brushes.get(name) is not None:
            suffix += 1
            name = f"{BRUSH_NAME} {suffix}"
        brush = bpy.data.brushes.new(name=name, mode="SCULPT")

    brush[BRUSH_MARKER] = True
    for legacy_marker in LEGACY_BRUSH_MARKERS:
        brush[legacy_marker] = False
    brush["ring_slide_version"] = BRUSH_VERSION
    brush["ring_slide_operator"] = BRUSH_OPERATOR
    brush.use_fake_user = True
    brush.use_paint_sculpt = True
    brush.sculpt_brush_type = "DRAW"
    brush.use_frontface = False
    brush.use_cursor_overlay = False
    brush.use_cursor_overlay_override = True

    if brush.asset_data is None:
        brush.asset_mark()

    asset_data = brush.asset_data
    asset_data.author = "OpenAI"
    asset_data.description = (
        "点击一个四边面环，然后沿屏幕切线方向拖动。松开确认，Esc 或右键取消。"
    )
    asset_data.copyright = ""
    asset_data.license = "SPDX:GPL-3.0-or-later"

    existing_tags = {tag.name for tag in asset_data.tags}
    for tag_name in ("HairBrush Library", "Mesh", "Sculpt", "Topology", "Ring Slide"):
        if tag_name not in existing_tags:
            try:
                asset_data.tags.new(tag_name, skip_if_exists=True)
            except TypeError:
                asset_data.tags.new(tag_name)
            except RuntimeError:
                pass

    _set_brush_preview(brush)
    return brush


def ensure_brush():
    """Public registration hook used by the HairBrush Library manager."""

    return _ensure_ring_slide_brush()


def _active_ring_slide_brush(context):
    if context.mode != "SCULPT" or context.tool_settings is None:
        return None
    sculpt = getattr(context.tool_settings, "sculpt", None)
    brush = getattr(sculpt, "brush", None)
    return brush if _is_ring_slide_brush(brush) else None


def _redraw_view(context) -> None:
    if context.view_layer is not None:
        context.view_layer.update()
    if context.area is not None:
        context.area.tag_redraw()


def _pressure_passes_deadzone(event) -> bool:
    """Ignore the low-pressure tail of a stylus stroke."""

    if not getattr(event, "is_tablet", False):
        return True
    try:
        return float(event.pressure) > PRESSURE_DEADZONE
    except (AttributeError, TypeError, ValueError):
        return True


def _screen_ray(context, mouse_x: float, mouse_y: float):
    global _base_mesh_bvh_cache

    region = context.region
    region_data = context.region_data
    if region is None or region_data is None or region.type != "WINDOW":
        return None

    coordinate = (mouse_x, mouse_y)
    origin = view3d_utils.region_2d_to_origin_3d(region, region_data, coordinate)
    direction = view3d_utils.region_2d_to_vector_3d(region, region_data, coordinate)
    obj = context.object
    if obj is None or obj.type != "MESH":
        return None

    mesh = obj.data
    cache_key = (obj.as_pointer(), len(mesh.vertices), len(mesh.polygons))
    if _base_mesh_bvh_cache is None or _base_mesh_bvh_cache[0] != cache_key:
        try:
            bvh = BVHTree.FromPolygons(
                [vertex.co.copy() for vertex in mesh.vertices],
                [tuple(polygon.vertices) for polygon in mesh.polygons],
                all_triangles=False,
                epsilon=0.0,
            )
        except (RuntimeError, ValueError):
            return None
        _base_mesh_bvh_cache = (cache_key, bvh)

    matrix_inverse = obj.matrix_world.inverted_safe()
    local_origin = matrix_inverse @ origin
    local_direction = (matrix_inverse.to_3x3() @ direction).normalized()
    location, _normal, face_index, _distance = _base_mesh_bvh_cache[1].ray_cast(
        local_origin,
        local_direction,
    )
    if location is None:
        return None
    if face_index < 0 or face_index >= len(mesh.polygons):
        return None

    # Pick the source cage, so subdivision faces never leak into ring indices.
    return obj, obj.matrix_world @ Vector(location), face_index


@dataclass(frozen=True)
class _HoverRing:
    object_pointer: int
    window_pointer: int
    area_pointer: int
    coordinates: tuple[Vector, ...]
    matrix_world: object


_hover_ring = None
_hover_cache = None
_hover_last_update = 0.0
_hover_watcher = None
_draw_handle = None
_drag_active = False
_base_mesh_bvh_cache = None


def _point_segment_distance_2d(point, start, end) -> float:
    edge = end - start
    edge_length_sq = edge.length_squared
    if edge_length_sq <= 1.0e-12:
        return (point - start).length
    factor = (point - start).dot(edge) / edge_length_sq
    factor = max(0.0, min(1.0, factor))
    return (point - (start + edge * factor)).length


def _ring_screen_distance(context, obj, snapshot, mouse_xy) -> float:
    region = context.region
    region_data = context.region_data
    if region is None or region_data is None:
        return math.inf

    matrix = obj.matrix_world
    points = [
        view3d_utils.location_3d_to_region_2d(
            region,
            region_data,
            matrix @ coordinate,
        )
        for coordinate in snapshot.coordinates
    ]
    if any(point is None for point in points):
        return math.inf

    mouse = Vector(mouse_xy)
    return min(
        _point_segment_distance_2d(
            mouse,
            points[index],
            points[(index + 1) % len(points)],
        )
        for index in range(len(points))
    )


def _ring_choices_for_face(obj, face_index: int):
    global _hover_cache

    pointer = obj.as_pointer()
    if (
        _hover_cache is not None
        and _hover_cache[0] == pointer
        and _hover_cache[1] == face_index
    ):
        return _hover_cache[2]

    choices = []
    for candidate in core.opposite_loop_candidates(obj.data, face_index):
        vertex_indices = core.loop_vertex_order(obj.data, candidate)
        if not vertex_indices:
            continue
        try:
            snapshot = core.create_ring_snapshot(obj.data, vertex_indices)
        except core.RingSlideError:
            continue
        choices.append((candidate, snapshot))

    _hover_cache = (pointer, face_index, choices)
    return choices


def _find_ring_under_cursor(context, mouse_x: float, mouse_y: float):
    """Resolve the hovered ring from the surface point under the cursor."""

    picked = _screen_ray(context, mouse_x, mouse_y)
    if picked is None:
        return None

    obj, hit_world, face_index = picked
    if obj != context.object:
        return None

    choices = _ring_choices_for_face(obj, face_index)
    if not choices:
        return None

    mouse_xy = (mouse_x, mouse_y)
    ranked = []
    for candidate, snapshot in choices:
        distance = _ring_screen_distance(context, obj, snapshot, mouse_xy)
        if math.isfinite(distance):
            ranked.append((distance, -len(snapshot.vertex_indices), candidate, snapshot))
    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1]))
    _distance, _negative_length, candidate, snapshot = ranked[0]
    return obj, hit_world, face_index, candidate, snapshot


def _clear_hover(context=None, *, clear_cache=False) -> None:
    global _hover_ring, _hover_cache, _base_mesh_bvh_cache

    had_hover = _hover_ring is not None
    _hover_ring = None
    if clear_cache:
        _hover_cache = None
        _base_mesh_bvh_cache = None
    if had_hover and context is not None and context.area is not None:
        context.area.tag_redraw()


def _update_hover(context, event, force=False) -> None:
    global _hover_ring, _hover_last_update

    if _drag_active or _active_ring_slide_brush(context) is None:
        _clear_hover(context, clear_cache=True)
        return
    if context.area is None or context.area.type != "VIEW_3D":
        _clear_hover(context, clear_cache=True)
        return
    if context.region is None or context.region.type != "WINDOW":
        _clear_hover(context, clear_cache=True)
        return

    now = time.monotonic()
    if not force and now - _hover_last_update < HOVER_UPDATE_INTERVAL:
        return
    _hover_last_update = now

    try:
        found = _find_ring_under_cursor(
            context,
            event.mouse_region_x,
            event.mouse_region_y,
        )
    except (
        AttributeError,
        IndexError,
        core.RingSlideError,
        ReferenceError,
        RuntimeError,
        ValueError,
    ):
        found = None

    if found is None:
        _clear_hover(context)
        return

    (
        obj,
        _hit_world,
        _face_index,
        _candidate,
        snapshot,
    ) = found
    _hover_ring = _HoverRing(
        object_pointer=obj.as_pointer(),
        window_pointer=context.window.as_pointer() if context.window else 0,
        area_pointer=context.area.as_pointer(),
        coordinates=snapshot.coordinates,
        matrix_world=obj.matrix_world.copy(),
    )
    context.area.tag_redraw()


def _ribbon_triangles(points, half_width: float) -> list[tuple[float, float]]:
    """Build a rounded, screen-space ribbon that does not depend on GL line width."""

    vertices = []
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        direction = end - start
        length = direction.length
        if length <= 1.0e-6:
            continue
        normal = Vector((-direction.y, direction.x)) * (half_width / length)
        a = start + normal
        b = start - normal
        c = end + normal
        d = end - normal
        vertices.extend(((a.x, a.y), (b.x, b.y), (c.x, c.y)))
        vertices.extend(((b.x, b.y), (d.x, d.y), (c.x, c.y)))

    # Rounded joins remove the hard notches that line strips show on dense loops.
    segments = 10
    for point in points[:-1]:
        for index in range(segments):
            angle_a = math.tau * index / segments
            angle_b = math.tau * (index + 1) / segments
            vertices.extend(
                (
                    (point.x, point.y),
                    (
                        point.x + math.cos(angle_a) * half_width,
                        point.y + math.sin(angle_a) * half_width,
                    ),
                    (
                        point.x + math.cos(angle_b) * half_width,
                        point.y + math.sin(angle_b) * half_width,
                    ),
                )
            )
    return vertices


def _draw_ribbon(shader, points, half_width: float, color) -> None:
    vertices = _ribbon_triangles(points, half_width)
    if not vertices:
        return
    batch = batch_for_shader(shader, "TRIS", {"pos": vertices})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_hover_ring():
    state = _hover_ring
    if state is None or _drag_active:
        return

    context = bpy.context
    if _active_ring_slide_brush(context) is None:
        return
    if context.window is None or context.area is None:
        return
    if context.window.as_pointer() != state.window_pointer:
        return
    if context.area.as_pointer() != state.area_pointer:
        return
    if context.region is None or context.region_data is None:
        return

    points = [
        view3d_utils.location_3d_to_region_2d(
            context.region,
            context.region_data,
            state.matrix_world @ coordinate,
        )
        for coordinate in state.coordinates
    ]
    if len(points) < 3 or any(point is None for point in points):
        return
    points.append(points[0])

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    try:
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        # Muted Blender selection orange, defined in linear color space.
        _draw_ribbon(shader, points, 1.75, (0.90, 0.34, 0.10, 1.0))
    except (ReferenceError, RuntimeError):
        pass
    finally:
        gpu.state.blend_set("NONE")


def _nearest_vertex_on_face(obj, face_index: int, hit_world: Vector) -> int:
    polygon = obj.data.polygons[face_index]
    matrix = obj.matrix_world
    return min(
        polygon.vertices,
        key=lambda index: (matrix @ obj.data.vertices[index].co - hit_world).length_squared,
    )


def _ring_screen_reference(context, obj, snapshot, hit_world, press_xy):
    ring_point, tangent_world = core.closest_point_on_ring(
        snapshot, hit_world, obj.matrix_world
    )
    if context.region is None or context.region_data is None:
        raise core.RingSlideError("起点不在 3D 视口中")

    screen_point = view3d_utils.location_3d_to_region_2d(
        context.region, context.region_data, ring_point
    )
    probe_world = max(snapshot.total_length / 200.0, 1.0e-6)
    screen_probe = view3d_utils.location_3d_to_region_2d(
        context.region,
        context.region_data,
        ring_point + tangent_world * probe_world,
    )
    if screen_point is None or screen_probe is None:
        raise core.RingSlideError("闭环切线不在当前视口中")

    screen_delta = screen_probe - screen_point
    if screen_delta.length <= 1.0e-6:
        raise core.RingSlideError("闭环切线在当前视角下无法投影")

    return _DragReference(
        press_xy=tuple(press_xy),
        tangent_screen=screen_delta,
        pixels_per_world_unit=screen_delta.length / probe_world,
    )


class RING_SLIDE_OT_modal(bpy.types.Operator):
    """Slide one topology ring along its original closed contour."""

    bl_idname = BRUSH_OPERATOR
    bl_label = "Ring Slide"
    bl_description = "点击一条四边形截面环，然后沿屏幕切线方向拖动"
    bl_options = {"REGISTER", "UNDO", "BLOCKING"}

    direct_start: bpy.props.BoolProperty(default=True, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.mode == "SCULPT"
            and context.object is not None
            and context.object.type == "MESH"
            and _active_ring_slide_brush(context) is not None
        )

    def invoke(self, context, event):
        self._reset_state()

        if context.window is None or event is None:
            self.report({"ERROR"}, "Ring Slide 需要交互式 3D 视口")
            return {"CANCELLED"}

        if not self.direct_start:
            self.report({"WARNING"}, "请从雕刻模式底部笔刷栏选择 Ring Slide")
            return {"FINISHED"}
        if not _pressure_passes_deadzone(event):
            return {"FINISHED"}

        try:
            picked = _find_ring_under_cursor(
                context,
                event.mouse_region_x,
                event.mouse_region_y,
            )
        except (
            AttributeError,
            IndexError,
            core.RingSlideError,
            ReferenceError,
            RuntimeError,
            ValueError,
        ):
            picked = None

        if picked is None:
            # Do not add a modal handler before the click is known to be valid.
            # The old order left a freed operator in Blender's modal stack.
            self.report({"WARNING"}, "这里没有可用的网格截面环")
            return {"FINISHED"}

        (
            obj,
            hit_world,
            face_index,
            _candidate,
            snapshot,
        ) = picked
        if obj != context.object:
            self.report({"WARNING"}, "请对当前激活的网格使用 Ring Slide")
            return {"FINISHED"}

        try:
            self._begin(
                context,
                event,
                obj,
                hit_world,
                face_index,
                snapshot,
            )
        except core.RingSlideError as exc:
            self.report({"WARNING"}, str(exc))
            return {"FINISHED"}
        except (IndexError, ReferenceError, RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"Ring Slide 无法开始：{exc}")
            return {"FINISHED"}

        context.window_manager.modal_handler_add(self)
        context.window.cursor_set("PAINT_BRUSH")
        return {"RUNNING_MODAL"}

    def _reset_state(self):
        self._phase = "IDLE"
        self._obj = None
        self._snapshot = None
        self._original_all = None
        self._reference = None
        self._last_distance = 0.0
        self._last_mouse = None

    def _begin(
        self,
        context,
        event,
        obj,
        hit_world,
        face_index,
        snapshot,
    ):
        global _drag_active

        mesh = obj.data
        if mesh is None or len(mesh.polygons) == 0:
            raise core.RingSlideError("网格没有可编辑的拓扑")

        hint_vertex = _nearest_vertex_on_face(obj, face_index, hit_world)
        vertex_indices = snapshot.vertex_indices
        original_all = core.read_vertex_coordinates(mesh)
        reference = _ring_screen_reference(
            context,
            obj,
            snapshot,
            hit_world,
            (event.mouse_region_x, event.mouse_region_y),
        )

        self._obj = obj
        self._snapshot = snapshot
        self._original_all = original_all
        self._reference = reference
        self._last_mouse = (event.mouse_region_x, event.mouse_region_y)
        self._last_distance = 0.0
        self._phase = "DRAGGING"
        _drag_active = True
        _clear_hover(context, clear_cache=True)

        print(
            f"[Ring Slide] 点击面 {face_index}，最近环点 {hint_vertex}，"
            f"环顶点 {len(vertex_indices)}，周长 {snapshot.total_length:.9f}。"
        )
        self.report(
            {"INFO"},
            f"环: {len(vertex_indices)} 点 / L={snapshot.total_length:.4f}",
        )

    def _update_drag(self, context, event):
        if self._obj is None or self._snapshot is None:
            return
        if event.mouse_region_x == self._last_mouse[0] and event.mouse_region_y == self._last_mouse[1]:
            return

        distance = core.arc_offset_from_mouse(
            (event.mouse_region_x, event.mouse_region_y),
            self._reference.press_xy,
            self._reference.tangent_screen,
            self._reference.pixels_per_world_unit,
        )
        if abs(distance - self._last_distance) <= 1.0e-9:
            return

        core.apply_offset_to_mesh(
            self._obj.data,
            self._snapshot,
            distance,
            self._original_all,
        )
        self._obj.update_tag()
        self._last_mouse = (event.mouse_region_x, event.mouse_region_y)
        self._last_distance = distance
        _redraw_view(context)

    def _finish(self, context):
        global _drag_active

        if self._obj is not None:
            self._obj.update_tag()
            _redraw_view(context)
            print(f"[Ring Slide] 确认滑动，d={self._last_distance:.9f}。")
        _drag_active = False
        _clear_hover(context, clear_cache=True)
        return {"FINISHED"}

    def _cancel(self, context):
        global _drag_active

        if self._obj is not None and self._snapshot is not None and self._original_all is not None:
            if len(self._original_all) == len(self._obj.data.vertices) * 3:
                core.write_vertex_coordinates(self._obj.data, self._original_all)
                self._obj.update_tag()
                _redraw_view(context)
                print("[Ring Slide] 已取消并还原按下前坐标。")
        _drag_active = False
        _clear_hover(context, clear_cache=True)
        return {"CANCELLED"}

    def modal(self, context, event):
        try:
            if event.type == "ESC" or (
                event.type == "RIGHTMOUSE" and event.value == "PRESS"
            ):
                return self._cancel(context)

            if event.type == "WINDOW_DEACTIVATE":
                return self._cancel(context)

            if self._phase != "DRAGGING":
                return {"CANCELLED"}

            if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
                if _pressure_passes_deadzone(event):
                    self._update_drag(context, event)
                return {"RUNNING_MODAL"}

            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                # Do not apply the release coordinate. This keeps the final
                # released stroke from drifting past the intended position.
                return self._finish(context)
        except core.RingSlideError as exc:
            self._cancel(context)
            self.report({"WARNING"}, f"Ring Slide 已取消：{exc}")
            return {"CANCELLED"}
        except (AttributeError, IndexError, ReferenceError, RuntimeError, ValueError) as exc:
            self._cancel(context)
            self.report({"ERROR"}, f"Ring Slide 已安全取消：{exc}")
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}


_cursor_window_pointer = 0
_cursor_kind = None
_show_brush_saved = None


def _apply_sculpt_brush_overlay(context) -> None:
    """Hide Blender's radius circle only while Ring Slide is active."""

    global _show_brush_saved

    tool_settings = getattr(context, "tool_settings", None)
    sculpt = getattr(tool_settings, "sculpt", None)
    if sculpt is None:
        return

    pointer = tool_settings.as_pointer()
    if _show_brush_saved is not None and _show_brush_saved[0] != pointer:
        _show_brush_saved = None

    if _active_ring_slide_brush(context) is not None:
        if _show_brush_saved is None:
            _show_brush_saved = (pointer, bool(sculpt.show_brush))
        if sculpt.show_brush:
            sculpt.show_brush = False
    elif _show_brush_saved is not None and _show_brush_saved[0] == pointer:
        sculpt.show_brush = _show_brush_saved[1]
        _show_brush_saved = None


def _apply_brush_cursor(context) -> None:
    """Use the pen cursor in the viewport and keep the native cursor elsewhere."""

    global _cursor_window_pointer, _cursor_kind

    window = context.window
    if window is None:
        return

    desired = "DEFAULT"
    if (
        context.mode == "SCULPT"
        and context.area is not None
        and context.area.type == "VIEW_3D"
    ):
        desired = "PAINT_BRUSH"

    pointer = window.as_pointer()
    if pointer == _cursor_window_pointer and desired == _cursor_kind:
        return
    try:
        window.cursor_set(desired)
        _cursor_window_pointer = pointer
        _cursor_kind = desired
    except (AttributeError, ReferenceError, RuntimeError):
        pass


class RING_SLIDE_OT_hover_watch(bpy.types.Operator):
    """Track the pen tip and highlight only Ring Slide's hovered ring."""

    bl_idname = HOVER_OPERATOR
    bl_label = "Ring Slide Hover Watcher"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def invoke(self, context, event):
        global _hover_watcher

        if bpy.app.background:
            return {"CANCELLED"}
        if _hover_watcher is not None:
            try:
                if _hover_watcher._watching:
                    return {"CANCELLED"}
            except ReferenceError:
                _hover_watcher = None

        self._watching = True
        self._stop_requested = False
        context.window_manager.modal_handler_add(self)
        _hover_watcher = self
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _hover_watcher

        try:
            if self._stop_requested:
                return self.cancel(context)

            _apply_brush_cursor(context)
            _apply_sculpt_brush_overlay(context)

            if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"}:
                _update_hover(context, event)
            elif _active_ring_slide_brush(context) is None:
                _clear_hover(context)
        except (AttributeError, IndexError, ReferenceError, RuntimeError, ValueError):
            _clear_hover(context)
        return {"PASS_THROUGH"}

    def cancel(self, context):
        global _hover_watcher, _cursor_kind, _cursor_window_pointer, _show_brush_saved

        self._watching = False
        self._stop_requested = True
        if _hover_watcher is self:
            _hover_watcher = None
        _clear_hover(context, clear_cache=True)
        tool_settings = getattr(context, "tool_settings", None)
        sculpt = getattr(tool_settings, "sculpt", None)
        if (
            sculpt is not None
            and _show_brush_saved is not None
            and _show_brush_saved[0] == tool_settings.as_pointer()
        ):
            sculpt.show_brush = _show_brush_saved[1]
        _show_brush_saved = None
        if context.window is not None:
            try:
                context.window.cursor_set("DEFAULT")
            except (AttributeError, ReferenceError, RuntimeError):
                pass
        _cursor_kind = None
        _cursor_window_pointer = 0
        return {"CANCELLED"}


classes = (RING_SLIDE_OT_modal, RING_SLIDE_OT_hover_watch)
_keymap_items = []


def _ensure_after_load(_scene=None):
    """Wait until file data is available instead of falling back to _RestrictData."""

    try:
        _ensure_ring_slide_brush()
    except (AttributeError, RuntimeError):
        pass
    _start_hover_watcher()


def _start_hover_watcher() -> None:
    if bpy.app.background or _hover_watcher is not None:
        return

    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next(
                (candidate for candidate in area.regions if candidate.type == "WINDOW"),
                None,
            )
            if region is None:
                continue
            try:
                with bpy.context.temp_override(
                    window=window,
                    screen=screen,
                    area=area,
                    region=region,
                ):
                    result = bpy.ops.hairbrush_library.ring_slide_hover_watch(
                        "INVOKE_DEFAULT"
                    )
            except (AttributeError, RuntimeError, TypeError):
                continue
            if "RUNNING_MODAL" in result:
                return


def _register_keymaps():
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is None:
        return

    keymap = keyconfig.keymaps.get("Sculpt")
    if keymap is None:
        keymap = keyconfig.keymaps.new(
            name="Sculpt", space_type="EMPTY", region_type="WINDOW"
        )

    modifier_combinations = (
        {},
        {"ctrl": True},
        {"shift": True},
        {"alt": True},
        {"ctrl": True, "shift": True},
        {"ctrl": True, "alt": True},
    )
    for modifiers in modifier_combinations:
        item = keymap.keymap_items.new(
            BRUSH_OPERATOR,
            type="LEFTMOUSE",
            value="PRESS",
            **modifiers,
        )
        _keymap_items.append((keymap, item))


def register():
    global _draw_handle

    for cls in classes:
        bpy.utils.register_class(cls)

    _register_keymaps()
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_hover_ring,
            (),
            "WINDOW",
            "POST_PIXEL",
        )

    if _ensure_after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_ensure_after_load)
    if not bpy.app.timers.is_registered(_ensure_after_load):
        bpy.app.timers.register(_ensure_after_load, first_interval=0.1)


def unregister():
    global _draw_handle, _hover_watcher, _drag_active, _show_brush_saved

    if _ensure_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_ensure_after_load)
    if bpy.app.timers.is_registered(_ensure_after_load):
        bpy.app.timers.unregister(_ensure_after_load)

    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        except (ReferenceError, RuntimeError, ValueError):
            pass
        _draw_handle = None

    watcher = _hover_watcher
    _hover_watcher = None
    if watcher is not None:
        try:
            watcher._stop_requested = True
        except ReferenceError:
            pass
    _drag_active = False
    _clear_hover(clear_cache=True)
    tool_settings = getattr(bpy.context, "tool_settings", None)
    sculpt = getattr(tool_settings, "sculpt", None)
    if (
        sculpt is not None
        and _show_brush_saved is not None
        and _show_brush_saved[0] == tool_settings.as_pointer()
    ):
        sculpt.show_brush = _show_brush_saved[1]
    _show_brush_saved = None

    for keymap, item in tuple(_keymap_items):
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError):
            pass
    _keymap_items.clear()

    brush = _find_ring_slide_brush()
    if brush is not None:
        try:
            brush.asset_clear()
        except RuntimeError:
            pass
        brush.use_fake_user = False
        if brush.users == 0:
            bpy.data.brushes.remove(brush)

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
