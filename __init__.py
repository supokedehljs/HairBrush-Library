"""HairBrush Library: a collection of independent sculpt brushes."""

from __future__ import annotations

import bpy

from .brushes import ring_slide


bl_info = {
    "name": "HairBrush Library",
    "author": "OpenAI",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "Sculpt Mode > Brush Asset Shelf",
    "description": "A modular collection of sculpt brushes",
    "category": "Sculpting",
}


# Adding a brush only requires creating an independent module and listing it here.
BRUSH_MODULES = (ring_slide,)
_registered_brush_modules = []


class HAIRBRUSH_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        layout.label(text="HairBrush Library", icon="BRUSH_DATA")
        layout.label(text=f"已注册 {len(BRUSH_MODULES)} 个雕刻笔刷")

        box = layout.box()
        box.label(text="已安装笔刷")
        for brush_module in BRUSH_MODULES:
            row = box.row()
            row.label(text=brush_module.DISPLAY_NAME, icon="BRUSH_DATA")

        layout.separator()
        layout.label(text="请在雕刻模式底部的笔刷资源栏中选择笔刷。")


def ensure_all_brushes():
    """Create or repair every registered brush asset."""

    for brush_module in BRUSH_MODULES:
        ensure = getattr(brush_module, "ensure_brush", None)
        if ensure is not None:
            ensure()


def register():
    global _registered_brush_modules

    _registered_brush_modules = []
    try:
        bpy.utils.register_class(HAIRBRUSH_AddonPreferences)
        for brush_module in BRUSH_MODULES:
            brush_module.register()
            _registered_brush_modules.append(brush_module)
    except Exception:
        unregister()
        raise


def unregister():
    global _registered_brush_modules

    while _registered_brush_modules:
        brush_module = _registered_brush_modules.pop()
        try:
            brush_module.unregister()
        except (AttributeError, RuntimeError):
            pass

    try:
        bpy.utils.unregister_class(HAIRBRUSH_AddonPreferences)
    except RuntimeError:
        pass


if __name__ == "__main__":
    register()
