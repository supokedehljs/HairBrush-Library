# HairBrush Library

HairBrush Library is a modular collection of Blender sculpt brushes. It
currently contains one brush:

- **Ring Slide** slides a closed topology ring along its own contour.

![Ring Slide brush thumbnail](preview.png)

## Ring Slide usage

1. Enter Sculpt Mode.
2. Select **Ring Slide** from the brush asset shelf at the bottom of the 3D
   viewport.
3. Hover over the target topology ring. The selected ring is shown in orange.
4. Press and drag in the tangent direction to slide it.
5. Release to confirm, or press `Esc` / right-click to cancel.

Ring Slide hides Blender's radius cursor while active and uses a small pressure
dead zone for pen tablets. Subdivision Surface modifiers do not change the
source control mesh used by the operation.

## Compatibility

- Blender 5.3
- Mesh objects in Sculpt Mode
- Quad topology with a closed edge ring

## Development

Every brush is an independent module under `brushes/`. New brushes must own
their brush asset, operators, keymaps, draw handlers and runtime state; the
library entry point only registers the modules listed in `BRUSH_MODULES`.

```text
hairbrush_library/
├── __init__.py                 # Library registration and preferences
├── brushes/
│   └── ring_slide/
│       ├── __init__.py         # Ring Slide brush implementation
│       └── core.py             # Topology traversal and deformation
└── blender_manifest.toml
```

## License

GPL-3.0-or-later.
