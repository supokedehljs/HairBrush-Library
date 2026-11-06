# Ring Slide

Ring Slide is a Blender 5.3 sculpt brush for sliding a closed topology ring
along its own contour.

![Ring Slide brush thumbnail](preview.png)

## Usage

1. Enter Sculpt Mode.
2. Select **Ring Slide** from the brush asset shelf at the bottom of the 3D
   viewport.
3. Hover over the target topology ring. The selected ring is shown in orange.
4. Press and drag in the tangent direction to slide it.
5. Release to confirm, or press `Esc` / right-click to cancel.

The brush hides Blender's radius cursor while active and uses a small pressure
dead zone for pen tablets. Subdivision Surface modifiers do not change the
source control mesh used by the operation.

## Compatibility

- Blender 5.3
- Mesh objects in Sculpt Mode
- Quad topology with a closed edge ring

## Development

The extension files are:

- `__init__.py` - registration, brush asset, cursor overlay, picking and modal
  interaction
- `core.py` - topology traversal and ring deformation
- `blender_manifest.toml` - extension metadata

## License

GPL-3.0-or-later.
