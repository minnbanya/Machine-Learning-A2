import numpy as np
import scipy.ndimage as ndi

def _marching_cubes_classic(volume, level, spacing, gradient_direction): ...
def mesh_surface_area(verts, faces) -> float: ...
def _correct_mesh_orientation(
    volume, actual_verts, faces, spacing=(1.0, 1.0, 1.0), gradient_direction="descent"
): ...
