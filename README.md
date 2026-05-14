# Regrid

Converts unstructured tokamak simulation grids into uniform voxel grids, then exports to OpenVDB for large file-size reductions (up to 99.95%).

This code was developed as part of the following paper:

> **AI-Machine Learning-Enabled Tokamak Digital Twin** <br>
> William Tang, Eliot Feibush, Ge Dong, Noah Borthwick, Apollo Lee, Juan-Felipe Gomez, Tom Gibbs, John Stone, Peter Messmer, Jack Wells, Xishuo Wei, Zhihong Lin <br>
> 29th IAEA Fusion Energy Conference (FEC 2023) <br>
> arXiv:2409.03112 [physics.comp-ph] <br>
> https://doi.org/10.48550/arXiv.2409.03112 <br>

## Background

A [tokamak](https://en.wikipedia.org/wiki/Tokamak) confines plasma in a torus using a magnetic field. The plasma's behavior is described by quantities like the electrostatic potential `phi`.

A companion program, `splineModel.py`, processes output from the Gyrokinetic Tokamak Simulation (GTS) and produces a VTK model that visualizes `phi` on an unstructured grid of triangular-prism wedges. That representation works but has two drawbacks: unstructured grids render slowly, and many advanced rendering techniques require a uniform grid.

`regrid.py` converts the unstructured wedge grid into a structured voxel grid ("volume pixels"), then optionally exports to OpenVDB.

### Reading

The input is a binary unstructured-grid VTK file (binary is far faster and smaller than ASCII). Using `vtkUnstructuredGridReader`, the program extracts:

- **Points**, each with coordinates and a `phi` value, stored as `(x, y, z, p)`. A full torus contains roughly 7.7 million points.
- **Wedges**, the grid cells. Each is a 6-vertex prism with 6 associated `phi` values.

### Voxel grid

The voxel grid is a PyVista `StructuredGrid` whose dimensions match the bounding box of the input. Since `StructuredGrid` requires 1D arrays, all attached data is flattened with NumPy.

### Processing: wedges to voxels

The program resamples the irregular wedge grid onto a uniform voxel grid. This involves a few approximations worth stating clearly.

For each wedge:

1. The two triangular faces are each collapsed to a single centroid sample, with `phi` averaged from the face's three vertices. The field inside the wedge is then treated as varying only along the axis between these two centroids, not across the triangular cross section.
2. The wedge's axis-aligned bounding box is computed, and every voxel inside that bounding box is iterated. (Voxels inside the bounding box but outside the wedge itself still receive a contribution; this is a bounding-box approximation, not exact wedge-voxel overlap.)
3. For each such voxel, the program finds the point `P` on the centroid-to-centroid line closest to the voxel center, computes `phi` at `P` by linear interpolation along that line, and adds it to the voxel.

Because multiple wedges can deposit `phi` into the same voxel, the program accumulates a running total (`phi_sum`) and a write count (`phi_count`), then averages at the end:

```
phi = phi_sum / phi_count   # only where phi_count > 0
```

The `phi` array is initialized to `NaN` so that empty voxels stay distinguishable from voxels with very small values. Most colormaps skip `NaN` automatically.

### OpenVDB conversion

VTK files are dense: they allocate memory for every potential data point. OpenVDB uses a hierarchical structure that only allocates memory for active voxels, which produces large savings. The conversion creates an OpenVDB grid matching the voxel-grid dimensions, copies over every voxel with a non-`NaN` `phi`, and saves.

Note: OpenVDB is natively a C++ library. The Python binding (`pyopenvdb`) is poorly maintained, lacks features, and cannot be installed through `pip` or `conda`. A few quirks remain unresolved. For example, a `FloatGrid` did not render correctly in some viewers while a `LevelSetSphere` did, so the current code builds a sphere grid and overwrites voxel values onto it as a workaround.

### Visualization

| File type | Viewable in |
|-----------|-------------|
| `.vtk`    | VisIt, ParaView |
| `.vdb`    | ParaView, Blender, Houdini |

The `visualize()` function opens an external PyVista window. Comment out the call in `run()` if you don't need it.

## Files

| File | Purpose |
|------|---------|
| `regrid.py`  | The program. |
| `config.ini` | Filename specifications, voxel-grid dimensions, and a flag for whether `phi` values are exact or normalized to `[0, 1]` (added for NVIDIA Omniverse compatibility). |

Input VTK files are not included in this repo due to size (the full torus is 645 MB).

## Results

OpenVDB is dramatically more efficient than VTK:

**Quarter-torus, low resolution:**

| Format | Size | Reduction |
|--------|------|-----------|
| Unstructured VTK | 178 MB | reference |
| Structured VTK   | 3 MB   | 98.31% |
| Structured VDB   | 92 KB  | 99.95% |

**Full torus, medium-high resolution (204 x 55 x 204):**

| Format | Size | Reduction |
|--------|------|-----------|
| Unstructured VTK | 645.3 MB | reference |
| Structured VTK   | 146.5 MB | 77.27% |
| Structured VDB   | 3.8 MB   | 99.44% |
