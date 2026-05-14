# regrid

**Author:** Apollo Lee — `apollo1@stanford.edu`

A Python tool for converting unstructured-grid tokamak simulation data into uniform voxel grids, with optional export to OpenVDB for compact, sparse storage.

---

## 1. Purpose

`regrid.py` improves a 3D model of a [tokamak](https://en.wikipedia.org/wiki/Tokamak) — a device that uses a powerful magnetic field to confine plasma in the shape of a torus. The plasma's behavior is described by several quantities, including the electrostatic potential `phi`.

The companion program `splineModel.py` processes output from the **Gyrokinetic Tokamak Simulation (GTS)** and produces a VTK model that visualizes `phi` on an unstructured grid of triangular-prism wedges.

That representation works, but it isn't ideal:

- Unstructured grids render slowly.
- Many advanced rendering techniques require a uniform grid.

A uniform 3D grid is made of **voxels** ("volume pixels"). `regrid.py` converts the unstructured wedge grid into a structured voxel grid while strictly preserving data integrity and accurate spatial placement, then optionally exports the result to **OpenVDB** for massive file-size savings.

---

## 2. How it works

The program is organized into 11 sections:

1. Init
2. Reading
3. Voxel Grid Components
4. Wedge Calculations
5. Voxel Calculations
6. Processing
7. Voxel Grid Generation
8. OpenVDB Conversion
9. PyVista Visualization (optional)
10. Run
11. Main

### Reading

The input is an unstructured-grid VTK file (binary — far faster and smaller than ASCII). Using `vtkUnstructuredGridReader`, we extract:

- **Points** — each with coordinates and a `phi` value, stored as `(x, y, z, p)`. A full torus contains ~7.7 million points.
- **Wedges** — the cells. Each is a 6-vertex prism with 6 associated `phi` values.

### Voxel grid

The voxel grid is a PyVista `StructuredGrid` whose dimensions match the bounding box of the input. Because `StructuredGrid` requires 1D arrays, all attached data is flattened with NumPy.

### Processing — wedges → voxels

Converting from the irregular wedge grid to a uniform voxel grid requires care:

- Wedges and voxels are *different sizes*. A single wedge may contain hundreds of voxels — or vice versa.
- A wedge can lie partially inside a voxel.
- The unstructured grid's shape is irregular and not described by any equation.
- We need to distinguish voxels with very small `phi` values from voxels with no data at all.
- The full torus has over 15 million wedges, so efficiency matters.

For each wedge:

1. Find the centroid of each of its two triangular faces and the `phi` value at each centroid (averaging the three vertices).
2. Determine which voxels the wedge touches.
3. For each touched voxel:
   - Connect a line between the two centroids.
   - Find the point on that line closest to the voxel's center (call it `P`).
   - Compute `phi` at `P` by linear interpolation along the centroid-to-centroid line.
   - Add `phi(P)` to the voxel.

Because multiple wedges can deposit `phi` into the same voxel, we accumulate:

- `phi_sum` — running total of `phi` deposited into each voxel.
- `phi_count` — how many times each voxel was written to.

At the end:

```
phi = phi_sum / phi_count   # only where phi_count > 0
```

The `phi` array is initialized to **NaN** so that empty voxels are clearly distinguishable from voxels with very small values (NaN is skipped by colormaps in visualization tools).

### OpenVDB conversion

VTK files allocate memory for *every* potential data point — they're "dense." OpenVDB, by contrast, uses a hierarchical structure that only allocates memory for *active* voxels — a "sparse" grid — yielding dramatic memory savings.

The conversion is straightforward:

1. Create a blank OpenVDB grid matching the voxel-grid dimensions.
2. For each voxel with a non-NaN `phi`, copy the value into the OpenVDB grid.
3. Save.

> **Note:** OpenVDB is natively a C++ library. The Python binding (`pyopenvdb`) is poorly maintained, lacks features, and **cannot be installed via `pip` or `conda`**. There are also some unresolved quirks — for instance, a `FloatGrid` did not display correctly in some viewers, while a `LevelSetSphere` did.

### Visualization

| File type | Viewable in |
|-----------|-------------|
| `.vtk`    | VisIt, ParaView |
| `.vdb`    | ParaView, Blender, Houdini |

The `visualize()` function opens an external PyVista window. Comment out the call in `run()` if you don't need it.

---

## 3. Usage

On `stellar-vis2`, the project lives at `/home/al3926/regrid`.

Activate the conda environment (contains all required packages):

```bash
module load anaconda3/2023.3
conda activate py36
```

Run the program:

```bash
python3 regrid.py
```

### Directories

| Path | Purpose |
|------|---------|
| `input/`   | Unstructured-grid VTK files (full torus, half, quarter, thin disk). *Not included in this repo due to size — the full torus is 645 MB.* |
| `output/`  | Output VTK and VDB files. |
| `archive/` | Past and unused files. |

### Files

| File | Purpose |
|------|---------|
| `regrid.py`  | The program. |
| `config.ini` | Filename specifications, voxel-grid dimensions, and a flag for whether `phi` values are exact or normalized to `[0, 1]` (added for NVIDIA Omniverse compatibility). |

---

## 4. Results

The program correctly transforms data from the unstructured grid to the structured grid while preserving data integrity and spatial accuracy.

OpenVDB is dramatically more efficient than VTK:

**Quarter-torus, low resolution:**

| Format | Size | Reduction |
|--------|------|-----------|
| Unstructured VTK | 178 MB | — |
| Structured VTK   | 3 MB   | 98.31% |
| Structured VDB   | 92 KB  | 99.95% |

**Full torus, medium-high resolution (204 × 55 × 204):**

| Format | Size | Reduction |
|--------|------|-----------|
| Unstructured VTK | 645.3 MB | — |
| Structured VTK   | 146.5 MB | 77.27% |
| Structured VDB   | 3.8 MB   | 99.44% |

### Future work

The biggest area for improvement is **speed**. Runtime scales linearly with the number of wedges:

- Disk (high resolution): a couple of minutes.
- Full torus (high resolution): days.

`process()` dominates the total runtime and is highly parallelizable — distributing the work across GPU cores would be a strong next step.
