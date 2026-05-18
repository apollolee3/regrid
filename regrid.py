import numpy as np
import pyvista as pv
import pyopenvdb as vdb
import vtk
import math
import configparser
import os
from tqdm import tqdm
import cProfile
import pstats
from io import StringIO

# Sections:
# 1.  Init
# 2.  Reading
# 3.  Voxel Grid Components
# 4.  Wedge Calculations
# 5.  Voxel Calculations
# 6.  Processing
# 7.  Voxel Grid Generation
# 8.  OpenVDB Conversion
# 9.  PyVista Visualization (Optional)
# 10. Run
# 11. Main

class Wedge:

    def __init__(self, vertices, phi_values):
        # A wedge must have 6 vertices
        if len(vertices) != 6 or len(phi_values) != 6:
            raise ValueError(f"ERROR: Got {len(vertices)} vertices and {len(phi_values)} phi values.")

        self.vertices = vertices
        self.phi_values = phi_values

    def __str__(self):
        return f"Vertices: {self.vertices}\nPhi Values: {self.phi_values}\n"

class Regrid:

    # ***************************** UNIQUE FILENAME *****************************

    @staticmethod
    def filename(base_filename):
        """
        Returns a unique filename by appending a number to avoid overwriting existing files.
        """
        dir_name, file_name = os.path.split(base_filename)
        file_root, file_ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(base_filename):
            base_filename = os.path.join(dir_name, f"{file_root} ({counter}){file_ext}")
            counter += 1

        return base_filename

    # ***************************** INIT *****************************

    def __init__(self, config_filename='config.ini'):
        '''
        Initializes the necessary variables.
        '''
        config = configparser.ConfigParser()
        config.read(config_filename)

        self.input_filename = config.get('DEFAULT', 'input_filename')
        self.output_filename_vtk = Regrid.filename(config.get('DEFAULT', 'output_filename_vtk'))
        self.output_filename_vdb = Regrid.filename(config.get('DEFAULT', 'output_filename_vdb'))

        self.x_points = config.getint('DEFAULT', 'X')
        self.y_points = config.getint('DEFAULT', 'Y')
        self.z_points = config.getint('DEFAULT', 'Z')

        self.vdb_normalization = config.getint('DEFAULT', 'vdb_normalization')

        self.phi = None
        self.phi_sum = None
        self.phi_count = None
        self.contains_wedge = None
        self.phi_min = None
        self.phi_max = None
        self.phi_normalized = None

        self.coordinates = None
        self.wedges = []

        self.voxel_grid = None
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self.z_min, self.z_max = None, None

    # ***************************** READING *****************************

    def read(self):
        """
        Reads the input .vtk file, extracting coordinate and wedge data and sorting them into lists.
        """
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(self.input_filename)
        reader.Update()
        unstructured_grid = reader.GetOutput()
        points = unstructured_grid.GetPoints()
        phi_array = unstructured_grid.GetPointData().GetArray("Phi")
        num_points = points.GetNumberOfPoints()
        assert num_points == phi_array.GetNumberOfTuples()

        # Extract coordinates and phi values directly into numpy array
        self.coordinates = np.empty((num_points, 4))
        for i in range(num_points):
            point = points.GetPoint(i)
            phi_value = phi_array.GetTuple1(i)
            self.coordinates[i] = [*point, phi_value]

        # Extract cells (wedges) from the .vtk file and append to the "wedges" list
        for i in range(unstructured_grid.GetNumberOfCells()):
            cell = unstructured_grid.GetCell(i)
            # Vertex indices
            point_indices = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            # Vertex coordinates
            vertices = [points.GetPoint(index) for index in point_indices]
            phi_values = [phi_array.GetTuple1(index) for index in point_indices]
            wedge = Wedge(vertices, phi_values)
            self.wedges.append(wedge)

    # ***************************** VOXEL GRID COMPONENTS *****************************

    def get_voxel_grid_bounds(self):
        """
        Gets the XYZ bounds of the 3D voxel grid by finding the min/max values in the input .vtk coordinates array.
        """
        self.x_min, self.x_max = self.coordinates[:, 0].min(), self.coordinates[:, 0].max()
        self.y_min, self.y_max = self.coordinates[:, 1].min(), self.coordinates[:, 1].max()
        self.z_min, self.z_max = self.coordinates[:, 2].min(), self.coordinates[:, 2].max()

        # Avoid getting "-0.00" as a bound
        def formatted(val):
            formatted_value = f"{val:.2f}"
            return formatted_value if formatted_value != "-0.00" else "0.00"

        print("\nVoxel Grid Bounds:")
        print(f"X: ({formatted(self.x_min)}, {formatted(self.x_max)})")
        print(f"Y: ({formatted(self.y_min)}, {formatted(self.y_max)})")
        print(f"Z: ({formatted(self.z_min)}, {formatted(self.z_max)})\n")

    def get_voxel_grid_arrays(self):
        """
        Initializes the 3D numpy arrays that will be added to the 3D voxel grid.
        phi_sum: The total "phi" in 1 voxel.
        phi_count: The total amount of times that "phi" was added to 1 voxel.
        phi: The weighted average of phi in 1 voxel, obtained by dividing phi_sum by phi_count.
        phi_normalized: The "phi" array, except on a normalized magnitude range from 0.0 to 1.0.
        contains_wedge: Checks whether a voxel contains a wedge in any capacity. Mostly for debugging purposes.
        """
        grid_dims = (self.x_points, self.y_points, self.z_points)
        self.phi_sum = np.zeros(grid_dims, dtype=np.float64).flatten()
        self.phi_count = np.zeros(grid_dims, dtype=np.float64).flatten()
        self.phi = np.full(grid_dims, np.nan, dtype=np.float64).flatten()
        self.phi_normalized = np.full(grid_dims, np.nan, dtype=np.float64).flatten()
        self.contains_wedge = np.zeros(grid_dims, dtype=int).flatten()

    # ***************************** WEDGE CALCULATIONS *****************************

    def get_bounding_box(self, wedge):
        """
        For 1 wedge, gets the geometric bounding box (rectangular prism).
        """
        vertices_array = np.array(wedge.vertices)
        x_min, x_max = vertices_array[:, 0].min(), vertices_array[:, 0].max()
        y_min, y_max = vertices_array[:, 1].min(), vertices_array[:, 1].max()
        z_min, z_max = vertices_array[:, 2].min(), vertices_array[:, 2].max()

        return {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}

    def get_centroids(self, wedge):
        """
        For 1 wedge, computes its 2 centroids.
        """
        triangle1 = wedge.vertices[:3]
        triangle2 = wedge.vertices[3:]
        centroid1_coords = [sum(coord) / 3 for coord in zip(*triangle1)]
        centroid2_coords = [sum(coord) / 3 for coord in zip(*triangle2)]

        return centroid1_coords, centroid2_coords

    def get_centroid_phi_values(self, wedge):
        """
        For 1 wedge, gets the phi values of the 2 centroids.
        """
        phi1 = wedge.phi_values[:3]
        phi2 = wedge.phi_values[3:]
        centroid1_phi = sum(phi1) / 3
        centroid2_phi = sum(phi2) / 3

        return centroid1_phi, centroid2_phi

    def get_voxels(self, wedge):
        """
        For 1 wedge, gets all the voxels that it touches, in any capacity.
        """
        bounding_box = self.get_bounding_box(wedge)

        # Calculates the size of each voxel in the respective direction
        x_res = (self.x_max - self.x_min) / self.x_points
        y_res = (self.y_max - self.y_min) / self.y_points
        z_res = (self.z_max - self.z_min) / self.z_points

        x_start = int((bounding_box['x'][0] - self.x_min) / x_res)
        x_end = math.ceil((bounding_box['x'][1] - self.x_min) / x_res)
        y_start = int((bounding_box['y'][0] - self.y_min) / y_res)
        y_end = math.ceil((bounding_box['y'][1] - self.y_min) / y_res)
        z_start = int((bounding_box['z'][0] - self.z_min) / z_res)
        z_end = math.ceil((bounding_box['z'][1] - self.z_min) / z_res)

        return x_start, x_end, y_start, y_end, z_start, z_end

    # ***************************** VOXEL CALCULATIONS *****************************

    def get_center_line_point(self, voxel_center, centroid1, centroid2):
        """
        For 1 voxel, computes a point on the line segment between two centroids that is closest to the voxel center.
        """
        P0 = np.array(centroid1)
        P1 = np.array(centroid2)
        V = np.array(voxel_center)
        direction = P1 - P0
        direction_norm_sq = np.dot(direction, direction)
        t = np.dot(V - P0, direction) / direction_norm_sq
        t = np.clip(t, 0, 1)

        return P0 + t * direction

    def get_interpolated_phi(self, closest_point, centroid1_coords, centroid1_phi, centroid2_coords, centroid2_phi):
        """
        For 1 voxel, determines the interpolated value of phi at a point based on its proximity to two centroids,
        weighting the contribution of each centroid's phi value by the inverse of its distance to the point.
        """
        dist_to_c1 = np.linalg.norm(np.array(centroid1_coords) - np.array(closest_point))
        dist_to_c2 = np.linalg.norm(np.array(centroid2_coords) - np.array(closest_point))
        total_dist = dist_to_c1 + dist_to_c2
        ratio_to_c1 = dist_to_c1 / total_dist
        interpolated_phi = ratio_to_c1 * centroid1_phi + (1 - ratio_to_c1) * centroid2_phi

        return interpolated_phi

    # ***************************** PROCESSING *****************************

    def process(self):
        """
        The heart of the program. Loops through each wedge, calculates which voxels the wedge touches in 3D space,
        and loops through those voxels. The exact contribution of phi to each voxel is calculated, then added.
        """
        x_spacing = (self.x_max - self.x_min) / (self.x_points - 1)
        y_spacing = (self.y_max - self.y_min) / (self.y_points - 1)
        z_spacing = (self.z_max - self.z_min) / (self.z_points - 1)

        # Loop through each wedge
        b = "\033[97m{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed} Remaining: {remaining}]"
        for wedge in tqdm(self.wedges, desc="Progress", bar_format=b, mininterval=1):
            centroid1_coords, centroid2_coords = self.get_centroids(wedge)
            centroid1_phi, centroid2_phi = self.get_centroid_phi_values(wedge)
            x_start, x_end, y_start, y_end, z_start, z_end = self.get_voxels(wedge)

            # Loop through the voxels that the wedge touches
            for i in range(x_start, x_end):
                for j in range(y_start, y_end):
                    for k in range(z_start, z_end):
                        voxel_center = [(i + 0.5) * x_spacing + self.x_min,
                                        (j + 0.5) * y_spacing + self.y_min,
                                        (k + 0.5) * z_spacing + self.z_min]

                        closest_point = self.get_center_line_point(voxel_center, centroid1_coords, centroid2_coords)

                        interpolated_phi = self.get_interpolated_phi(
                            closest_point, centroid1_coords, centroid1_phi, centroid2_coords, centroid2_phi)

                        index = (k * (self.x_points * self.y_points) + j * self.x_points + i)

                        '''
                        Index Calculation for a Flattened 3D Array
                        1) For every 1 step in Z, you go over 1 XY grid.
                        2) For every 1 step in Y, you go over 1 X row.
                        3) For every 1 step in X, you go over 1 voxel.
                        '''

                        self.phi_sum[index] += interpolated_phi
                        self.phi_count[index] += 1
                        self.contains_wedge[index] = 1

        # Avoid division by zero
        valid_indices = self.phi_count != 0

        # The "phi" array is the weighted average of phi in each voxel
        self.phi[valid_indices] = self.phi_sum[valid_indices] / self.phi_count[valid_indices]

        # Normalized magnitude range from 0.0 to 1.0, for optional use
        self.phi_min = np.nanmin(self.phi)
        self.phi_max = np.nanmax(self.phi)

        # Avoid division by zero
        if self.phi_max != self.phi_min:
            self.phi_normalized = (self.phi - self.phi_min) / (self.phi_max - self.phi_min)
        else:
            self.phi_normalized = np.full_like(self.phi, 0.0)

    # ***************************** VOXEL GRID GENERATION *****************************

    def get_voxel_grid(self):
        """
        Generates the 3D voxel grid based on the accumulated data.
        """
        X, Y, Z = np.mgrid[self.x_min:self.x_max:self.x_points * 1j,
                           self.y_min:self.y_max:self.y_points * 1j,
                           self.z_min:self.z_max:self.z_points * 1j]
        grid = pv.StructuredGrid(X, Y, Z)

        grid["Phi"] = self.phi
        grid["Phi Normalized"] = self.phi_normalized
        grid["Phi Sum"] = self.phi_sum
        grid["Phi Count"] = self.phi_count
        grid["Contains Wedge"] = self.contains_wedge

        return grid

    # ***************************** OPENVDB CONVERSION **************************

    def vdb(self, voxel_grid):
        """
        Converts the voxel grid to an OpenVDB sparse grid. In a sparse grid, we consider a voxel to be "active"
        if it is allocated and involved in the computation. The rest of the grid simply becomes inactive
        and contains no data. This significantly reduces the file size, often by as much as 99%.
        """
        global_min_bounds = voxel_grid.bounds[::2]
        global_max_bounds = voxel_grid.bounds[1::2]
        global_center = 0.5 * (np.array(global_max_bounds) + np.array(global_min_bounds))
        global_center_tuple = tuple(global_center)

        # Compute all 8 corners of the bounding box of the voxel grid
        corners = [
            np.array([global_min_bounds[0], global_min_bounds[1], global_min_bounds[2]]),
            np.array([global_min_bounds[0], global_min_bounds[1], global_max_bounds[2]]),
            np.array([global_min_bounds[0], global_max_bounds[1], global_min_bounds[2]]),
            np.array([global_min_bounds[0], global_max_bounds[1], global_max_bounds[2]]),
            np.array([global_max_bounds[0], global_min_bounds[1], global_min_bounds[2]]),
            np.array([global_max_bounds[0], global_min_bounds[1], global_max_bounds[2]]),
            np.array([global_max_bounds[0], global_max_bounds[1], global_min_bounds[2]]),
            np.array([global_max_bounds[0], global_max_bounds[1], global_max_bounds[2]])
        ]

        # Find the largest distance from the center to any corner
        radius = max(np.linalg.norm(corner - global_center) for corner in corners)

        ''' 
        For some unknown reason, FloatGrid doesn't display properly in Houdini/ParaView/Omniverse, while 
        LevelSetSphere does. Unfortunately, with the limitations of pyopenvdb compared to the C++ version, 
        the background value (3 in this case) cannot be changed. Thus, when viewing the .vdb file in any 
        visualization software, you must apply a threshold to eliminate the background value and adjust the 
        color map to see the phi values.
        '''

        grid = vdb.createLevelSetSphere(radius=radius, center=global_center_tuple)
        grid.name = "density"
        accessor = grid.getAccessor()
        min_bounds = np.full(3, np.inf)
        max_bounds = np.full(3, -np.inf)

        for i in range(self.x_points):
            for j in range(self.y_points):
                for k in range(self.z_points):
                    index = (k * (self.x_points * self.y_points) + j * self.x_points + i)
                    coord = (i, j, k)
                    if self.vdb_normalization == 1:
                        value = self.phi_normalized[index]
                    else:
                        value = self.phi[index]
                    if not np.isnan(value):
                        accessor.setValueOn(coord, value)
                        min_bounds = np.minimum(min_bounds, [i, j, k])
                        max_bounds = np.maximum(max_bounds, [i, j, k])

        metadata_dict = {
            "global_center": list(global_center),
            "global_max_bounds": list(global_max_bounds),
            "global_min_bounds": list(global_min_bounds),
            "min_bounds": list(min_bounds),
            "max_bounds": list(max_bounds),
            "name": "density",
            "is_local_space": False,
            "is_saved_as_half_float": False,
            "value_type": "float",
        }

        grid.updateMetadata(metadata_dict)
        vdb.write(self.output_filename_vdb, grids=[grid])

    # ***************************** PYVISTA VISUALIZATION **************************

    def visualize(self, voxel_grid):
        """
        Optionally creates a PyVista window to visualize the voxel grid.
        """
        plotter = pv.Plotter()
        plotter.add_mesh(voxel_grid, scalars="Phi", show_edges=True)
        plotter.show()

    # ***************************** RUN **************************

    def run(self):
        """
        Runs all the functions in the necessary order.
        """
        self.read()
        self.get_voxel_grid_bounds()
        self.get_voxel_grid_arrays()
        self.process()
        self.voxel_grid = self.get_voxel_grid()
        # Save as VTK
        self.voxel_grid.save(self.output_filename_vtk, binary=True)
        # Save as VDB
        self.vdb(self.voxel_grid)
        print("\nSuccessfully saved voxel grid as .vtk and .vdb files!\n")
        print(self.voxel_grid)
        # self.visualize(self.voxel_grid)

# ***************************** MAIN *****************************

if __name__ == "__main__":
    # To enable profiling, uncomment the commented lines below.
    # pr = cProfile.Profile()
    # pr.enable()

    config_file = 'config.ini'
    m = Regrid(config_file)
    m.run()

    # pr.disable()
    # s = StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
