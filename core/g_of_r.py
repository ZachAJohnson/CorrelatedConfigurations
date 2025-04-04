import numpy as np

def periodic_distance(vec, L):
    """
    Compute the minimum image distance for a vector difference in a periodic box of side L.
    This function is fully vectorized.
    """
    return vec - L * np.round(vec / L)

class GofrCalculator:
    def __init__(self, positions, L, r_max, dr):
        """
        Parameters:
          positions : numpy.ndarray, shape (N, 3)
              Array of particle positions.
          L : float
              Side length of the (cubic) periodic simulation box.
          r_max : float
              Maximum distance at which g(r) is computed (should be <= L/2).
          dr : float
              Width of the histogram bins.
        """
        self.positions = positions
        self.L = L
        self.r_max = r_max
        self.dr = dr
        self.N = positions.shape[0]
        self.V = L**3
        self.density = self.N / self.V
        self.nbins = int(np.ceil(r_max / dr))
        
        # Set the cell (subcell) size to be the cutoff distance.
        # This way, only adjacent cells need to be checked.
        self.cell_size = r_max  
        self.n_cells = int(np.floor(L / self.cell_size))
        if self.n_cells < 1:
            self.n_cells = 1
            self.cell_size = L
        
        self.cells = self.build_cell_list()
    
    def build_cell_list(self):
        """
        Build a dictionary (cell list) mapping cell index (a tuple) to a list of particle indices.
        Particle positions are assigned to cells based on their coordinates.
        """
        cells = {}
        for i, pos in enumerate(self.positions):
            # Compute cell index using floor division (with periodic wrap-around)
            cell_idx = tuple(np.floor(pos / self.cell_size).astype(int) % self.n_cells)
            cells.setdefault(cell_idx, []).append(i)
        return cells
    
    def compute(self):
        """
        Compute the radial distribution function g(r) using the cell-list approach.
        
        Returns:
          r_centers : numpy.ndarray
              The center of each radial bin.
          gr : numpy.ndarray
              The computed radial distribution function.
        
        The method loops over all cells and, for each cell, over its adjacent (neighboring)
        cells. To avoid double counting, only cell pairs (i,j) with neighbor cell index 
        lexicographically >= current cell index are considered. For particles within the same cell,
        only pairs with i < j are used.
        """
        gr_hist = np.zeros(self.nbins, dtype=np.float64)
        
        # Define all 27 neighbor shifts in 3D.
        neighbor_shifts = np.array([[i, j, k] 
                                    for i in (-1, 0, 1)
                                    for j in (-1, 0, 1)
                                    for k in (-1, 0, 1)], dtype=int)
        
        # Loop over each cell in the cell list.
        for cell_idx, indices in self.cells.items():
            cell_idx_arr = np.array(cell_idx)
            pos_cell = self.positions[indices]  # positions in current cell
            
            # Loop over neighbor cells
            for shift in neighbor_shifts:
                # Determine neighbor cell index (with periodic wrap-around)
                neighbor_idx = tuple((cell_idx_arr + shift) % self.n_cells)
                # Avoid double counting by processing only if neighbor cell >= current cell (in lexicographic order)
                if neighbor_idx < cell_idx:
                    continue
                if neighbor_idx not in self.cells:
                    continue
                indices_neighbor = self.cells[neighbor_idx]
                pos_neighbor = self.positions[indices_neighbor]
                
                if neighbor_idx == cell_idx:
                    # Same cell: compute only pairs with i < j
                    if len(pos_cell) < 2:
                        continue
                    # Use vectorized broadcasting for particles in the same cell
                    diff = pos_cell[:, np.newaxis, :] - pos_cell[np.newaxis, :, :]
                    diff = periodic_distance(diff, self.L)
                    dists = np.linalg.norm(diff, axis=-1)
                    # Extract upper triangle (i<j) to avoid double counting
                    iu = np.triu_indices(len(pos_cell), k=1)
                    dists = dists[iu]
                else:
                    # Different cells: all pairs are unique
                    diff = pos_cell[:, np.newaxis, :] - pos_neighbor[np.newaxis, :, :]
                    diff = periodic_distance(diff, self.L)
                    dists = np.linalg.norm(diff, axis=-1).ravel()
                
                # Bin the distances for these pairs
                hist, _ = np.histogram(dists, bins=self.nbins, range=(0, self.r_max))
                gr_hist += hist
        
        # Normalize the histogram.
        # For a homogeneous system, the expected number of pairs in a shell of thickness dr at radius r is:
        # (N * density * 4 * π * r² dr)/2   [the factor 1/2 accounts for counting each pair once]
        # Thus, the radial distribution function is given by:
        # g(r) = (2 * gr_hist) / (N * density * shell_volume)
        r_edges = np.linspace(0, self.r_max, self.nbins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_volumes = 4 * np.pi * r_centers**2 * self.dr
        gr = (2.0 * gr_hist) / (self.N * self.density * shell_volumes)
        return r_centers, gr
