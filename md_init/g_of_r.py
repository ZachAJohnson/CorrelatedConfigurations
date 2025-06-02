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
        gr_hist = np.zeros(self.nbins, dtype=np.float64)

        # all possible shifts
        neighbor_shifts = np.array([[i, j, k]
                                    for i in (-1, 0, 1)
                                    for j in (-1, 0, 1)
                                    for k in (-1, 0, 1)], dtype=int)

        for cell_idx, indices in self.cells.items():
            cell_idx_arr = np.array(cell_idx)
            pos_cell = self.positions[indices]

            # 1) find all unique neighbor cells (with PBC)
            neighbor_cells = {
                tuple((cell_idx_arr + shift) % self.n_cells)
                for shift in neighbor_shifts
            }

            # 2) loop over each neighbor cell exactly once
            for neighbor_idx in sorted(neighbor_cells):
                # avoid double counting
                if neighbor_idx < cell_idx:
                    continue
                if neighbor_idx not in self.cells:
                    continue

                indices_neighbor = self.cells[neighbor_idx]
                pos_neighbor = self.positions[indices_neighbor]

                if neighbor_idx == cell_idx:
                    # same cell: only i<j
                    if len(pos_cell) < 2:
                        continue
                    diff = pos_cell[:, None, :] - pos_cell[None, :, :]
                    diff = periodic_distance(diff, self.L)
                    dists = np.linalg.norm(diff, axis=-1)
                    iu = np.triu_indices(len(pos_cell), k=1)
                    dists = dists[iu]
                else:
                    # different cells: all pairs
                    diff = pos_cell[:, None, :] - pos_neighbor[None, :, :]
                    diff = periodic_distance(diff, self.L)
                    dists = np.linalg.norm(diff, axis=-1).ravel()

                # bin them
                hist, _ = np.histogram(dists,
                                       bins=self.nbins,
                                       range=(0, self.r_max))
                gr_hist += hist

        # normalization (unchanged)
        r_edges = np.linspace(0, self.r_max, self.nbins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_volumes = 4 * np.pi * r_centers**2 * self.dr
        gr = (2.0 * gr_hist) / (self.N * self.density * shell_volumes)
        return r_centers, gr


import numpy as np


def periodic_distance(vec, L):
    """
    Compute the minimum image distance for a vector difference in a periodic box of side L.
    This function is fully vectorized.
    """
    return vec - L * np.round(vec / L)


class MultiSpeciesGofrCalculator:
    """
    Multi-species radial distribution function calculator.
    
    Parameters
    ----------
    species_ids : array-like of int, shape (N,)
        Integer species identifier for each particle.
    positions : array-like of float, shape (N, 3)
        Particle positions in a cubic box of side length L.
    L : float
        Side length of the cubic periodic box.
    r_max : float
        Maximum distance for g(r) (<= L/2).
    dr : float
        Bin width for the histogram.
    """
    def __init__(self, species_ids, positions, L, r_max, dr):
        self.species_ids = np.asarray(species_ids, dtype=int)
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.shape[0] != self.species_ids.shape[0]:
            raise ValueError("species_ids and positions must have the same length.")

        self.L = float(L)
        self.r_max = float(r_max)
        self.dr = float(dr)
        self.nbins = int(np.ceil(self.r_max / self.dr))

        # Volume and species counts
        self.V = self.L**3
        self.unique_species = np.unique(self.species_ids)
        self.N_species = len(self.unique_species)
        # Map species ID to matrix index
        self.species_to_index = {s: i for i, s in enumerate(self.unique_species)}
        # Count particles per species
        self.counts = np.array([np.sum(self.species_ids == s) for s in self.unique_species], dtype=int)

        # Build cell list for O(N) neighbor search
        self.cell_size = self.r_max
        self.n_cells = int(np.floor(self.L / self.cell_size))
        if self.n_cells < 1:
            self.n_cells = 1
            self.cell_size = self.L
        self.cells = self._build_cell_list()

    def _build_cell_list(self):
        """
        Assign particles to subcells for efficient neighbor lookup.
        """
        cells = {}
        for idx, pos in enumerate(self.positions):
            ci = tuple(np.floor(pos / self.cell_size).astype(int) % self.n_cells)
            cells.setdefault(ci, []).append(idx)
        return cells

    def compute(self):
        """
        Compute the multi-species g(r).

        Returns
        -------
        r_centers : ndarray, shape (nbins,)
            Radii at the center of each bin.
        gr_matrix : ndarray, shape (N_species, N_species, nbins)
            g_ij(r) for each species pair (i,j).
        """
        # Initialize histograms
        hist = np.zeros((self.N_species, self.N_species, self.nbins), dtype=np.float64)

        # All possible neighbor shifts (including PBC)
        neighbor_shifts = np.array([[i, j, k]
                                    for i in (-1, 0, 1)
                                    for j in (-1, 0, 1)
                                    for k in (-1, 0, 1)], dtype=int)

        # Loop over each cell and its neighbors
        for cell_idx, indices in self.cells.items():
            cell_arr = np.array(cell_idx)
            pos_cell = self.positions[indices]
            sp_cell = np.array([self.species_to_index[s] for s in self.species_ids[indices]], dtype=int)

            # Determine neighbor cells with periodic wrap
            neighbors = {tuple((cell_arr + shift) % self.n_cells)
                         for shift in neighbor_shifts}

            for nbr in sorted(neighbors):
                # Avoid double counting
                if nbr < cell_idx or nbr not in self.cells:
                    continue

                idx_nbr = self.cells[nbr]
                pos_nbr = self.positions[idx_nbr]
                sp_nbr = np.array([self.species_to_index[s] for s in self.species_ids[idx_nbr]], dtype=int)

                if nbr == cell_idx:
                    # Same cell: only i < j
                    m = len(indices)
                    if m < 2:
                        continue
                    diff = pos_cell[:, None, :] - pos_cell[None, :, :]
                    diff = periodic_distance(diff, self.L)
                    dists = np.linalg.norm(diff, axis=-1)
                    iu = np.triu_indices(m, k=1)
                    dists = dists[iu]
                    si = sp_cell[iu[0]]
                    sj = sp_cell[iu[1]]
                else:
                    # Different cells: all pairs
                    m = len(indices)
                    n = len(idx_nbr)
                    diff = pos_cell[:, None, :] - pos_nbr[None, :, :]
                    diff = periodic_distance(diff, self.L)
                    dists = np.linalg.norm(diff, axis=-1).ravel()
                    si = np.repeat(sp_cell, n)
                    sj = np.tile(sp_nbr, m)

                # Bin distances
                bins = np.floor(dists / self.dr).astype(int)
                valid = bins < self.nbins
                bins = bins[valid]
                si = si[valid]
                sj = sj[valid]

                # Accumulate counts (and symmetrize for distinct species)
                for a, b, bi in zip(si, sj, bins):
                    hist[a, b, bi] += 1
                    if a != b:
                        hist[b, a, bi] += 1

        # Prepare distance bins
        r_edges = np.linspace(0, self.r_max, self.nbins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        shell_vol = 4.0 * np.pi * r_centers**2 * self.dr

        # Normalize to obtain g(r)
        gr = np.zeros_like(hist)
        for i in range(self.N_species):
            Ni = self.counts[i]
            rhoi = Ni / self.V
            for j in range(self.N_species):
                Nj = self.counts[j]
                rhoj = Nj / self.V
                if i == j:
                    # Autocorrelation: unordered pairs
                    denom = Ni * rhoi * shell_vol
                    gr[i, i, :] = 2.0 * hist[i, i, :] / denom
                else:
                    # Cross-correlation: unordered a-b pairs
                    denom = Ni * rhoj * shell_vol
                    gr[i, j, :] = hist[i, j, :] / denom

        return r_centers, gr
