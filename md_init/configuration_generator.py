import numpy as np
import random

class PeriodicConfigurationGenerator_MS:
    def __init__(self, L_cell, gofr_funcs, N_particles_per_species_per_subcell, N_subcells_per_dim,
                 r_correlation=3, dx_over_a=None, perturb=True,
                 δ_perturb=None, min_distance_δ=None):
        '''
        Multi-species configuration generator in a periodic box.

        Parameters
        ----------
        L_cell : float
            Length of the full cell in each dimension.
        gofr_funcs : list of list of callables
            gofr_funcs[i][j](r) returns g_{ij}(r) for distance r.
        N_particles_per_species_per_subcell : list of int
            Number of particles of each species to place in each subcell.
        N_subcells_per_dim : int
            Number of subcells along each dimension.
        r_correlation : float, optional
            Correlation length (unused in this class but kept for compatibility).
        dx_over_a : float, optional
            Mesh spacing fraction; if None, defaults to 0.1.
        perturb : bool, optional
            Whether to apply a small random perturbation after placement.
        δ_perturb : float, optional
            Standard deviation of perturbation.
        min_distance_δ : float, optional
            Minimum allowed displacement per axis after perturbation.
        '''
        self.L_cell = L_cell
        self.gofr_funcs = gofr_funcs
        self.N_species = len(N_particles_per_species_per_subcell)
        self.N_particles_per_species_per_subcell = N_particles_per_species_per_subcell
        # total per subcell and full cell
        self.N_particles_per_subcell = sum(N_particles_per_species_per_subcell)
        self.N_subcells_per_dim = N_subcells_per_dim
        self.N_subcells = N_subcells_per_dim**3
        self.N_particles = self.N_particles_per_subcell * self.N_subcells
        self.L_subcell = L_cell / N_subcells_per_dim
        self.perturb = perturb
        self.δ_perturb = δ_perturb
        self.min_distance_δ = min_distance_δ

        # set mesh spacing
        if dx_over_a is None:
            dx_over_a = 0.1
        a = self.L_subcell / (self.N_particles_per_subcell)**(1/3)
        self.dx = dx_over_a * a
        self.create_mesh()

        # build and shuffle species placement sequence
        self.subcell_species = []
        for s, count in enumerate(self.N_particles_per_species_per_subcell):
            self.subcell_species += [s] * count
        random.shuffle(self.subcell_species) 

        print(f'Creating {self.N_species} species; subcell length {self.L_subcell}. Total particles/subcell: {self.N_particles_per_subcell}')

    def periodic_distance(self, x1, x2):
        x1, x2 = np.array(x1), np.array(x2)
        diff = np.abs(x1 - x2) - self.L_subcell * np.round(np.abs(x1 - x2) / self.L_subcell)
        r = np.linalg.norm(diff, axis=0)
        return r

    def create_mesh(self):
        # determine grid resolution
        self.Nx = int(self.L_subcell / self.dx)
        self.x = np.linspace(0, self.L_subcell, self.Nx, endpoint=False)
        self.y = np.linspace(0, self.L_subcell, self.Nx, endpoint=False)
        self.z = np.linspace(0, self.L_subcell, self.Nx, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        # flattened list of all mesh points
        self.XYZ_list = np.vstack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()]).T
        # one product mesh per species
        self.G = np.ones((self.N_species, self.Nx, self.Nx, self.Nx))

    def update_G_from_position(self, ion_position, species_id):
        # compute distances to all grid points
        r_mesh = self.periodic_distance(ion_position[:, None, None, None], np.array([self.X, self.Y, self.Z]))
        # update each species' product mesh
        for t in range(self.N_species):
            gmesh = self.gofr_funcs[t][species_id](r_mesh)
            self.G[t] *= gmesh

    def get_random_new_position(self, species_id):
        rng = np.random.default_rng()
        flat = self.G[species_id].ravel()
        p = flat / flat.sum()
        pos = rng.choice(self.XYZ_list, p=p)
        return pos

    def fill_subcell_with_particles(self):
        self.subcell_positions = []  # list of (pos, species_id)
        idx = 0
        while idx < len(self.subcell_species):
            s = self.subcell_species[idx]
            try:
                pos = self.get_random_new_position(s)
                self.update_G_from_position(pos, s)
                self.subcell_positions.append((pos, s))
                idx += 1
            except ValueError:
                print(f'ValueError placing species {s}: retrying')
        # convert to array of objects
        self.subcell_positions = np.array(self.subcell_positions, dtype=object)

    def fill_cell_from_subcell(self):
        # split positions and species
        pos_arr = np.array([p for p, _ in self.subcell_positions])
        sp_arr = np.array([s for _, s in self.subcell_positions])
        # shifts for each subcell location
        shifts = np.arange(self.N_subcells_per_dim) * self.L_subcell
        shift_array = np.vstack(np.meshgrid(shifts, shifts, shifts, indexing='ij')).reshape(3, -1).T
        # tile subcell into full cell
        positions, species = [], []
        for shift in shift_array:
            for p, s in zip(pos_arr, sp_arr):
                positions.append(p + shift)
                species.append(s)
        self.ion_positions = np.array(positions)
        self.ion_species = np.array(species)
        
    def fill_cell_with_particles(self):
        self.fill_subcell_with_particles()
        self.fill_cell_from_subcell()
        if self.perturb:
            self.perturb_particles()

    def perturb_particles(self):
        if self.δ_perturb is None:
            self.δ_perturb = self.dx / 4
        if self.min_distance_δ is None:
            self.min_distance_δ = self.dx / 2
        if self.min_distance_δ > self.dx:
            print('Warning: minimum distance > dx; setting to dx/2')
            self.min_distance_δ = self.dx / 2
        max_δ = (self.dx - self.min_distance_δ) / 2
        δ = np.random.normal(-self.δ_perturb, self.δ_perturb, size=self.ion_positions.shape)
        δ = np.clip(δ, -max_δ, max_δ)
        self.ion_positions += δ
        self.ion_positions = np.clip(self.ion_positions, 0, self.L_cell)

class PeriodicConfigurationGenerator():
    def __init__(self, L_cell, gofr_func, N_particles_per_subcell, N_subcells_per_dim, r_correlation = 3, 
                 dx_over_a = None, perturb=True, δ_perturb = None, min_distance_δ =None):
        """
        ConfigurationGenerator class to create a configuration of particles in a 3D periodic box.
        Parameters
        ----------
        L_cell : float
            Length of the cell in each dimension.
        gofr_func : function
            Function to calculate the g(r) value for a given distance r.
        N_particles_per_subcell : int
            Number of particles to be placed in each subcell.
        N_subcells_per_dim : int
            Number of subcells in each dimension.
        r_correlation : float, optional
            Correlation length for the particles. Default is 3.
        dx_over_a : float, optional
            Distance between mesh points over the average a^3 N = L^3. If None, it is 0.1.
        perturb : bool, optional
            Whether to perturb the particles to remove perfect periodicity. Default is True.
        δ_perturb : float, optional
            Perturbation distance for the particles. Default is None.
        min_distance_δ : float, optional
            Minimum distance between particles after perturbation. Default is None.
        """
        self.L_cell = L_cell
        self.r_correlation = r_correlation
        self.gofr_func = gofr_func
        self.N_particles_per_subcell = N_particles_per_subcell
        self.N_subcells_per_dim = N_subcells_per_dim
        self.N_subcells = self.N_subcells_per_dim**3
        self.N_particles = N_particles_per_subcell * N_subcells_per_dim**3
        self.L_subcell = L_cell/self.N_subcells_per_dim
        self.perturb = perturb
        self.δ_perturb  = δ_perturb
        self.min_distance_δ  = min_distance_δ
        
        # Set discretization
        if dx_over_a is None:
            dx_over_a = 0.1
        a = self.L_subcell/(self.N_particles_per_subcell)**(1/3)
        self.dx = dx_over_a*a    
        self.create_mesh()
        print(f"Creating subcells of length {self.L_subcell}. Distances beyond this have spurious correlations.")
        
    def periodic_distance(self, x1, x2):
        """ 
        Calculate the minimum image distance accounting for periodic boundary conditions.
        Parameters
        ----------
        x1 : array_like
            First set of coordinates.
        x2 : array_like
            Second set of coordinates.
        Returns
        -------
        r : array_like
            Minimum image distance between x1 and x2.
        """
        x1, x2 = np.array(x1), np.array(x2)
        r = np.linalg.norm(np.abs(x1-x2) -self.L_subcell*np.round(np.abs(x1-x2)/self.L_subcell),axis=0)
        return r
    
    def create_mesh(self):
        """
        Create a mesh grid for the entire domain.
        Parameters
        ----------
        dx : float, optional
            Distance between mesh points. If None, it will be calculated based on the number of particles per subcell.
        """
        # Mesh grids for the entire domain (could be adjusted to only create necessary subcell meshes)
        self.Nx = int(self.L_subcell/self.dx)
        self.x = np.linspace(0, self.L_subcell, self.Nx, endpoint=False)
        self.y = np.linspace(0, self.L_subcell, self.Nx, endpoint=False)
        self.z = np.linspace(0, self.L_subcell, self.Nx, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.XYZ_list = np.array([self.X,self.Y,self.Z]).reshape(3, self.Nx**3).T
        self.G = np.ones_like(self.X) # The g(r) product mesh

    def update_G_from_position(self, ion_position):
        """
        Update the g(r) product mesh based on the position of a new ion.
        Parameters
        ----------
        ion_position : array_like
            Position of the new ion.
        """
        ion_position = np.array(ion_position)
        r_mesh = self.periodic_distance(ion_position[:, np.newaxis, np.newaxis, np.newaxis], np.array([self.X,self.Y,self.Z]) )
        g_mesh = self.gofr_func(r_mesh) 
        self.G *= g_mesh
        
    def get_random_new_position(self):
        """
        Get a random new position for an ion based on the g(r) product mesh.
        Returns
        -------
        atom_pos : array_like
            Random position for the new ion.
        """
        rng = np.random.default_rng()
        flat_normalized_probability  = self.G.flatten()/np.sum(self.G)
        atom_pos = rng.choice(self.XYZ_list, p = flat_normalized_probability  )
        return atom_pos
    
    def perturb_particles(self):
        """
        Randomly perturb all particles to remove perfect periodicity.
        """
        if self.δ_perturb is None:
            self.δ_perturb = self.dx/4
        if self.min_distance_δ is None:
            self.min_distance_δ = self.dx/2
        if self.min_distance_δ>self.dx:
            print("Warning, minimum distance set too large (> self.dx). Setting to dx/2.")
            self.min_distance_δ = self.dx/2
        max_δ = (self.dx - self.min_distance_δ)/2
        # Randomly perturb the ion positions
        perturbation = np.random.normal(-self.δ_perturb, self.δ_perturb, size=self.ion_positions.shape)
        perturbation = np.clip(perturbation, -max_δ, max_δ)
        self.ion_positions += perturbation
        self.ion_positions = np.clip(self.ion_positions, 0, self.L_cell)

    def fill_subcell_with_particles(self):
        self.subcell_ion_positions = []
        N_particles_placed = 0
        fails = 0
        max_fail = 10
        while N_particles_placed < self.N_particles_per_subcell:
            try:                 
                ion_position = self.get_random_new_position()
                self.update_G_from_position(ion_position)
                self.subcell_ion_positions.append(ion_position)
                N_particles_placed += 1
                fails = 0
            except ValueError as err:
                fails+=1
                print("ValueError: err. Retrying placement.") 
            if fails>max_fail:
                print("Failed to place particle {fails} times. Breaking.")
                break
        self.subcell_ion_positions = np.array(self.subcell_ion_positions) 
    
    
    def fill_cell_from_subcell(self):
        """
        Fill the cell with particles from the subcell.
        """
        self.ion_positions = np.repeat(self.subcell_ion_positions[None,...], self.N_subcells, axis=0) # repeat subcell positions
        ΔL = np.arange(0, self.N_subcells_per_dim)*self.L_subcell # 1D shift array
        shift_array = np.array(np.meshgrid(ΔL,ΔL,ΔL, indexing='ij')).reshape(3, self.N_subcells).T
        self.ion_positions = (self.ion_positions + shift_array[:,None,:]).reshape(self.N_particles, 3)

    def fill_cell_with_particles(self):
        """
        Fill the cell with particles.
        """
        self.fill_subcell_with_particles()
        self.fill_cell_from_subcell()
        if self.perturb:
            self.perturb_particles()

class SubCellConfigurationGenerator():
    def __init__(self, L_cell, gofr_func, N_particles, r_correlation = None, approx_dx = None, δ_perturb = None, min_distance_δ =None):
        self.L_cell = L_cell
        self.gofr_func = gofr_func
        self.N_particles = N_particles
        self.δ_perturb  = δ_perturb
        self.min_distance_δ  = min_distance_δ
        
        if r_correlation is None:
            print("Using default value for r_correlation of 3.0")
            self.r_correlation = 3.0
        else:
            self.r_correlation = r_correlation
        if approx_dx is None:
            print("Using default value for approx_dx of 0.3")
            self.approx_dx = 0.3
        else:   
            self.approx_dx = approx_dx

        self.create_mesh()
        self.print_subcell_info()
        self.make_adjacent_information()
        self.create_subcell_array()
    
    def print_subcell_info(self):
        print(f"Cell of side-length {self.L_cell:0.2e}, and correlation length: {self.r_correlation:0.2e}")
        print(f"Created {self.N_cells_per_dim}x{self.N_cells_per_dim}x{self.N_cells_per_dim} = {self.N_cells_per_dim**3} subcells ")
        
    def periodic_distance(self, x1, x2):
        """ Calculate the minimum image distance accounting for periodic boundary conditions """
        # self.distance = lambda x1, x2: np.abs(x1 - x2) - self.L_cell * np.round(np.abs(x1 - x2) / self.L_cell)
        # self.distance = np.vectorize(self.distance)
        # self.distance = np.linalg.norm(self.distance, axis=0)

        # r = np.linalg.norm(np.min([np.abs(x1-x2), np.abs(np.abs(x1-x2)-self.L_cell)],axis=0), axis=0)
        x1, x2 = np.array(x1), np.array(x2)
        r = np.linalg.norm(np.abs(x1-x2) -self.L_cell*np.round(np.abs(x1-x2)/self.L_cell),axis=0)
        return r
    
    def make_adjacent_information(self):
        shift_Xi  = np.meshgrid( np.arange(-1,2),np.arange(-1,2), np.arange(-1,2), indexing='ij' )
        self.adjacent_subcell_indices = lambda subcell_indices: (np.vstack([Xi.ravel() for Xi in shift_Xi]).T + subcell_indices)%self.N_cells_per_dim
        
    def create_mesh(self):
        # Define number of cells
        # correlation_distance = 6
        self.N_cells_per_dim = np.max([3, int(self.L_cell//self.r_correlation)])

        # Make grid compatible with these cells with approximate dx
        Nx_approx = (self.L_cell/self.approx_dx)
        self.Nx = int(self.N_cells_per_dim*(Nx_approx//self.N_cells_per_dim))
        self.dx = self.L_cell/self.Nx
        
        # Mesh grids for the entire domain (could be adjusted to only create necessary subcell meshes)
        self.x = np.linspace(0, self.L_cell, self.Nx)
        self.y = np.linspace(0, self.L_cell, self.Nx)
        self.z = np.linspace(0, self.L_cell, self.Nx)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.G = np.ones_like(self.X)
        
    
    # Creates a 3D array of subcells
    def create_subcell_array(self):
        # Actual subcell slicing
        self.xi_sub_slice = lambda i: slice(self.x_indices_list[i][0], self.x_indices_list[i][-1] + 1  ) 
        self.subcell_mesh_slice = lambda xi, yi, zi: (self.xi_sub_slice(xi), self.xi_sub_slice(yi), self.xi_sub_slice(zi) )
        
        self.x_indices_list = np.array(np.split(np.arange(self.Nx),self.N_cells_per_dim)).astype(int) # splits mesh into submeshes 
        self.L_subcell = self.x[self.x_indices_list[0,-1]] - self.x[self.x_indices_list[0,0]]
        self.subcell_list = np.ones((self.N_cells_per_dim,self.N_cells_per_dim,self.N_cells_per_dim)).tolist()
        # Instantiate subcells and calcuate density
        for ix in range(self.N_cells_per_dim):
            for iy in range(self.N_cells_per_dim):
                for iz in range(self.N_cells_per_dim):
                    subcell_mesh = self.X[self.subcell_mesh_slice(ix,iy,iz)], self.Y[self.subcell_mesh_slice(ix,iy,iz)], self.Z[self.subcell_mesh_slice(ix,iy,iz)]
                    subcell_G = self.G[self.subcell_mesh_slice(ix,iy,iz)]
                    self.subcell_list[ix][iy][iz] = SubCell( (ix,iy,iz), subcell_mesh, self.L_cell, subcell_G)
    
    def update_G_from_position(self, ion_subcell_indices, ion_position):
        for subcell_indices in self.adjacent_subcell_indices(ion_subcell_indices):
            self.subcell_list[subcell_indices[0]][subcell_indices[1]][subcell_indices[2]].update_G_from_position(ion_position, self.periodic_distance, self.gofr_func)
    
    def fill_space_with_particles(self):
        self.ion_positions = []
        N_particles_placed = 0
        while N_particles_placed < self.N_particles:
            try: 
                rand_subcell_indices = np.random.randint(self.N_cells_per_dim, size = (3)) # Improve later by creating list of remaining subcells that need population
                rand_subcell = self.subcell_list[rand_subcell_indices[0]][rand_subcell_indices[1]][rand_subcell_indices[2]] 
                ion_position = rand_subcell.get_random_new_position()
                self.update_G_from_position(rand_subcell_indices, ion_position)
                self.ion_positions.append(ion_position)
                N_particles_placed += 1
            except ValueError as err:
                print("ValueError: err. Retrying placement.") 
        self.ion_positions = np.array(self.ion_positions)
        self.perturb_particles()

    def perturb_particles(self):
        """
        Randomly perturb all particles to remove perfect periodicity.
        """
        if self.δ_perturb is None:
            self.δ_perturb = self.dx/4
        if self.min_distance_δ is None:
            self.min_distance_δ = self.dx/2
        if self.min_distance_δ>self.dx:
            print("Warning, minimum distance set too large (> self.dx). Setting to dx/2.")
            self.min_distance_δ = self.dx/2
        max_δ = (self.dx - self.min_distance_δ)/2
        # Randomly perturb the ion positions
        perturbation = np.random.normal(-self.δ_perturb, self.δ_perturb, size=self.ion_positions.shape)
        perturbation = np.clip(perturbation, -max_δ, max_δ)
        self.ion_positions += perturbation
        self.ion_positions = np.clip(self.ion_positions, 0, self.L_cell)

class SubCell():
    def __init__(self, cell_position, cell_mesh, L_full_cell, subcell_G):
        self.cell_position = cell_position
        self.X, self.Y, self.Z = cell_mesh
        self.Nx = len(self.X)
        self.L_full_cell = L_full_cell
        self.subcell_G = subcell_G
    
    def update_G_from_position(self, ion_position, distance_func, gofr_func):
        ion_position = np.array(ion_position)
        r_mesh = distance_func(ion_position[:, np.newaxis, np.newaxis, np.newaxis], np.array([self.X,self.Y,self.Z]) )
        g_mesh = gofr_func(r_mesh)
        self.subcell_G *= g_mesh
        
    def get_random_new_position(self):
        rng = np.random.default_rng()
        XYZ = np.array([self.X,self.Y,self.Z]).reshape(3, self.Nx**3).T
        flat_normalized_probability  = self.subcell_G.flatten()/np.sum(self.subcell_G)
        atom_pos = rng.choice(XYZ, p = flat_normalized_probability  )
        return atom_pos


def random_reject(total_num_ptcls, box_length, r_reject, rnd_gen = np.random.default_rng()):
    """
    Place particles with rejection sampling using linked cell list for efficiency.
    
    Parameters
    ----------
    total_num_ptcls : int
        Total number of particles to place
    r_reject : float 
        Rejection radius
    box_length: numpy.ndarray
        Box length of particle box
    rnd_gen : numpy.random.Generator
        Random number generator
        
    Returns
    -------
    pos_temp : numpy.ndarray
        Array of placed particle positions
    """
    
    # Initialize positions array
    pos_temp = np.zeros((total_num_ptcls, 3))
    
    # Set up cell list
    cell_size = r_reject
    box_lengths = np.array([box_length] * 3)
    ncells = (box_lengths / cell_size).astype(np.int32)
    ncells = np.maximum(ncells, np.ones(3, dtype=np.int32))
    
    # Initialize cell list arrays
    head = -np.ones(ncells[0] * ncells[1] * ncells[2], dtype=np.int32)
    list_next = -np.ones(total_num_ptcls, dtype=np.int32)
    
    # Place first particle
    for dim in range(3):
        pos_temp[0, dim] = rnd_gen.uniform(0, box_lengths[dim])
    
    # Add first particle to cell list
    cell_idx = get_cell_index(pos_temp[0], cell_size, ncells)
    head[cell_idx] = 0
    
    # Place remaining particles
    for i in range(1, total_num_ptcls):
        while True:
            # Sample new position
            pos_new = np.zeros(3)
            for d in range(3):
                pos_new[d] = rnd_gen.uniform(0, box_lengths[d])
            
            # Get cell index for new position
            cell_idx = get_cell_index(pos_new, cell_size, ncells)
            
            # Check neighboring cells
            reject = False
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        neigh_cell = get_neighbor_cell(cell_idx, dx, dy, dz, ncells)
                        if neigh_cell < 0:
                            continue
                            
                        # Check particles in this cell
                        p = head[neigh_cell]
                        while p >= 0:
                            pos_diff = pos_new - pos_temp[p]
                            
                            # Apply PBC
                            for k in range(3):
                                if pos_diff[k] > box_lengths[k]/2:
                                    pos_diff[k] -= box_lengths[k]
                                elif pos_diff[k] < -box_lengths[k]/2:
                                    pos_diff[k] += box_lengths[k]
                            
                            dist = np.sqrt(np.sum(pos_diff**2))
                            if dist <= r_reject:
                                reject = True
                                break
                            p = list_next[p]
                        
                        if reject:
                            break
                    if reject:
                        break
                if reject:
                    break
                    
            if not reject:
                # Accept position and add to cell list
                pos_temp[i] = pos_new
                list_next[i] = head[cell_idx]
                head[cell_idx] = i
                break
                
    return pos_temp

def get_cell_index(pos: np.ndarray, cell_size: float, ncells: np.ndarray) -> int:
    """
    Convert a 3D position to a cell index in the acceleration grid.
    
    Parameters
    ----------
    pos : np.ndarray
        3D position vector
    cell_size : float
        Size of each cell
    ncells : np.ndarray
        Number of cells in each dimension
        
    Returns
    -------
    int
        Flattened cell index
    """
    idx = np.floor(pos / cell_size).astype(np.int32)
    idx = np.minimum(idx, ncells - 1)
    return idx[0] + idx[1]*ncells[0] + idx[2]*ncells[0]*ncells[1]

def get_neighbor_cell(cell_idx: int, dx: int, dy: int, dz: int, ncells: np.ndarray) -> int:
    """
    Get the index of a neighboring cell, handling periodic boundary conditions.
    
    Parameters
    ----------
    cell_idx : int
        Current cell index
    dx, dy, dz : int
        Relative cell coordinates (-1, 0, or 1)
    ncells : np.ndarray
        Number of cells in each dimension
        
    Returns
    -------
    int
        Index of the neighboring cell
    """
    idx = cell_idx
    x = idx % ncells[0]
    idx = (idx - x) // ncells[0]
    y = idx % ncells[1]
    z = (idx - y) // ncells[1]
    
    x = (x + dx) % ncells[0]
    y = (y + dy) % ncells[1]
    z = (z + dz) % ncells[2]
    
    return x + y*ncells[0] + z*ncells[0]*ncells[1]