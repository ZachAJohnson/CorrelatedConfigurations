import numpy as np


class ConfigurationGenerator():

    def __init__(self, L_cell, gofr_func, N_particles_per_subcell, N_subcells_per_dim, r_correlation = 3, approx_dx = 0.3):
        self.L_cell = L_cell
        self.r_correlation = r_correlation
        self.gofr_func = gofr_func
        self.N_particles_per_subcell = N_particles_per_subcell
        self.N_subcells_per_dim = N_subcells_per_dim
        
        self.N_particles = N_particles_per_subcell * N_subcells_per_dim**3
        self.L_subcell = L_cell/self.N_subcells_per_dim
        
    def periodic_distance(self, x1, x2):
        """ Calculate the minimum image distance accounting for periodic boundary conditions """
        x1, x2 = np.array(x1), np.array(x2)
        r = np.linalg.norm(np.abs(x1-x2) -self.L_cell*np.round(np.abs(x1-x2)/self.L_cell),axis=0)
        return r
    
    def create_mesh(self, dx=None):
        # Define number of cells

        # Make grid compatible with these cells with approximate dx
        if dx is None:
            a = self.L_subcell/(self.N_particles_per_subcell)**(1/3)
            self.dx = 0.1 * a
        else:
            self.dx = dx
    
        # Mesh grids for the entire domain (could be adjusted to only create necessary subcell meshes)
        Nx = int(self.L_subcell/self.dx)
        self.x = np.linspace(0, self.L_subcell, Nx, endpoint=False)
        self.y = np.linspace(0, self.L_subcell, Nx, endpoint=False)
        self.z = np.linspace(0, self.L_subcell, Nx, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.XYZ_list = np.array([self.X,self.Y,self.Z]).reshape(3, self.Nx**3).T

    def update_G_from_position(self, ion_position, distance_func, gofr_func):
        ion_position = np.array(ion_position)
        r_mesh = distance_func(ion_position[:, np.newaxis, np.newaxis, np.newaxis], np.array([self.X,self.Y,self.Z]) )
        g_mesh = gofr_func(r_mesh)
        self.subcell_G *= g_mesh
        
    def get_random_new_position(self):
        rng = np.random.default_rng()
        
        flat_normalized_probability  = self.subcell_G.flatten()/np.sum(self.subcell_G)
        atom_pos = rng.choice(XYZ, p = flat_normalized_probability  )
        return atom_pos
    
    def fill_subcell_with_particles(self):
        self.ion_positions = []
        N_particles_placed = 0
        while N_particles_placed < self.N_particles_per_subcell:
            try:                 
                ion_position = rand_subcell.get_random_new_position()
                self.update_G_from_position(rand_subcell_indices, ion_position)
                self.ion_positions.append(ion_position)
                N_particles_placed += 1
            except ValueError as err:
                print("ValueError: err. Retrying placement.") 
            
            
if __name__ == "__main__":
    config = ConfigurationGenerator(10, lambda x: np.heaviside(x-1,0), 100, 2)
    config.create_mesh()
    print(config.dx, config.x, config.X.size, config.N_particles, config.X.size/config.N_particles)

class Cell():
    def __init__(self, L_cell, gofr_func, N_particles, r_correlation = 3, approx_dx = 0.3):
        self.L_cell = L_cell
        self.r_correlation = r_correlation
        self.gofr_func = gofr_func
        self.N_particles = N_particles
        
        self.create_mesh(approx_dx)
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
        
        
    def create_mesh(self, approx_dx):
        # Define number of cells
        # correlation_distance = 6
        self.N_cells_per_dim = np.max([3, int(self.L_cell//self.r_correlation)])

        # Make grid compatible with these cells with approximate dx
        Nx_approx = (self.L_cell/approx_dx)
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


