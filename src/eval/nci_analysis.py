import numpy as np

def calculate_rdg(density_grid, step_size_bohr):
    """
    density_grid: (Nx, Ny, Nz) array
    step_size_bohr: distance between points in Bohr
    Returns: (Nx, Ny, Nz) RDG map
    """
    # 1. Compute the norm of the density gradient
    # np.gradient returns a list of N arrays, one for each dimension
    grad = np.gradient(density_grid, step_size_bohr)
    grad_norm = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)
    
    # 2. RDG Formula
    # RDG = |∇ρ| / (2 * (3 * pi^2)^(1/3) * ρ^(4/3))
    const = 2.0 * (3.0 * np.pi**2)**(1.0/3.0)
    
    # To prevent division by zero, mask out extremely low density regions
    epsilon = 1e-6
    safe_density = np.maximum(density_grid, epsilon)
    
    rdg = grad_norm / (const * safe_density**(4.0/3.0))
    
    # Zero out RDG in areas of virtually zero density to remove noise
    rdg[density_grid < epsilon] = 0.0
    
    return rdg
