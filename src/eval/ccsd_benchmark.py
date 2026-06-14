import numpy as np
from pyscf import gto, scf, cc, dft
from pyscf.lib.parameters import BOHR

Z_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

def generate_ccsd_density(z, pos, query_coords_angstrom):
    """
    z: (N,) atomic numbers
    pos: (N, 3) atomic coordinates in Angstrom
    query_coords_angstrom: (Q, 3) spatial query points in Angstrom
    Returns: (Q,) scalar density in electrons / Bohr^3
    """
    # 1. Build PySCF Molecule
    atom_str = ""
    for atom_z, (x, y, z_coord) in zip(z, pos):
        symbol = Z_TO_SYMBOL[int(atom_z)]
        atom_str += f"{symbol} {x} {y} {z_coord}; "
    atom_str = atom_str.rstrip("; ")

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = 'def2-svp'
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    # 2. Run Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()

    # 3. Run CCSD
    mycc = cc.CCSD(mf)
    mycc.kernel()
    
    # Get unrelaxed CCSD density matrix
    # Using the 1-particle density matrix in the atomic orbital basis
    dm_ao = mycc.make_rdm1()
    
    # 4. Evaluate density on query grid
    query_coords_bohr = query_coords_angstrom / BOHR
    ao = dft.numint.eval_ao(mol, query_coords_bohr, deriv=0)
    density = dft.numint.eval_rho(mol, ao, dm_ao, xctype='LDA')
    
    return np.maximum(density, 0.0)
