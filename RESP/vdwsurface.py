#######################################################################
## Calculation of the fused-sphere van der waals surface of a molecule.
#######################################################################

import numpy as np
from dotsphere import dotsphere

# A. Bondi (1964). "van der Waals Volumes and Radii". J. Phys. Chem. 68: 441. doi:10.1021/j100785a001
vdw_r = {'H':1.2, 'C':1.7, 'N':1.55, 'O':1.52, 'F':1.47, 'P':1.8, 'S':1.8, 'CL':1.75, 
        'AR':1.88, 'AS':1.85, 'BR':1.85, 'CD':1.62, 'CU':1.4, 'GA':1.87, 'AU':1.66, 
        'HE':1.4, 'IN':1.93, 'I':1.98, 'KR':2.02, 'PB':2.02, 'LI':1.82, 'MG':1.73,
        'HG':1.70, 'NE':1.54, 'NI':1.64, 'PD':1.63, 'PT':1.8, 'K':2.75, 'SE':1.90,
       'SI':2.1, 'AG':1.9, 'NA':2.27, 'TE':2.06, 'TL':1.96, 'SN':2.17, 'U':1.86,
        'XE':2.16, 'ZN':1.37} 

def vdw_surface(coordinates, elements, scale_factor, density):
    """Compute points on the van der Wall surface of molecules

    Parameters:
        coordinates: np.ndarray, shape=(n_atoms, 3)
            The cartesian coordinates of the nuclei, in units of ANGSTROMS
        elements: list, shape=(n_atoms)
            The element symbols (C, H, etc) for each of the nuceli
        scale_factor: float
            The points on the molecular surface are set at a distance of
            scale_factor * vdw_radius away from each of the atoms.
        density: float
            The (approximate) number of points to generate per square angstrom
            of surface area. 1.0 is the default recommended by Kollman & Singh.

    Outputs:
        surface_points: numpy array

   """ 
    radii = []
    surface_points = []
    if len(coordinates) != len(elements):
        raise ValueError('coordinates length does not match elements length')

    for i in elements:
        if i in vdw_r.keys():
            radii.append(vdw_r[i] * scale_factor)
        else:
            raise KeyError('{0} is a not supported element'.format(i))

    for i in range(len(coordinates)):
        dots = dotsphere(density * (4.0/3.0) * np.pi* np.power(radii[i], 3))
        for j in range(len(dots)):
            dots[j] = coordinates[i] + radii[i] * dots[j]

        # all of the atoms that i is close to
        neighbors = []
        neighbors_distance = []
        for j in range(len(coordinates)):
            if i == j:
                continue
            d = np.linalg.norm(coordinates[i] - coordinates[j])
            if d < (radii[i] + radii[j]):
                neighbors.append(j)
                neighbors_distance.append(d)

        neighbors = np.array(neighbors)[np.argsort(neighbors_distance)]

        for j  in range(len(dots)):
            accessible = True
            for k in neighbors:
                if np.linalg.norm(dots[j] - coordinates[k]) < radii[k]:
                    accessible = False
                    break
            if accessible:
                surface_points.append(dots[j])
    return np.array(surface_points)
