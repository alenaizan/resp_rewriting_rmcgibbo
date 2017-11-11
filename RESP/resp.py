import psi4
import numpy as np
import scipy
import scipy.spatial
import scipy.optimize

from vdwsurface import *
from respfit import *

bohr_to_angstrom = 0.52917721092

def resp(wfn, options={}):
    mol = wfn.molecule()
    n_atoms = mol.natom()
    # Check options
    for i in options.keys():
        options[i.upper()] = options.pop(i)
    if not ('N_VDW_LAYERS' in options.keys()):
        options['N_VDW_LAYERS'] = 4
    if not ('VDW_SCALE_FACTOR' in options.keys()):
        options['VDW_SCALE_FACTOR'] = 1.4
    if not ('VDW_INCREMENT' in options.keys()):
        options['VDW_INCREMENT'] = 0.2
    if not ('VDW_POINT_DENSITY' in options.keys()):
        options['VDW_POINT_DENSITY'] = 1.0
    if not ('RESP_A' in options.keys()):
        options['RESP_A'] = 0.0005
    if not ('RESP_B' in options.keys()):
        options['RESP_B'] = 0.1
    if not ('METHOD' in options.keys()):
        options['METHOD'] = 'scf'
    if not ('CHARGE_GROUPS' in options.keys()):
        options['CHARGE_GROUPS'] = range(n_atoms)
    if not ('TWO_STAGE_FIT' in options.keys()):
        options['TWO_STAGE_FIT'] = False
    if options['TWO_STAGE_FIT']:
        if not ('RESP_A2' in options.keys()):
            options['RESP_A2'] = 0.001
        if not ('FIT2' in options.keys()):
            options['FIT2'] = range(n_atoms)
        if not ('CHARGE_GROUPS2' in options.keys()):
            options['CHARGE_GROUPS2'] = range(n_atoms)

    options['mol_charge'] = mol.molecular_charge()
    # Get the coordinates of the nuclei in Angstroms
    symbols = []
    coordinates = np.asarray(mol.geometry())
    for i in range(n_atoms):
        symbols.append(mol.symbol(i))
    # the vdwsurface code expects coordinates in angstroms
    coordinates *= bohr_to_angstrom
    # Get the points at which we're going to calculate the ESP surface
    points = []
    for i in range(options['N_VDW_LAYERS']):
        scale_factor = options['VDW_SCALE_FACTOR'] + i * options['VDW_INCREMENT']
        this_shell = vdw_surface(coordinates, symbols, scale_factor, options['VDW_POINT_DENSITY'])
        points.append(this_shell)
    points = np.concatenate(points)
    # Calculate ESP values at the grid
    np.savetxt('grid.dat', points, fmt='%.9f', delimiter='\t')
    e, wfn = psi4.prop(options['METHOD'], properties=['GRID_ESP', 'MULLIKEN_CHARGES'], molecule=wfn.molecule(), return_wfn=True)
    options['esp_values'] = wfn.oeprop.Vvals()
    options['charges'] = np.asarray(wfn.atomic_point_charges()) 
    # Build a matrix of the inverse distance from each ESP point to each nucleus
    options['invr'] = 1.0/scipy.spatial.distance_matrix(points, coordinates)*bohr_to_angstrom
    # Run the optimization using SciPy
    charges = scipy.optimize.minimize(resp_objective, options['charges'], args=(options), method='SLSQP', tol=1e-8, options={'ftol':1e-8},
              constraints={'type': 'eq', 'fun':resp_constraint, 'args':[options['mol_charge'], options['CHARGE_GROUPS'], False]})
    charges = charges.x
    if options['TWO_STAGE_FIT']:
        a = options['RESP_A']
        options['RESP_A'] = options['RESP_A2']
        options['charges'] = charges
        charges2 = scipy.optimize.minimize(resp_objective, charges[options['FIT2']], args=(options, True), method='SLSQP', tol=1e-8, options={'ftol':1e-8},
                   constraints={'type': 'eq', 'fun':resp_constraint, 'args':[options['mol_charge'], options['CHARGE_GROUPS2'], True, options['charges'], options['FIT2']]})
        charges[options['FIT2']] = charges2.x
        options['RESP_A'] = a
    np.savetxt('charges', charges)
    # Print the parameters to disk
    psi4.core.print_out("\n ---------------------------------------------------\n")
    psi4.core.print_out(" RESTRAINED ELECTROSTATIC POTENTIAL PARAMETERS\n")
    psi4.core.print_out(" ---------------------------------------------------\n")
    psi4.core.print_out(" N_VDW_LAYERS:       %d\n" %(options["N_VDW_LAYERS"]))
    psi4.core.print_out(" VDW_SCALE_FACTOR:   %.3f\n" %(options["VDW_SCALE_FACTOR"]))
    psi4.core.print_out(" VDW_INCREMENT:      %.3f\n" %(options["VDW_INCREMENT"]))
    psi4.core.print_out(" VDW_POINT_DENSITY:  %.3f\n" %(options["VDW_POINT_DENSITY"]))
    psi4.core.print_out(" RESP_A:             %.4f\n" %(options["RESP_A"]))
    psi4.core.print_out(" RESP_B:             %.4f\n" %(options["RESP_B"]))
    psi4.core.print_out(" CHARGE_GROUPS:      [")
    for i in options['CHARGE_GROUPS'][:-1]:
        psi4.core.print_out('%d, ' %i)
    psi4.core.print_out('%d]\n' %options['CHARGE_GROUPS'][-1])
    psi4.core.print_out(" TWO_STAGE_FIT:      %s\n" %(options["TWO_STAGE_FIT"]))
    if options["TWO_STAGE_FIT"]:
        psi4.core.print_out(" RESP_A2:            %.4f\n" %(options["RESP_A2"]))
        psi4.core.print_out(" FIT2:               [")
        for i in options["FIT2"][:-1]:
            psi4.core.print_out('%d, ' %i)
        psi4.core.print_out('%d]\n' %options["FIT2"][-1])
        psi4.core.print_out(" CHARGE_GROUPS2:     [")
        for i in options['CHARGE_GROUPS2'][:-1]:
            psi4.core.print_out('%d, ' %i)
        psi4.core.print_out('%d]\n' %options['CHARGE_GROUPS2'][-1])

    psi4.core.print_out(" METHOD:             %s\n" %(options["METHOD"]))
    psi4.core.print_out(" ---------------------------------------------------\n")

    # Print the results to disk
    psi4.core.print_out("\n ----------------------------------------------\n")
    psi4.core.print_out(" RESTRAINED ELECTROSTATIC POTENTIAL CHARGES\n")
    psi4.core.print_out("   Center  Symbol  RESP Charge (a.u.)\n")
    psi4.core.print_out(" ----------------------------------------------\n")
    for i in range(len(charges)):
        psi4.core.print_out("   %5d    %s     %8.5f\n" %(i, symbols[i], charges[i]))
