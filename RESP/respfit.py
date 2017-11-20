import numpy as np
bohr_to_angstrom = 0.52917721092
#**************************************************************************
# Resp fitting optimization
#**************************************************************************

def resp_objective(charges, options, two_stage=False):
    expected_esp = np.zeros(len(options['esp_values']))
    invr = np.copy(options['invr'])
    if two_stage:
        index = [i for i in range(len(options['charges'])) if i not in options['FIT2']]
        expected_esp = np.dot(options['invr'][:, index], options['charges'][index])
        invr = np.copy(options['invr'][:, options['FIT2']])

    expected_esp += np.dot(invr, charges)
    # predicted esp values at the grid points based on the point charge model
    esp_error = options['esp_values'] - expected_esp
    # figure of merit for how well the predicted charges match the actual
    chi2_esp = np.sum(esp_error**2)
    chi2_rstr = 0
    # hyperbolic restraint term  a*sum(sqrt(q**2 + b**2)-b)
    for charge in charges:
        chi2_rstr += np.sqrt(charge**2  + options['RESP_B']**2) - options['RESP_B']
    chi2_rstr *= options['RESP_A']
    chi2_d_esp = []
    for i in range(len(charges)):
        grad = 0
        for j in range(len(options['esp_values'])):
            grad += (options['esp_values'] - expected_esp)[j]*options['invr'][j][i]
        grad *= -2
        chi2_d_esp.append(grad)
    chi2_d_rstr = []
    for i in range(len(charges)):
        chi2_d_rstr.append(options['RESP_A']*charges[i]/np.sqrt(charges[i]**2+options['RESP_B']**2))
    chi2_d_esp = np.array(chi2_d_esp)
    chi2_d_rstr = np.array(chi2_d_rstr)
    return (chi2_esp + chi2_rstr, chi2_d_esp + chi2_d_rstr)

def resp_constraint(charges, mol_charge, charge_groups=None, two_stage=False, all_charges=None, fit2=None):
    total_charge = 0
    if two_stage:
       index = [i for i in range(len(all_charges)) if i not in fit2]
       total_charge += np.sum(all_charges[index])
    total_charge += np.sum(charges) - mol_charge
    error = total_charge**2

    groups = {}
    for i in range(len(charge_groups)):
        item = charge_groups[i]
        try:
           groups[item].append(i)
        except:
           groups[item] = [i]
    for i in groups.keys():
        items = groups[i]
        for j in range(len(items)):
            diff = charges[items[j]] - charges[items[0]]
            error += diff*diff
    return error
