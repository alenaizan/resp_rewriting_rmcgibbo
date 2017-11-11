import psi4
import resp

mol = psi4.geometry("""
C            0.012220093121    -0.717626540721     0.000000000000
O           -0.062545506204     0.679938040344     0.000000000000
H            0.518735639503    -1.098516178616     0.883563931671
H            0.518735639503    -1.098516178616    -0.883563931671
H           -1.002097021106    -1.091505681690     0.000000000000
H            0.811765758420     1.042084199023     0.000000000000
units angstrom
""")

psi4.set_options({'basis': '6-31g*'})

options = {'N_VDW_LAYERS'       : 4,
           'VDW_SCALE_FACTOR'   : 1.4,
           'VDW_INCREMENT'      : 0.2,
           'VDW_POINT_DENSITY'  : 1.0,
           'resp_a'             : 0.0005,
           'RESP_B'             : 0.1,
           'CHARGE_GROUPS'      : [0, 1, 2, 3, 4, 5],
           'TWO_STAGE_FIT'      : True,
           'RESP_A2'            : 0.001,
           'FIT2'               : [0, 2, 3, 4],
           'CHARGE_GROUPS2'     : [0, 1, 1, 1],
           'METHOD'             : 'scf'
           }

e, wfn = psi4.energy('scf', return_wfn=True)

resp.resp(wfn, options)
