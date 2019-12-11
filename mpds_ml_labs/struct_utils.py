
import math
import random
import itertools
import fractions
from functools import reduce

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from ase.atoms import Atom, Atoms
from ase.io.vasp import read_vasp
from ase.spacegroup import crystal

import spglib


__author__ = 'Evgeny Blokhin <eb@tilde.pro>'
__copyright__ = 'Copyright (c) 2018, Evgeny Blokhin, Tilde Materials Informatics'
__license__ = 'LGPL-2.1+'


def detect_format(string):
    """
    Detect CIF or POSCAR
    checking the most common features
    """
    if '_cell_angle_gamma' in string \
    and 'loop_' in string:
        return 'cif'

    lines = string.splitlines()

    for nline in [6, 7, 8]:
        if len(lines) <= nline:
            break
        if lines[nline].strip().lower().startswith('direct') \
        or lines[nline].strip().lower().startswith('cart'):
            return 'poscar'

    return None


def poscar_to_ase(poscar_string):
    """
    Parse POSCAR using ase

    Returns:
        Refined ASE structure (object) *or* None
        None *or* error (str)
    """
    ase_obj, error = None, None
    buff = StringIO(poscar_string)
    try:
        ase_obj = read_vasp(buff)
    except AttributeError:
        error = 'Types of atoms can be neither found nor inferred'
    except Exception:
        error = 'Cannot process POSCAR: invalid or missing data'
    buff.close()
    return ase_obj, error


def json_to_ase(datarow):
    """
    An extended *mpds_client* static *compile_crystal* method
    for handling the disordered structures
    in an oversimplified, very narrow-purpose way

    TODO?
    Avoid els_noneq rewriting
    """
    if not datarow or not datarow[-1]:
        return None, "No structure found"

    occs_noneq, cell_abc, sg_n, basis_noneq, els_noneq = \
        datarow[-5], datarow[-4], int(datarow[-3]), datarow[-2], datarow[-1]

    occ_data = None
    if any([occ != 1 for occ in occs_noneq]):
        partial_pos, occ_data = {}, {}
        for n in range(len(occs_noneq) - 1, -1, -1):
            if occs_noneq[n] != 1:
                disordered_pos = basis_noneq.pop(n)
                disordered_el = els_noneq.pop(n)
                partial_pos.setdefault(tuple(disordered_pos), {})[disordered_el] = occs_noneq[n]

        for xyz, occs in partial_pos.items():
            index = len(els_noneq)
            els_noneq.append(sorted(occs.keys())[0])
            basis_noneq.append(xyz)
            occ_data[index] = occs

    atom_data = []

    for n, xyz in enumerate(basis_noneq):
        atom_data.append(Atom(els_noneq[n], tuple(xyz), tag=n))

    if not atom_data:
        return None, "No atoms found"

    try:
        return crystal(
            atom_data,
            spacegroup=sg_n,
            cellpar=cell_abc,
            primitive_cell=True,
            onduplicates='error',
            info=dict(disordered=occ_data) if occ_data else {}
        ), None
    except:
        return None, "ASE cannot handle structure"


def refine(ase_obj, accuracy=1E-03, conventional_cell=False):
    """
    Refine ASE structure using spglib

    Args:
        ase_obj: (object) ASE structure
        accuracy: (float) spglib tolerance, normally within [1E-02, 1E-04]

    Returns:
        Refined ASE structure (object) *or* None
        None *or* error (str)
    """
    try:
        symmetry = spglib.get_spacegroup(ase_obj, symprec=accuracy)
        lattice, positions, numbers = spglib.standardize_cell(ase_obj, symprec=accuracy, to_primitive=not conventional_cell)
    except:
        return None, 'Error while structure refinement'

    try:
        spacegroup = int( symmetry.split()[1].replace("(", "").replace(")", "") )
    except (ValueError, IndexError, AttributeError):
        return None, 'Symmetry error (coinciding atoms?) in structure'

    try:
        return crystal(
            Atoms(
                numbers=numbers,
                cell=lattice,
                scaled_positions=positions,
                pbc=True
            ),
            spacegroup=spacegroup,
            primitive_cell=not conventional_cell,
            onduplicates='replace'
        ), None
    except:
        return None, 'Unrecognized sites or invalid site symmetry in structure'


FORMULA_SEQUENCE = ['Fr','Cs','Rb','K','Na','Li',  'Be','Mg','Ca','Sr','Ba','Ra',  'Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',  'Ac','Th','Pa','U','Np','Pu',  'Ti','Zr','Hf',  'V','Nb','Ta',  'Cr','Mo','W',  'Fe','Ru','Os',  'Co','Rh','Ir',  'Mn','Tc','Re',  'Ni','Pd','Pt',  'Cu','Ag','Au',  'Zn','Cd','Hg',  'B','Al','Ga','In','Tl',  'Pb','Sn','Ge','Si','C',   'N','P','As','Sb','Bi',   'H',   'Po','Te','Se','S','O',  'At','I','Br','Cl','F',  'He','Ne','Ar','Kr','Xe','Rn']

def get_formula(ase_obj, find_gcd=True, as_dict=False):
    parsed_formula = {}

    for label in ase_obj.get_chemical_symbols():
        if label not in parsed_formula:
            parsed_formula[label] = 1
        else:
            parsed_formula[label] += 1

    expanded = reduce(fractions.gcd, parsed_formula.values()) if find_gcd else 1
    if expanded > 1:
        parsed_formula = {el: int(content / float(expanded))
                        for el, content in parsed_formula.items()}

    if as_dict: return parsed_formula

    atoms = parsed_formula.keys()
    atoms = [x for x in FORMULA_SEQUENCE if x in atoms] + [x for x in atoms if x not in FORMULA_SEQUENCE]
    formula = ''
    for atom in atoms:
        index = parsed_formula[atom]
        index = '' if index == 1 else str(index)
        formula += atom + index

    return formula


def sgn_to_crsystem(number):
    if   195 <= number <= 230:
        return 'cubic'
    elif 168 <= number <= 194:
        return 'hexagonal'
    elif 143 <= number <= 167:
        return 'trigonal'
    elif 75  <= number <= 142:
        return 'tetragonal'
    elif 16  <= number <= 74:
        return 'orthorhombic'
    elif 3   <= number <= 15:
        return 'monoclinic'
    else:
        return 'triclinic'


MAX_ATOMS = 1000
SITE_SUM_OCCS_TOL = 0.99

def order_disordered(ase_obj):
    """
    This is a toy algo to get rid of the structural disorder;
    just one random possible ordered structure is returned
    (out of may be billions). No attempt to embrace all permutations is made.
    For that one needs to consider the special-purpose software (e.g.
    https://doi.org/10.1186/s13321-016-0129-3 etc.).

    Args:
        ase_obj: (object) ASE structure; must have *info* dict *disordered* and *Atom* tags
            *disordered* dict format: {'disordered': {at_index: {element: occupancy, ...}, ...}

    Returns:
        ASE structure (object) *or* None
        None *or* error (str)

    TODO?
    Rewrite space group info accordingly
    """
    for index in ase_obj.info['disordered']:
        if sum(ase_obj.info['disordered'][index].values()) < SITE_SUM_OCCS_TOL:
            ase_obj.info['disordered'][index].update(
                {'X': 1 - sum(ase_obj.info['disordered'][index].values())}
            )

    min_occ = min(
        sum(
            [list(item.values()) for item in ase_obj.info['disordered'].values()],
            []
        )
    )

    needed_det = math.ceil(1. / min_occ)
    if needed_det * len(ase_obj) > MAX_ATOMS:
        return None, 'Supercell size x%s is too big' % int(needed_det)

    diag = needed_det ** (1. / 3)
    supercell_matrix = [int(x) for x in (round(diag), math.ceil(diag), math.ceil(diag))]
    actual_det = reduce(lambda x, y: x * y, supercell_matrix)

    occ_data = {}
    for index, occs in ase_obj.info['disordered'].items():
        disorder = []
        for el, occ in occs.items():
            disorder += [el] * int(round(occ * actual_det))
        random.shuffle(disorder)
        occ_data[index] = itertools.cycle(disorder)

    order_obj = ase_obj.copy()
    order_obj *= supercell_matrix
    del order_obj.info['disordered']

    for index, occs in occ_data.items():
        for n in range(len(order_obj) - 1, -1, -1):
            if order_obj[n].tag == index:
                distrib_el = next(occs)
                if distrib_el == 'X':
                    del order_obj[n]
                else:
                    order_obj[n].symbol = distrib_el

    return order_obj, None
