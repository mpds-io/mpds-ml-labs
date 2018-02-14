
import re
import fractions
import cStringIO

from ase.atoms import Atoms
from ase.io.vasp import read_vasp
from ase.spacegroup import crystal

import spglib


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
    buff = cStringIO.StringIO(poscar_string)
    try:
        ase_obj = read_vasp(buff)
    except:
        error = 'Unexpected data occured in POSCAR'
    buff.close()
    return ase_obj, error


def symmetrize(ase_obj, accuracy=1E-03):
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
        lattice, positions, numbers = spglib.refine_cell(ase_obj, symprec=accuracy)
    except:
        return None, 'Error while structure refinement'

    try:
        spacegroup = int( symmetry.split()[1].replace("(", "").replace(")", "") )
    except (ValueError, IndexError):
        return None, 'Symmetry error (probably, coinciding atoms) in structure'

    try:
        return crystal(
            Atoms(
                numbers=numbers,
                cell=lattice,
                scaled_positions=positions,
                pbc=True
            ),
            spacegroup=spacegroup,
            primitive_cell=True,
            onduplicates='replace'
        ), None
    except:
        return None, 'Unrecognized sites or invalid site symmetry in structure'


FORMULA_SEQUENCE = ['Fr','Cs','Rb','K','Na','Li',  'Be','Mg','Ca','Sr','Ba','Ra',  'Sc','Y','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',  'Ac','Th','Pa','U','Np','Pu',  'Ti','Zr','Hf',  'V','Nb','Ta',  'Cr','Mo','W',  'Fe','Ru','Os',  'Co','Rh','Ir',  'Mn','Tc','Re',  'Ni','Pd','Pt',  'Cu','Ag','Au',  'Zn','Cd','Hg',  'B','Al','Ga','In','Tl',  'Pb','Sn','Ge','Si','C',   'N','P','As','Sb','Bi',   'H',   'Po','Te','Se','S','O',  'At','I','Br','Cl','F',  'He','Ne','Ar','Kr','Xe','Rn']

def get_formula(ase_obj, find_gcd=True):
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
