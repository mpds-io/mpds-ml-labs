
import re
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
