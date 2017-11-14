
import tempfile # used by pycodcif; FIXME

import numpy as np

from pycodcif import parse

from ase.atoms import Atoms
from ase.spacegroup import crystal
from ase.geometry import cell_to_cellpar


def cif_to_ase(cif_string):
    """
    Naive pycodcif usage
    FIXME:
    as soon as pycodcif supports CIFs as strings,
    the tempfile below should be removed

    Args:
        cif_string: (str) WYSIWYG

    Returns:
        ASE atoms (object) *or* None
        None *or* error (str)
    """
    with tempfile.NamedTemporaryFile(suffix='.cif') as tmp:
        tmp.write(cif_string)
        tmp.flush()

        try:
            parsed_cif = parse(tmp.name)[0][0]['values']
        except:
            return None, 'Invalid or non-standard CIF'

        if '_symmetry_int_tables_number' in parsed_cif:
            try:
                spacegroup = int(parsed_cif['_symmetry_int_tables_number'][0])
            except ValueError:
                return None, 'Invalid space group info in CIF'
        elif '_symmetry_space_group_name_h-m' in parsed_cif:
            spacegroup = parsed_cif['_symmetry_space_group_name_h-m'][0].replace(' ', '').strip()
            if not spacegroup:
                return None, 'Empty space group info in CIF'
        else:
            return None, 'Absent space group info in CIF'

        try:
            cellpar = (
                float( parsed_cif['_cell_length_a'][0].split('(')[0] ),
                float( parsed_cif['_cell_length_b'][0].split('(')[0] ),
                float( parsed_cif['_cell_length_c'][0].split('(')[0] ),
                float( parsed_cif['_cell_angle_alpha'][0].split('(')[0] ),
                float( parsed_cif['_cell_angle_beta'][0].split('(')[0] ),
                float( parsed_cif['_cell_angle_gamma'][0].split('(')[0] )
            )
            basis = np.transpose(
                np.array([
                    [ char.split('(')[0] for char in parsed_cif['_atom_site_fract_x'] ],
                    [ char.split('(')[0] for char in parsed_cif['_atom_site_fract_y'] ],
                    [ char.split('(')[0] for char in parsed_cif['_atom_site_fract_z'] ]
                ]).astype(np.float)
            )
        except:
            return None, 'Unexpected non-numerical values occured in CIF'

    symbols = parsed_cif.get('_atom_site_type_symbol')
    if not symbols:
        symbols = parsed_cif.get('_atom_site_label')
        if not symbols:
            return None, 'Cannot find atomic positions in CIF'
        symbols = [char.encode('ascii').translate(None, ".0123456789") for char in symbols]

    try:
        return crystal(
            symbols,
            basis=basis,
            spacegroup=spacegroup,
            cellpar=cellpar,
            primitive_cell=True,
            onduplicates='replace'
        ), None
    except:
        return None, 'Unrecognized sites or invalid site symmetry in CIF'


def ase_to_eq_cif(ase_obj):
    """
    From ASE object generate CIF
    with symmetry-equivalent atoms
    for the browser-based cif player
    """
    parameters = cell_to_cellpar(ase_obj.cell)

    cif_data  = 'data_tilde_labs\n'
    cif_data += '_cell_length_a    ' + "%2.6f" % parameters[0] + "\n"
    cif_data += '_cell_length_b    ' + "%2.6f" % parameters[1] + "\n"
    cif_data += '_cell_length_c    ' + "%2.6f" % parameters[2] + "\n"
    cif_data += '_cell_angle_alpha ' + "%2.6f" % parameters[3] + "\n"
    cif_data += '_cell_angle_beta  ' + "%2.6f" % parameters[4] + "\n"
    cif_data += '_cell_angle_gamma ' + "%2.6f" % parameters[5] + "\n"
    cif_data += "_symmetry_space_group_name_H-M '%s'" % getattr(ase_obj.info.get('spacegroup', object), 'symbol', 'P1') + "\n"
    cif_data += "_symmetry_Int_Tables_number %s" % getattr(ase_obj.info.get('spacegroup', object), 'no', 1) + "\n"
    cif_data += 'loop_' + "\n"
    cif_data += '_symmetry_equiv_pos_as_xyz' + "\n"
    cif_data += '+x,+y,+z' + "\n"
    cif_data += 'loop_' + "\n"
    cif_data += '_atom_site_type_symbol' + "\n"
    cif_data += '_atom_site_fract_x' + "\n"
    cif_data += '_atom_site_fract_y' + "\n"
    cif_data += '_atom_site_fract_z' + "\n"

    pos = ase_obj.get_scaled_positions(wrap=False)
    for n, item in enumerate(ase_obj):
        cif_data += "%s   % 1.8f   % 1.8f   % 1.8f\n" % (item.symbol, pos[n][0], pos[n][1], pos[n][2])
    return cif_data
