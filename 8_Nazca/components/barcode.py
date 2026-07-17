"""
Barcode marking.
"""

import nazca as nz
import numpy as np
from pathlib import Path

from ..layers import wg_width, wvg

def barcode(barcode_length, barcode_sequence):
    """Building a barcode cell: one vertical bar for every '1' in barcode_sequence."""
    with nz.Cell(name='barcode') as C:
        x_displacement = 0
        for barcode_number in barcode_sequence:
            x_displacement += wg_width
            if barcode_number == 1:
                wvg.strt(length=barcode_length).put(x_displacement, 0, 90)
    return C

if __name__ == "__main__":

    with nz.Cell(name="Die") as C:
        barcode(barcode_length=100, barcode_sequence=np.random.randint(2, size=70)).put()

    C.put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "barcode.gds"
    nz.export_gds(filename=str(output_path))
