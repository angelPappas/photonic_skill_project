"""
Photodiode component.
"""

import nazca as nz
import itertools
from pathlib import Path

from ..layers import wvg, pmt, nmt

_counter = itertools.count(1)

def photodiode(photodiode_pcon_width, photodiode_ncon_width, photodiode_length):

    with nz.Cell(name="Photodiode") as C:
        # This is the input taper of the photodiode
        taper = wvg.taper(width2=photodiode_pcon_width, length=15).put(0,0)
        # P-doped material
        pmt_strt = pmt.strt(length=photodiode_length, width=photodiode_pcon_width).put(taper)
        # N-doped material
        y_disp = 0.5*(photodiode_pcon_width + photodiode_ncon_width) + 1
        nmt_strt = nmt.strt(length=photodiode_length, width=photodiode_ncon_width).put(taper.pin['b0'].move(0, -y_disp))

        # Create pins
        nz.Pin('p0', pin=pmt_strt.pin['a0'].move(-1.5,0)).put()
        nz.Pin('p1', pin=pmt_strt.pin['b0'].move(-1.5,0)).put()
        nz.Pin('n0', pin=nmt_strt.pin['a0'].move(-3,0)).put()
        nz.Pin('n1', pin=nmt_strt.pin['b0'].move(-3,0)).put()

        nz.put_stub()

    return C

if __name__ == "__main__":

    with nz.Cell(name="Die") as C:
        photodiode(photodiode_pcon_width=10, photodiode_ncon_width=15, photodiode_length=100).put()

    C.put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "photodiode.gds"
    nz.export_gds(filename=str(output_path))
