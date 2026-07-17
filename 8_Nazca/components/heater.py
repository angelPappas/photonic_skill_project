"""
Phase-shifter using electrode heating as modulation mechanism.
"""

import nazca as nz
from pathlib import Path

from ..layers import phl, wvg, wg_width, heater_width

HeaterCounter=0
def heater(heater_length):
    global HeaterCounter
    global wg_width
    global heater_width
    HeaterCounter+=1

    with nz.Cell(name="Heater") as C:
        # Place the input taper
        input_taper = nz.taper(width1=wg_width, width2=heater_width, layer='WVG', length=7.5).put(0,0)

        heater_strt = phl.strt(length=heater_length).put(input_taper)

        output_taper = nz.taper(width2=wg_width, width1=heater_width, layer='WVG', length=7.5).put(heater_strt)

        nz.Pin('a0', pin=input_taper.pin['a0']).put()
        nz.Pin('b0', pin=output_taper.pin['b0']).put()
        nz.Pin('h0', pin=heater_strt.pin['a0']).put()
        nz.Pin('h1', pin=heater_strt.pin['b0']).put()
    return C


if __name__ == "__main__":
    with nz.Cell(name='Die') as C:
        w1 = wvg.strt(length=15).put()
        heater(heater_length=50).put(w1.pin['b0'])

    C.put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "heater.gds"
    nz.export_gds(filename=str(output_path))