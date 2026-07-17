"""
Mach-Zehnder Modulator (MZM) component.

Consists of an input 2x2 directional coupler, two arms with heaters,
and an output 2x2 directional coupler.
"""

import nazca as nz
from .heater import heater
from .directional_coupler import coupler
from .product_cell import newProductCell
from ..layers import wvg
import numpy as np
from pathlib import Path

def MZM(coupler_gap, coupler_length, arm_distance, coupler_radius, heater_length):

    with nz.Cell(name="MZM") as C:
        # Input coupler
        input_coupler = coupler(coupler_length=coupler_length, gap=coupler_gap, input_distance=arm_distance,coupler_radius=coupler_radius).put()

        # Heater phase-shifters
        heater_bot = heater(heater_length=heater_length).put(input_coupler.pin['b0'])
        heater_top = heater(heater_length=heater_length).put(input_coupler.pin['b1'])

        # Output coupler
        output_coupler = coupler(coupler_length=coupler_length, gap=coupler_gap, input_distance=arm_distance,coupler_radius=coupler_radius).put(heater_bot.pin['b0'])

        # Place pins for MZM
        nz.Pin(name='a0', pin=input_coupler.pin['a0']).put()
        nz.Pin(name='a1', pin=input_coupler.pin['a1']).put()
        nz.Pin(name='b0', pin=output_coupler.pin['b0']).put()
        nz.Pin(name='b1', pin=output_coupler.pin['b1']).put()
        nz.Pin(name='htop0', pin=heater_top.pin['h0']).put()
        nz.Pin(name='htop1', pin=heater_top.pin['h1']).put()
        nz.Pin(name='hbot0', pin=heater_bot.pin['h0']).put()
        nz.Pin(name='hbot1', pin=heater_bot.pin['h1']).put()
        nz.put_stub()


    return C

if __name__ == "__main__":

    C = newProductCell(cell_name='T1', cell_width=2500, cell_height=2500, barcode_sequence=np.random.randint(2,size=70)).put()

    mzm = MZM(coupler_gap=4, coupler_length=150, arm_distance=200, coupler_radius=50, heater_length=200).put(300, 800)

    length_left = mzm.pin['a0'].x
    wvg.strt(length=length_left).put(mzm.pin['a0'])

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "mzm.gds"
    nz.export_gds(filename=str(output_path))