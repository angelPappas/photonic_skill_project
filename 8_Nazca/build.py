"""
Assembles the full circuit (5x MZM + ring resonator + photodiode chain,
wired to bond pads on the product die) and exports it to GDS.

Equivalent to notebook cell 11. Run directly with ``python -m nazca_design.build``,
or import ``run()`` from a notebook / other script.
"""

import numpy as np

import nazca as nz

from . import config
from .components import mzm, new_product_cell, photodiode, ring_resonator
from .layers import mtl, wvg
from pathlib import Path


def run(output_filename="T1.gds", num_channels=5, seed=None):
    """Build the full circuit and export it as a GDS file.

    Parameters
    ----------
    output_filename : str
        Path/name of the GDS file to write.
    num_channels : int
        Number of MZM + ring + photodiode channels to place.
    seed : int, optional
        Seed for the barcode's random bit sequence, for reproducible builds.
    """
    if seed is not None:
        np.random.seed(seed)

    C = new_product_cell(
        cell_name='T1',
        cell_width=config.cell_dim,
        cell_height=config.cell_dim,
        barcode_sequence=np.random.randint(2, size=70),
    ).put()

    for idx in range(num_channels):
        m = mzm(
            coupler_gap=config.coupler_gap,
            coupler_length=config.coupler_length,
            arm_distance=config.mzm_arm_distance,
            coupler_radius=config.coupler_radius,
            heater_length=config.heater_length,
        ).put(300, 400 * idx + 300)

        length_left = m.pin['a0'].x
        length_right = config.cell_dim - m.pin['b0'].x
        wvg.strt(length=length_left).put(m.pin['a0'])
        wvg.strt(length=length_left).put(m.pin['a1'])
        wvg.strt(length=length_right).put(m.pin['b0'])

        # Waveguide towards the ring resonator
        w1 = wvg.strt(length=400).put(m.pin['b1'])

        r1 = ring_resonator(
            ring_radius=config.ring_radius,
            coupler_radius=config.coupler_radius,
            coupler_length=config.coupler_length,
            coupler_gap=config.coupler_gap,
        ).put('rIn0', w1.pin['b0'])
        wvg.strt(length=config.cell_dim - r1.pin['rOut0'].x).put(r1.pin['rOut0'])

        pd = photodiode(
            photodiode_length=config.photodiode_length,
            photodiode_ncon_width=config.photodiode_ncon_width,
            photodiode_pcon_width=config.photodiode_pcon_width,
        ).put(r1.pin['rOut1'])

        # Metal connections for heaters
        mtl.sbend_p2p(pin1=C.pin[config.bb_pad_list[f"H{idx * 2 + 1}_in"]], pin2=m.pin['hbot0']).put()
        mtl.sbend_p2p(pin1=C.pin[config.bb_pad_list[f"H{idx * 2 + 1}_out"]].rot(180), pin2=m.pin['hbot1']).put()
        mtl.sbend_p2p(pin1=C.pin[config.bb_pad_list[f"H{idx * 2 + 2}_in"]], pin2=m.pin['htop0']).put()
        mtl.sbend_p2p(pin1=C.pin[config.bb_pad_list[f"H{idx * 2 + 2}_out"]].rot(180), pin2=m.pin['htop1']).put()

        # Metal connections for the photodiode
        mtl.sbend_p2p(pin1=C.pin[config.bb_pad_list[f"PD{idx + 1}"]].rot(180), pin2=pd.pin['p0']).put()
        mtl.ubend_p2p(pin2=pd.pin['n1'], pin1=C.pin[config.bb_pad_list["PD_GND"]].rot(180)).put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_filename
    nz.export_gds(filename=str(output_path))

    return C


if __name__ == "__main__":
    run()
