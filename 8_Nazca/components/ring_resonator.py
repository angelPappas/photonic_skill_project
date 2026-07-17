"""
Ring resonator component.
"""

import nazca as nz

from pathlib import Path


from ..layers import wvg, wg_width

def ring_resonator(coupler_length, coupler_gap, coupler_radius,ring_radius):

  with nz.Cell(name = 'ring_resonator') as C:
    coupler_gap_updated = coupler_gap + wg_width
    top_wg = wvg.strt(length = coupler_length).put(0,0)
    wvg.bend(angle=360, radius=ring_radius).put(coupler_length/2, -ring_radius*2-coupler_gap_updated)
    bot_wg = wvg.strt(length = coupler_length).put(0,-ring_radius*2-coupler_gap_updated*2)

    x_shift = coupler_radius*2

    input_top = wvg.strt(length = x_shift).put(top_wg.pin['a0'])
    input_bot = wvg.strt(length = x_shift).put(bot_wg.pin['a0'])

    output_top = wvg.strt(length = x_shift).put(top_wg.pin['b0'])
    output_bot = wvg.strt(length = x_shift).put(bot_wg.pin['b0'])

    nz.Pin('rOut1', pin = input_top.pin['b0']).put()
    nz.Pin('rIn1', pin = output_top.pin['b0']).put()
    nz.Pin('rIn0', pin = input_bot.pin['b0']).put()
    nz.Pin('rOut0', pin = output_bot.pin['b0']).put()
    nz.put_stub()

  return C


if __name__ == "__main__":

    with nz.Cell(name = 'Cell') as C:
        ring_resonator(coupler_gap = 10,
            coupler_length = 30,
            coupler_radius = 50,
            ring_radius = 75).put()

    C.put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ring_resonator.gds"
    nz.export_gds(filename=str(output_path))
