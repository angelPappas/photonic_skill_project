"""
Directional coupler component.
"""

import itertools
import nazca as nz

from pathlib import Path


from ..layers import wvg

_counter = itertools.count(1)

def coupler (coupler_length, gap, input_distance, coupler_radius):
    
    name = f"Directional_Coupler{next(_counter)}"
    with nz.Cell(name=name) as C:

        top_wg = wvg.strt(length=coupler_length).put()
        bot_wg = wvg.strt(length=coupler_length).put(0, -gap)

        x_shift = coupler_radius*2
        y_shift = (input_distance - gap)/2

        # If I understand correctly, because the connection is to a0 of straight, then the input straigths have flipped ports.
        # which is like their axis are flipped in a way that >0 x_shift means go to the left :/
        input_top = wvg.strt(length=1).put(top_wg.pin['a0'].move(x_shift, -y_shift))
        wvg.sbend_p2p(pin2=input_top.pin['a0'], pin1=top_wg.pin['a0']).put()

        input_bot = wvg.strt(length=1).put(bot_wg.pin['a0'].move(x_shift, y_shift))
        wvg.sbend_p2p(pin2=input_bot.pin['a0'], pin1=bot_wg.pin['a0']).put()

        output_top = wvg.strt(length=1).put(top_wg.pin['b0'].move(x_shift, y_shift))
        wvg.sbend_p2p(pin2=output_top.pin['a0'], pin1=top_wg.pin['b0']).put()

        output_bot = wvg.strt(length=1).put(bot_wg.pin['b0'].move(x_shift, -y_shift))
        wvg.sbend_p2p(pin2=output_bot.pin['a0'], pin1=bot_wg.pin['b0']).put()
    
        nz.Pin(name='a0', pin=input_bot.pin['b0']).put()
        nz.Pin(name='a1', pin=input_top.pin['b0']).put()
        nz.Pin(name='b0', pin=output_bot.pin['b0']).put()
        nz.Pin(name='b1', pin=output_top.pin['b0']).put()

    return C


if __name__ == "__main__":

    with nz.Cell(name='Cell') as C:
        coupler(coupler_length=100, gap=10, input_distance=30, coupler_radius=50).put()

    C.put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "DC.gds"
    nz.export_gds(filename=str(output_path))
