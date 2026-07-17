"""
Die / product cell with edge outline, barcode, and bond pads.
"""

import nazca as nz
from ..layers import edg, mtl
from .barcode import barcode
from pathlib import Path

import numpy as np

def newProductCell(cell_name, cell_width, cell_height, barcode_sequence):

    with nz.Cell(name=cell_name) as C:

        # Bottom line
        edg.strt(length=cell_width).put(0,0)

        # Top line
        edg.strt(length=cell_width).put(0,cell_height)

        # Left line
        edg.strt(length=cell_height).put(0,0,90)

        # Right line
        edg.strt(length=cell_height).put(cell_width,0,90)

        barcode(barcode_length=100, barcode_sequence=barcode_sequence).put(cell_width-200, 150)

        # Metal pads
        metal_pad_size = 50
        metal_pad_separation = 50
        total_number_pads = int((cell_width - metal_pad_separation*2) / (metal_pad_size + metal_pad_separation))

        for idx in range(total_number_pads):
            x_displacement = idx * (metal_pad_size+metal_pad_separation) + metal_pad_separation + 10
            mbot = mtl.strt(length=metal_pad_size, width=metal_pad_size).put(x_displacement, metal_pad_separation)
            mtop = mtl.strt(length=metal_pad_size, width=metal_pad_size).put(x_displacement, cell_height - metal_pad_separation)
    
            nz.text(text=f'BOT_{idx}', height=12, align='cc', layer=1002).put(mbot.pin['b0'].move(-metal_pad_size/2))
            nz.text(text=f'TOP_{idx}', height=12, align='cc', layer=1002).put(mtop.pin['b0'].move(-metal_pad_size/2))

            nz.Pin(f'BOT_{idx}', pin=mbot.pin['b0'].move(-metal_pad_size/2)).put()
            nz.Pin(f'TOP_{idx}', pin=mtop.pin['b0'].move(-metal_pad_size/2)).put()

        nz.put_stub()

    return C

if __name__ == "__main__":

    C = newProductCell(
        cell_name='T1',
        cell_height=2500,
        cell_width=2500,
        barcode_sequence=np.random.randint(2, size=70)).put()

    here = Path(__file__).resolve().parent          # .../components
    output_dir = here.parent / "gds_files"           # .../gds_files (sibling of components)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "final_circuit.gds"
    nz.export_gds(filename=str(output_path))
