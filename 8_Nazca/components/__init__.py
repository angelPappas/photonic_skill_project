"""
Photonic building blocks, one per module.

Importing this package guarantees ``nazca_design.layers`` has already run
(registering layers/xsections and creating the ``wvg``/``nmt``/``pmt``/``mtl``/
``phl``/``edg`` interconnects), so every component module below can safely
``from ..layers import wvg, ...`` without worrying about import order.
"""

from .. import layers  # noqa: F401  (side effect: registers xsections first)

from .directional_coupler import coupler
from .ring_resonator import ring_resonator
from .photodiode import photodiode
from .heater import heater
from .barcode import barcode
from .product_cell import new_product_cell
from .mzm import mzm

__all__ = [
    "coupler",
    "ring_resonator",
    "photodiode",
    "heater",
    "barcode",
    "new_product_cell",
    "mzm",
]
