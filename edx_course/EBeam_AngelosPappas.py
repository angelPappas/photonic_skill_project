"""Reconstruction of EBeam_AngelosPappas.gds (UBC PDK, TE/TM 1550 nm)."""
import gdsfactory as gf
from ubcpdk import PDK

PDK.activate()
C = PDK.cells
route = gf.routing.route_single

DIE_W, DIE_H = 605, 410
GC_PITCH = 127  # fibre-array pitch between the two grating couplers of a device

xs8 = gf.get_cross_section("strip", radius=8)
xs10 = gf.get_cross_section("strip", radius=10)
xs12 = gf.get_cross_section("strip", radius=12)


# ---------------------------------------------------------------- building blocks
def gc_pair(c, name="ebeam_gc_te1550"):
    """Two grating couplers (top, bottom) at the origin of c. Returns (top, bottom)."""
    gc = C[name]()
    top = c << gc
    top.dmove((0, GC_PITCH))
    return top, c << gc


def mzi_y(x_end, steps, wide=False):
    """Y-branch MZI: straight bottom arm, folded top arm.
    x_end : x of the combiner; steps : waypoints of the top arm; wide : 0.6 um arms + tapers."""
    c = gf.Component()
    y = C["ebeam_y_1550"]()
    sp = c << y
    cb = c << y
    cb.drotate(180)
    cb.dmove((x_end, 0))
    xs = gf.get_cross_section("strip", width=0.6, radius=8) if wide else xs8
    p_top = (sp.ports["o2"], cb.ports["o3"])
    p_bot = (sp.ports["o3"], cb.ports["o2"])
    if wide:
        taper = gf.c.taper(length=6, width1=0.5, width2=0.6)
        tp = [c << taper for _ in range(4)]
        for t, p in zip(tp, [sp.ports["o2"], sp.ports["o3"], cb.ports["o3"], cb.ports["o2"]]):
            t.connect("o1", p)
        p_top = (tp[0].ports["o2"], tp[2].ports["o2"])
        p_bot = (tp[1].ports["o2"], tp[3].ports["o2"])
    route(c, *p_bot, cross_section=xs)
    route(c, *p_top, cross_section=xs, steps=steps)
    c.add_port("o1", port=sp.ports["o1"])
    c.add_port("o2", port=cb.ports["o1"])
    return c


def loop_mirror_dc(dy=9):
    """Sagnac loop mirror from a directional coupler (gap 0.2, L 5)."""
    c = gf.Component()
    k = c << gf.c.coupler(gap=0.2, length=5)
    s1 = c << gf.c.bend_s(size=(15, dy), cross_section=xs10)
    s2 = c << gf.c.bend_s(size=(15, -dy), cross_section=xs10)
    s1.connect("o1", k.ports["o3"])
    s2.connect("o1", k.ports["o4"])
    route(c, s1.ports["o2"], s2.ports["o2"], cross_section=xs10)
    c.add_port("o1", port=k.ports["o1"])
    c.add_port("o2", port=k.ports["o2"])
    return c


def loop_mirror_y(dy=8.25):
    """Sagnac loop mirror from a Y-branch."""
    c = gf.Component()
    y = c << C["ebeam_y_1550"]()
    s1 = c << gf.c.bend_s(size=(15, dy), cross_section=xs10)
    s2 = c << gf.c.bend_s(size=(15, -dy), cross_section=xs10)
    s1.connect("o1", y.ports["o2"])
    s2.connect("o1", y.ports["o3"])
    route(c, s1.ports["o2"], s2.ports["o2"], cross_section=xs10)
    c.add_port("o1", port=y.ports["o1"])
    return c


# ---------------------------------------------------------------- devices (each with its 2 GCs)
def mzi_device(x_end, steps, wide=False):
    c = gf.Component()
    gt, gb = gc_pair(c)
    d = c << mzi_y(x_end, steps, wide)
    d.drotate(270)
    d.dmove((14, 107.6))
    route(c, gt.ports["o1"], d.ports["o1"], cross_section=xs10)
    route(c, gb.ports["o1"], d.ports["o2"], cross_section=xs10)
    return c


def fabry_perot():
    """Two loop mirrors joined by a waveguide cavity."""
    c = gf.Component()
    gt, gb = gc_pair(c)
    lm = loop_mirror_dc()
    m1 = c << lm
    m1.dmove((53.863, 102.65))
    m2 = c << lm
    m2.dmove((53.863, 23.65))
    route(c, gt.ports["o1"], m1.ports["o2"], cross_section=xs10)
    route(c, m1.ports["o1"], m2.ports["o2"], cross_section=xs10, steps=[dict(x=28), dict(y=26)])
    route(c, gb.ports["o1"], m2.ports["o1"], cross_section=xs10, steps=[dict(x=33.863), dict(y=22)])
    return c


def michelson():
    """BDC splitter, two unequal arms, each ended by a Y-branch loop mirror."""
    c = gf.Component()
    gt, gb = gc_pair(c)
    b = c << C["ebeam_bdc_te1550"]()
    b.dmove((105.45, 63.55))
    route(c, gt.ports["o1"], b.ports["o2"], cross_section=xs10)
    route(c, gb.ports["o1"], b.ports["o1"], cross_section=xs10)
    s1 = c << gf.c.bend_s(size=(15, 14), cross_section=xs10)
    s2 = c << gf.c.bend_s(size=(15, -14), cross_section=xs10)
    s1.connect("o1", b.ports["o3"])
    s2.connect("o1", b.ports["o4"])
    lm = loop_mirror_y()
    l1 = c << lm
    l1.dmove((197.15, 79.9))
    l2 = c << lm
    l2.dmove((197.15, 47.2))
    route(c, s1.ports["o2"], l1.ports["o1"], cross_section=xs8,
          steps=[dict(x=163.75), dict(y=117.782), dict(x=179.75), dict(y=79.9)])
    route(c, s2.ports["o2"], l2.ports["o1"], cross_section=xs10)
    return c


def tm_mzi():
    """TM MZI: two BDCs, delay arm on top, terminators on the unused ports."""
    c = gf.Component()
    gt, gb = gc_pair(c, "ebeam_gc_tm1550")
    b1 = c << C["ebeam_bdc_tm1550"]()
    b1.dmove((79.9, 63.5))
    b2 = c << C["ebeam_bdc_tm1550"]()
    b2.drotate(180)
    b2.dmove((212.408, 63.5))
    term = C["ebeam_terminator_tm1550"]()
    (c << term).connect("o1", b1.ports["o1"])
    (c << term).connect("o1", b2.ports["o1"])
    route(c, gt.ports["o1"], b1.ports["o2"], cross_section=xs10)
    route(c, gb.ports["o1"], b2.ports["o2"], cross_section=xs10, steps=[dict(x=222.308), dict(y=61.1)])
    route(c, b1.ports["o4"], b2.ports["o3"], cross_section=xs12)
    route(c, b1.ports["o3"], b2.ports["o4"], cross_section=xs12,
          steps=[dict(x=136.708), dict(y=89.9), dict(x=107.6), dict(y=113.9),
                 dict(x=160.708), dict(y=65.9)])
    return c


# ---------------------------------------------------------------- top cell
def build():
    top = gf.Component("EBeam_AngelosPappas")
    top.add_polygon([(0, 0), (DIE_W, 0), (DIE_W, DIE_H), (0, DIE_H)], layer="FLOORPLAN")

    devices = [
        # (device, polarization, x, y)
        (mzi_device(46.8,   [dict(x=15.4), dict(y=48.132), dict(x=31.4), dict(y=2.75)]),
         "TE", 44.969, 263),
        (mzi_device(71.873, [dict(x=40.473), dict(y=18.75), dict(x=-0.6), dict(y=34.75),
                             dict(x=56.473), dict(y=2.75)]),
         "TE", 155.82, 263),
        (mzi_device(80.528, [dict(x=49.128), dict(y=18.75), dict(x=-0.6), dict(y=34.75),
                             dict(x=49.128), dict(y=50.75), dict(x=-0.6), dict(y=66.75),
                             dict(x=65.128), dict(y=2.75)]),
         "TE", 253.289, 263),
        (mzi_device(83.873, [dict(x=46.473), dict(y=18.75), dict(x=5.4), dict(y=34.75),
                             dict(x=62.473), dict(y=2.75)], wide=True),
         "TE", 382.758, 263),
        (fabry_perot(), "TE", 480.277, 263),
        (michelson(),   "TE", 44.969, 20),
        (tm_mzi(),      "TM", 345.0,  20),
    ]
    for n, (dev, pol, x, y) in enumerate(devices, start=1):
        (top << dev).dmove((x, y))
        top.add_label(
            f"opt_in_{pol}_device_AngelosPappas_{n}",
            position=(x, y + GC_PITCH),   # o1 of the top GC
            layer="TEXT",
        )
    return top


if __name__ == "__main__":
    cell = build()
    cell.write("EBeam_AngelosPappas_rebuilt.gds")
    cell.show()
