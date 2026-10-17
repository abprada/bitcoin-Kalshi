import numpy as np

from lib import slides

SD = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
DAYS = [1, 2, 3, 4, 5, 6, 7]


def _fig():
    z = np.linspace(0.4, 0.8, len(SD))[None, :] + 0.01 * np.arange(len(DAYS))[:, None]
    z_d = z + 0.02
    z_d[0, 0] = np.nan                       # sparse cells stay empty
    return slides.two_surfaces_3d(SD, DAYS, z, z_d, "Kalshi", "Deribit", "title", "subtitle"), z, z_d


def test_two_opaque_surfaces_red_then_gray():
    f, z, z_d = _fig()
    kalshi, deribit = f.data
    assert kalshi.type == deribit.type == "surface"
    assert (kalshi.name, deribit.name) == ("Kalshi", "Deribit")
    assert kalshi.colorscale[-1][1] == slides.MIT_RED and deribit.colorscale[-1][1] == slides.DARK_GRAY
    assert kalshi.opacity == deribit.opacity == 1
    assert not kalshi.showscale and not deribit.showscale
    assert kalshi.showlegend and deribit.showlegend
    np.testing.assert_array_equal(np.asarray(deribit.z, dtype=float), z_d)
    np.testing.assert_array_equal(np.asarray(kalshi.z, dtype=float), z)


def test_both_surfaces_share_one_color_range():
    f, z, z_d = _fig()
    lo, hi = np.nanmin([z, z_d]), np.nanmax([z, z_d])
    for t in f.data:
        assert (t.cmin, t.cmax) == (lo, hi)


def test_axes_are_labelled_in_sd_days_and_percent():
    f, z, z_d = _fig()
    s = f.layout.scene
    assert list(s.xaxis.tickvals) == [-3, -2, -1, 0, 1, 2, 3] and s.xaxis.ticktext[3] == "ATM"
    assert list(s.yaxis.tickvals) == DAYS and list(s.yaxis.ticktext) == [f"{d}d" for d in DAYS]
    assert s.zaxis.tickformat == ".0%"
    assert s.zaxis.range[0] <= np.nanmin([z, z_d]) and s.zaxis.range[1] >= np.nanmax([z, z_d])
    assert f.layout.font.family.startswith("Arial")
