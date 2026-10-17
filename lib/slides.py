"""Slide styling: MIT colors (brand.mit.edu/color) and the Kalshi-over-Deribit 3D surface (16).

MIT Sloan publishes no hex values of its own; it uses MIT's palette, with Arial for slides.
"""

import numpy as np
import plotly.graph_objects as go

MIT_RED, LIGHT_RED = "#750014", "#c4314b"          # MIT Red, and a lighter shade of it for low vols
DARK_GRAY, SILVER = "#40464c", "#b8c2cc"           # MIT Dark Gray 1 and Light Silver Gray
INK, MUTED, GRID, PLANE = "#212326", "#626a73", "#c9d0d8", "#f2f4f8"
FONT = "Arial, Helvetica, sans-serif"


def two_surfaces_3d(sd, days, z_red, z_gray, red_name, gray_name, title, subtitle) -> go.Figure:
    """Two opaque implied-vol surfaces on the same axes: `z_red` in MIT red, `z_gray` in gray. Where they cross,
    the color on top shows which market's vol is higher.

    `z_*` are (len(days), len(sd)) arrays of vols (decimal); empty cells are NaN and left as holes. Strikes in
    standard deviations from spot (ticks -3..+3, 0 = ATM), time in days, vol in percent. Both surfaces share one
    color range, so the same vol has the same shade on each.
    """
    z_red, z_gray = np.asarray(z_red, dtype=float), np.asarray(z_gray, dtype=float)
    lo, hi = float(np.nanmin([z_red, z_gray])), float(np.nanmax([z_red, z_gray]))

    def surface(z, name, low, high):
        return go.Surface(x=sd, y=days, z=z, name=name, showlegend=True, showscale=False, opacity=1.0,
                          colorscale=[[0, low], [1, high]], cmin=lo, cmax=hi,
                          lighting=dict(ambient=0.75, diffuse=0.6, specular=0.05, roughness=0.9),
                          hovertemplate=name + "<br>%{x:+.1f} SD<br>%{y} days left<br>vol %{z:.1%}<extra></extra>")

    axis = dict(showbackground=True, backgroundcolor=PLANE, gridcolor=GRID, zeroline=False, showspikes=False,
                linecolor=MUTED, tickfont=dict(size=12, color=MUTED), title_font=dict(size=14, color=INK))
    z_lo, z_hi = np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20       # to the nearest 5 points
    f = go.Figure([surface(z_red, red_name, LIGHT_RED, MIT_RED), surface(z_gray, gray_name, SILVER, DARK_GRAY)])
    f.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>",
                   x=0.03, y=0.95, font=dict(size=20, color=INK)),
        font=dict(family=FONT, color=INK), paper_bgcolor="white", height=640, margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.03, y=0.88, font=dict(size=13), itemsizing="constant", bgcolor="rgba(255,255,255,0.8)"),
        scene=dict(
            xaxis=dict(axis, title="strike: standard deviations from spot", tickvals=[-3, -2, -1, 0, 1, 2, 3],
                       ticktext=["−3", "−2", "−1", "ATM", "+1", "+2", "+3"], range=[-3.2, 3.2]),
            yaxis=dict(axis, title="days to Kalshi's expiry", tickvals=list(days), ticktext=[f"{d}d" for d in days],
                       range=[min(days) - 0.2, max(days) + 0.2]),
            zaxis=dict(axis, title="implied vol", tickformat=".0%", range=[z_lo, z_hi],
                       tickvals=np.arange(np.ceil(lo * 10) / 10, z_hi + 1e-9, 0.1)),
            aspectmode="manual", aspectratio=dict(x=1.35, y=1.1, z=0.75), domain=dict(x=[0, 1], y=[0, 0.95]),
            camera=dict(eye=dict(x=1.45, y=-1.5, z=0.62), center=dict(x=0, y=0, z=-0.2))))
    return f
