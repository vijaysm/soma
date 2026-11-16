# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022-2026, Vijay Mahadevan.

"""The look every documentation figure shares: colours, sizes and how files are saved.

White backgrounds, so a figure reads the same in the site's light and dark
themes and in the wiki; ranks and categories in the Okabe-Ito palette, which
stays distinct to colour-blind readers; scalar fields in viridis.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe and Ito, "Color Universal Design" (2002): black first, then the colours.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
# Categories (ranks, ownership kinds) use the colours without black.
CATEGORY = OKABE_ITO[1:]
FIELD_CMAP = "viridis"
# Rendered (ParaView) figures: pixels, drawn at twice the size they are shown.
PNG_SIZE = (1400, 1000)
FONT = "DejaVu Sans"

matplotlib.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "svg.fonttype": "path",  # text as outlines: the same in every viewer, no font needed
    "svg.hashsalt": "soma",  # stable element ids, so the same figure gives the same file
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def svg_figure(width_in, height_in, **kwargs):
    """A matplotlib figure and axes for a schematic, white, without axes decoration."""
    fig, axes = plt.subplots(figsize=(width_in, height_in), **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_axis_off()
        ax.set_aspect("equal")
    return fig, axes


def save_svg(fig, path):
    """Saves a schematic with no date or tool metadata, so the bytes depend only on the drawing."""
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.05, metadata={"Date": None, "Creator": None})
    plt.close(fig)
