"""Colors Utilities."""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors

COLORS_DICT = {
    "red_paper": (176 / 255, 44 / 255, 39 / 255),
    "light_red_paper": (214 / 255, 155 / 255, 153 / 255),
    "middle_red_paper": (208 / 255, 102 / 255, 98 / 255),
    "dark_red_paper": (147 / 255, 36 / 255, 33 / 255),
    "dark2_red_paper": (128 / 255, 36 / 255, 33 / 255),
    "blue_paper": (41 / 255, 67 / 255, 102 / 255),
    "light_blue_paper": (173 / 255, 225 / 255, 244 / 255),
    "middle_blue_paper": (124 / 255, 192 / 255, 205 / 255),
    "dark_blue_paper": (28 / 255, 43 / 255, 74 / 255),
    "grey_paper": (184 / 255, 184 / 255, 184 / 255),
    "light_gray_paper": (242 / 255, 242 / 255, 242 / 255),
    "dark_gray_paper": (85 / 255, 85 / 255, 85 / 255),
    "light_brown_paper": (143 / 255, 122 / 255, 110 / 255),
    "brown_paper": (102 / 255, 87 / 255, 78 / 255),
}

CUSTOM_MAP = LinearSegmentedColormap.from_list(
    "custom",
    [
        COLORS_DICT["dark_blue_paper"],
        COLORS_DICT["blue_paper"],
        COLORS_DICT["middle_blue_paper"],
        COLORS_DICT["light_blue_paper"],
        COLORS_DICT["grey_paper"],
        COLORS_DICT["light_red_paper"],
        COLORS_DICT["middle_red_paper"],
        COLORS_DICT["red_paper"],
        COLORS_DICT["dark_red_paper"],
    ],
    N=200,
)

CUSTOM_MAP_2 = LinearSegmentedColormap.from_list(
    "custom",
    [
        COLORS_DICT["dark_red_paper"],
        COLORS_DICT["red_paper"],
        COLORS_DICT["middle_red_paper"],
        COLORS_DICT["light_red_paper"],
        COLORS_DICT["grey_paper"],
        COLORS_DICT["light_blue_paper"],
        COLORS_DICT["middle_blue_paper"],
        COLORS_DICT["blue_paper"],
        COLORS_DICT["dark_blue_paper"],
    ],
    N=200,
)

# Sequential colormap using paper colors without dark_blue domination.
CUSTOM_MAP_SEQ = LinearSegmentedColormap.from_list(
    "custom_seq",
    [
        (0.00, COLORS_DICT["dark_red_paper"]),
        (0.12, COLORS_DICT["red_paper"]),
        (0.25, COLORS_DICT["middle_red_paper"]),
        (0.38, COLORS_DICT["light_red_paper"]),
        (0.50, COLORS_DICT["grey_paper"]),
        (0.62, COLORS_DICT["light_blue_paper"]),
        (0.78, COLORS_DICT["middle_blue_paper"]),
        (1.00, COLORS_DICT["blue_paper"]),
    ],
    N=200,
)

# Sequential red colormap: light (low) to dark (high).
CUSTOM_MAP_RED_SEQ = LinearSegmentedColormap.from_list(
    "custom_red_seq",
    [
        COLORS_DICT["light_gray_paper"],
        COLORS_DICT["light_red_paper"],
        COLORS_DICT["middle_red_paper"],
        COLORS_DICT["red_paper"],
        COLORS_DICT["dark_red_paper"],
        COLORS_DICT["dark2_red_paper"],
    ],
    N=200,
)

# Sequential blue colormap: light (low) to dark (high).
CUSTOM_MAP_BLUE_SEQ = LinearSegmentedColormap.from_list(
    "custom_blue_seq",
    [
        COLORS_DICT["light_gray_paper"],
        COLORS_DICT["light_blue_paper"],
        COLORS_DICT["middle_blue_paper"],
        COLORS_DICT["blue_paper"],
        COLORS_DICT["dark_blue_paper"],
    ],
    N=200,
)


def mix_colors(color1: str,
               color2: str,
               ratio: float = 0.5) -> tuple[float, float, float]:
    """Mix two hex colors.

    Args:
        color1: First color in hex format.
        color2: Second color in hex format.
        ratio: Ratio of the first color in the mix (0 to 1).

    Returns:
        Mixed color in RGB format.
    """
    red1, green1, blue1 = mcolors.to_rgb(color1)
    red2, green2, blue2 = mcolors.to_rgb(color2)
    return (red1 * ratio + red2 * (1 - ratio),
            green1 * ratio + green2 * (1 - ratio),
            blue1 * ratio + blue2 * (1 - ratio))
