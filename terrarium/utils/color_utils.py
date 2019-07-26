###################################################
#
# Color and printing utilities
#
###################################################

import random

import webcolors
from colorama import Fore, Style, Back

colors_rgb = {}
for c in vars(Fore):
    try:
        colors_rgb[c] = webcolors.name_to_rgb(c)
    except:
        pass
colors_rgb


def rgb_to_hex(r, g, b):
    def clamp(x):
        return max(0, min(x, 255))

    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))


def hex_to_color_name(h):
    """Get closest named color for terminal printing"""
    distance = 3 * 255 ** 2 + 1
    color = None
    for name, rgb in colors_rgb.items():
        rgb2 = webcolors.hex_to_rgb(h)
        d = 0
        for _c in ["red", "green", "blue"]:
            d += (getattr(rgb2, _c) - getattr(rgb, _c)) ** 2
        if d < distance:
            distance = d
            color = name
    return color


def _colored(text, hex_color, fore_or_back):
    """Return colored text"""
    if hex_color is None:
        return text
    try:
        color_name = hex_to_color_name(hex_color)
    except ValueError:
        color_name = hex_color
    s = "{color}{text}{end}".format(
        color=getattr(fore_or_back, color_name.upper()), text=text, end=Style.RESET_ALL
    )
    return s


def colored(text, hex_color):
    """Return colored text"""
    return _colored(text, hex_color, Fore)


def colored_background(text, hex_color):
    """Return colored text"""
    return _colored(text, hex_color, Back)


def random_color():
    rgb = [int(random.random() * 255) for x in range(3)]
    return rgb_to_hex(*rgb)


def cstring(string, fore=None, back=None):
    s = string
    if fore:
        s = colored(s, fore)
    if back:
        s = colored_background(s, back)
    return s


def cprint(string, fore=None, back=None):
    print(cstring(string, fore, back))
