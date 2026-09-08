import numpy as np


def write_xpm(array: np.ndarray, filename: str, title: str = "XPM Image"):
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("XPM input must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("XPM input contains NaN or infinite values.")

    height, width = array.shape
    palette = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-"
    color_count = len(palette)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum == minimum:
        fill_index = color_count - 1 if maximum != 0 else 0
        color_indices = np.full(array.shape, fill_index, dtype=int)
    else:
        normalized = (array - minimum) / (maximum - minimum)
        color_indices = np.rint(normalized * (color_count - 1)).astype(int)

    with open(filename, "w") as f:
        safe_title = title.replace("*/", "* /")
        f.write("/* XPM */\n")
        f.write(f"/* {safe_title} */\n")
        f.write("static char * xpm_data[] = {\n")
        f.write(f'"{width} {height} {color_count} 1",\n')

        for index, char in enumerate(palette):
            shade = 255 - round(255 * index / (color_count - 1))
            f.write(f'"{char} c #{shade:02X}{shade:02X}{shade:02X}",\n')

        for row_index, row in enumerate(color_indices):
            line = "".join(palette[index] for index in row)
            suffix = "," if row_index < height - 1 else ""
            f.write(f'"{line}"{suffix}\n')

        f.write("};\n")
