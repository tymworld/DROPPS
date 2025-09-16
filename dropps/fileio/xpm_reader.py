import numpy as np

def write_xpm(array: np.ndarray, filename: str, title: str = "XPM Image"):
    height, width = array.shape

    # Map values to characters
    chars = {0: '.', 1: 'x'}  # You can define more if needed
    color_table = {
        '.': 'c white',
        'x': 'c black'
    }

    with open(filename, 'w') as f:
        f.write(f'/* XPM */\n')
        f.write(f'static char * xpm_data[] = {{\n')
        f.write(f'"{title}",\n')
        f.write(f'"{width} {height} {len(chars)} 1",\n')

        # Write color definitions
        for char, color in color_table.items():
            f.write(f'"{char} {color}",\n')

        # Write pixel data
        for row in array:
            line = ''.join(chars[int(val)] for val in row)
            f.write(f'"{line}",\n')

        f.write('};\n')

