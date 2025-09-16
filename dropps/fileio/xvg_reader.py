import numpy as np

def read_xvg(xvg_path):
    try:
        data = np.loadtxt(xvg_path, commens=('#', '@'))
    except:
        print(f"ERROR: Cannot load xvg file {xvg_path}.")
    return data

def write_xvg(file_name, x, y_list, title="Plot", xlabel="X", ylabel="Y", legends=None):

    ys = np.array(y_list)
    ys = np.expand_dims(ys, axis=0) if ys.ndim == 1 else ys

    with open(file_name, 'w') as f:
        # Metadata
        f.write(f'@    title "{title}"\n')
        f.write(f'@    xaxis  label "{xlabel}"\n')
        f.write(f'@    yaxis  label "{ylabel}"\n')
        f.write('@TYPE xy\n')

        # Legends
        if legends:
            for i, legend in enumerate(legends):
                f.write(f'@ s{i} legend "{legend}"\n')
        
        # Data
        for i in range(len(x)):
            line = f"{x[i]:.5f}"
            for y in ys:
                line += f"\t{y[i]:.5f}"
            f.write(line + "\n")