import pickle

class tpr_content():
    def __init__(self, tpr_path):
        try:
            f = open(tpr_path, "rb")

        except:
            print(f"ERROR: Cannot open tpr file {tpr_path}")
            quit()
        else:
            with f:
                loaded_params = pickle.load(f)
                self.parameters = loaded_params["parameters"]
                self.mdsystem = loaded_params["mdsystem"]
                self.mdtopology = loaded_params["mdtopology"]
                self.positions = loaded_params["positions"]
                self.pdb_raw = loaded_params["pdb_raw"]
                self.itp_list = loaded_params["ITP_list"]

def read_tpr(tpr_path):
    return tpr_content(tpr_path)