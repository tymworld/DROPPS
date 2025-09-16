from copy import deepcopy

parameters_dict_template = {

    # Integration
    "integrator" : "Langevin",
    "friction"   : 1.0,
    "dt"         : 0.01,
    "nsteps"     : 100000,
    "minimize"   : True,
    "max_step"   : 100000,
    "seed"       : 1215,
    "comm_mode"  : "Linear",
    "nstcomm"    : 100,
    "bondtype"   : "bond",

    # Energy minimization
    #"energytol"  : 10.0,
    "forcetol"   : 100.0,
    
    # Output control
    "nst_xout"    : 1000,
    "nst_cp"      : 10000,
    #"nstvout"    : 10000,

    "nst_screenlog"  : 1000,
    "nst_filelog"    : 10000,
    "screenlog_grps" : "step,elapsedTime,remainingTime,speed,progress",
    "filelog_grps"    : "step,potentialEnergy,kineticEnergy",
    
    # Neighbor searching
    "nstlist"     : 10,
    #"cutoff_scheme_lj": "static",
    #"cutoff_scheme_coul": "static",
    #"cutoff_lj_multi": 3.0,
    #"cutoff_coul_multi": 3.0,
    "cutoff_lj"   : 1.5,
    "cutoff_coul" : 1.5,
    "buffer"      : 0.5,

    # Force calculations
    "coulombtype" : 'yukawa',
    "vdwtype"     : 'pLJ',
    "salt_conc"   : 0.1,

    # Temperature coupling
    "tcoulp"                 : "Bussi",
    "production_temperature" : 300.0,
    "gen_vel"                : True,
    "initial_temperature"    : 150.0,
    "warming_speed"          : 1,

    # Pressure coupling
    "pcoulp"      : True,
    "ref_P"       : 1.0,
    "tau_P"       : 5.0,
}

def getparameter(mdp_file):

    parameters = deepcopy(parameters_dict_template)
    
    try:
        raw_parameters = {line.strip().split("#")[0].split("=")[0].strip().replace("-", "_"):
                          line.strip().split("#")[0].split("=")[1].strip().replace("-", "_")
                          for line in open(mdp_file, 'r').readlines()
                          if len(line.strip().split()) > 0 and line[0] != '#'}
    except:
        print("ERROR: Cannot process parameter file %s." % mdp_file)
        quit()
        
    for param in raw_parameters.keys():
        if not param in parameters.keys():
            print("ERROR: Unknown key %s in parameter file." % param)
            quit()
        parameters[param] = type(parameters[param])(raw_parameters[param])

    for param in raw_parameters.keys():
        if type(parameters[param]) == bool:
            if raw_parameters[param] == "True":
                parameters[param] = True
            elif raw_parameters[param] == "False":
                parameters[param] = False
            else:
                print("ERROR: Cannot process parameter %s = %s." % (param, raw_parameters[param]))
                quit()
    
    return parameters
    

if __name__ == '__main__':
    print(getparameter("../templates/md.mdp"))
    


