import numpy as np
from stats.stat_h5 import afobs, read

import re
import pandas as pd
import sys

def mu_from_direc_name(s):
    m = re.search(r'mu_(m?)(\d+)_(\d+)', s)
    if not m:
        return None
    sign = "-" if m.group(1) == "m" else ""
    return float(f"{sign}{m.group(2)}.{m.group(3)}")

direc = sys.argv[1]

energy_data = pd.read_csv(direc+"/ftafqmc.s000.scalar.dat", sep='(?<!#)\s+',engine='python')
energy_data.columns = pd.Series(energy_data.columns.str.replace("#\s", ""))

nmeas = energy_data.shape[0]

Etot          = (energy_data['EnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).mean()
Etot_err      = (energy_data['EnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).std()/np.sqrt(nmeas-1)
Eonebody      = (energy_data['OneBodyEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).mean()
Eonebody_err  = (energy_data['OneBodyEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).std()/np.sqrt(nmeas-1)
Ehubb         = (energy_data['ECoulEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).mean()
Ehubb_err     = (energy_data['ECoulEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).std()/np.sqrt(nmeas-1)

mu = mu_from_direc_name(direc)
print(f"mu = {mu}")
print_all = True

if(print_all):
    f = read(direc+"/ftafqmc.s000.stat.h5")
    rho_avg, delta_rho = afobs(f,"FullOneRDM",0,group="Mixed")

    Ns = rho_avg.shape[-1]

    nup = np.trace(rho_avg[0,:,:])
    ndn = np.trace(rho_avg[1,:,:])

    #print(f"nup = {nup}, ndn = {ndn}")

    nup_err = np.diag(delta_rho[0,:,:])
    ndn_err = np.diag(delta_rho[1,:,:])

    ntot     = (nup+ndn).real
    ntot_err = np.sqrt(np.sum(nup_err**2+ndn_err**2))/Ns

    if(mu is not None):
        Etot = Etot + mu*ntot
        Etot_err = np.sqrt(Etot_err**2+ntot_err**2)

    print(f" {'Etot':<10}  {'Etot_err':<10}  {'E_U':<10}  {'E_U_err':<10}  {'E_K':<10}  {'E_K_err':<10}  {'ntot':<10}  {'ntot_err':<10}")
    print(f"{Etot: 10.8f} {Etot_err: 10.8f} {Ehubb: 10.8f} {Ehubb_err: 10.8f} {Eonebody: 10.8f} {Eonebody_err: 10.8f} {ntot: 10.8f} {ntot_err: 10.8f}")

else:
    print(f" {'Etot':<10}  {'Etot_err':<10}  {'E_U':<10}  {'E_U_err':<10}  {'E_K':<10}  {'E_K_err':<10}")
    print(f" {Etot: 10.8f} {Etot_err: 10.8f} {Ehubb: 10.8f} {Ehubb_err: 10.8f} {Eonebody: 10.8f} {Eonebody_err: 10.8f}")
