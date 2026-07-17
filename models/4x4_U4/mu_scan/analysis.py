import numpy as np
from afqmctools.analysis.average import average_one_rdm, average_two_rdm

import re
import pandas as pd
import sys

from pathlib import Path

def mu_value(s):
    return float(s.name[3:].replace("_","."))


Lx = 4
Ly = 4
Ns = Lx*Ly

def run_analysis(direc,f,print_all=True,print_header=False):
    energy_data = pd.read_csv(direc.name+"/ftafqmc.s000.scalar.dat", sep='(?<!#)\s+',engine='python')
    energy_data.columns = pd.Series(energy_data.columns.str.replace("#\s", ""))

    nmeas = energy_data.shape[0]

    Etot          = (energy_data['EnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).mean()/Ns
    Etot_err      = (energy_data['EnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).std()/np.sqrt(nmeas-1)
    Etot_err     /= Ns
    Eonebody      = (energy_data['OneBodyEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).mean()/Ns
    Eonebody_err  = (energy_data['OneBodyEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).std()/np.sqrt(nmeas-1)
    Eonebody_err /= Ns
    Ehubb         = (energy_data['ECoulEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).mean()/Ns
    Ehubb_err     = (energy_data['ECoulEnergyEstim__nume_real']/energy_data['EnergyEstim__deno_real']).std()/np.sqrt(nmeas-1)
    Ehubb_err    /= Ns

    mu = mu_value(direc)

    if(print_all):
        stat_file = direc.name+"/ftafqmc.s000.stat.h5"
        rho_avg, delta_rho = average_one_rdm(stat_file, estimator="mixed", eqlb=0)
        pmat_avg, delta_pmat = average_two_rdm(stat_file, estimator="mixed", eqlb=0)

        nn_list = np.zeros((Ns,3),dtype=int)
        for i in range(Ns):
            nn_list[i,0] = i
            # nearest-neighbor +x
            nn_list[i,1] = (i+1)%4+(i//Lx)*Lx
            # nearest-neighbor +y
            nn_list[i,2] = (i+4)%Ns

        nup = np.trace(rho_avg[0,:,:])/Ns
        ndn = np.trace(rho_avg[1,:,:])/Ns

        #print(f"nup = {nup}, ndn = {ndn}")

        nup_err = np.diag(delta_rho[0,:,:])
        ndn_err = np.diag(delta_rho[1,:,:])

        ntot     = (nup+ndn).real
        ntot_err = np.sqrt(np.sum(nup_err**2+ndn_err**2))/Ns

        #"""
        niup_nipxup = pmat_avg[0,nn_list[:,0],nn_list[:,0],nn_list[:,1],nn_list[:,1]]
        niup_nipyup = pmat_avg[0,nn_list[:,0],nn_list[:,0],nn_list[:,2],nn_list[:,2]]

        niup_nipxup_err = delta_pmat[0,nn_list[:,0],nn_list[:,0],nn_list[:,1],nn_list[:,1]]
        niup_nipyup_err = delta_pmat[0,nn_list[:,0],nn_list[:,0],nn_list[:,2],nn_list[:,2]]
        
        niup_nipxdn = pmat_avg[1,nn_list[:,0],nn_list[:,0],nn_list[:,1],nn_list[:,1]]
        niup_nipydn = pmat_avg[1,nn_list[:,0],nn_list[:,0],nn_list[:,2],nn_list[:,2]]

        niup_nipxdn_err = delta_pmat[1,nn_list[:,0],nn_list[:,0],nn_list[:,1],nn_list[:,1]]
        niup_nipydn_err = delta_pmat[1,nn_list[:,0],nn_list[:,0],nn_list[:,2],nn_list[:,2]]

        nipxup_nidn = pmat_avg[1,nn_list[:,1],nn_list[:,1],nn_list[:,0],nn_list[:,0]]
        nipyup_nidn = pmat_avg[1,nn_list[:,2],nn_list[:,2],nn_list[:,0],nn_list[:,0]]

        nipxup_nidn_err = delta_pmat[1,nn_list[:,1],nn_list[:,1],nn_list[:,0],nn_list[:,0]]
        nipyup_nidn_err = delta_pmat[1,nn_list[:,2],nn_list[:,2],nn_list[:,0],nn_list[:,0]]

        nidn_nipxdn = pmat_avg[2,nn_list[:,0],nn_list[:,0],nn_list[:,1],nn_list[:,1]]
        nidn_nipydn = pmat_avg[2,nn_list[:,0],nn_list[:,0],nn_list[:,2],nn_list[:,2]]
        
        nidn_nipxdn_err = delta_pmat[2,nn_list[:,0],nn_list[:,0],nn_list[:,1],nn_list[:,1]]
        nidn_nipydn_err = delta_pmat[2,nn_list[:,0],nn_list[:,0],nn_list[:,2],nn_list[:,2]]
        #"""

        #"""
        # +x neighbors
        nn_corr   = niup_nipxdn
        # +y neighbors
        nn_corr   = np.append(nn_corr, niup_nipydn)

        nn_corr_err = np.sqrt(np.sum(niup_nipxdn_err**2+niup_nipydn_err**2))/len(nn_corr)
        #"""

        """
        nn_corr = (niup_nipxup+niup_nipxdn+nipxup_nidn+nidn_nipxdn)
        nn_corr = np.append(nn_corr,(niup_nipyup+niup_nipydn+nipyup_nidn+nidn_nipydn))

        nn_corr_err = np.sqrt(np.sum(niup_nipxup_err**2+niup_nipyup_err**2
                                        +niup_nipxdn_err**2+niup_nipydn_err**2
                                        +nipxup_nidn_err**2+nipyup_nidn_err**2
                                        +nidn_nipxdn_err**2+nidn_nipydn_err**2))/len(nn_corr)
        """
        szsz_corr = 0.25*(niup_nipxup-niup_nipxdn-nipxup_nidn+nidn_nipxdn)
        szsz_corr = np.append(szsz_corr,0.25*(niup_nipyup-niup_nipydn-nipyup_nidn+nidn_nipydn))

        szsz_corr_err = 0.25*np.sqrt(np.sum(niup_nipxup_err**2+niup_nipyup_err**2
                                        +niup_nipxdn_err**2+niup_nipydn_err**2
                                        +nipxup_nidn_err**2+nipyup_nidn_err**2
                                        +nidn_nipxdn_err**2+nidn_nipydn_err**2))/len(szsz_corr)
        #"""
        if(mu is not None):
            if(print_header):
                print(f" {'mu':<10}  {'Etot':<10}  {'Etot_err':<10}  {'E_U':<10}  {'E_U_err':<10}  {'E_K':<10}  {'E_K_err':<10}  {'ntot':<10}  {'ntot_err':<10}  {'nn_corr':<10}  {'nn_corr_err':<10}  {'szsz_corr':<10}  {'szsz_corr_err':<10}", file=f)
            print(f"{mu: 10.8f} {Etot: 10.8f} {Etot_err: 10.8f} {Ehubb: 10.8f} {Ehubb_err: 10.8f} {Eonebody: 10.8f} {Eonebody_err: 10.8f} {ntot: 10.8f} {ntot_err: 10.8f} {np.mean(nn_corr).real: 10.8f} {nn_corr_err.real: 10.8f} {np.mean(szsz_corr).real: 10.8f} {szsz_corr_err.real: 10.8f}", file=f)
        else:
            if(print_header):
                print(f"  {'Etot':<10}  {'Etot_err':<10}  {'E_U':<10}  {'E_U_err':<10}  {'E_K':<10}  {'E_K_err':<10}  {'ntot':<10}  {'ntot_err':<10}  {'nn_corr':<10}  {'nn_corr_err':<10}  {'szsz_corr':<10}  {'szsz_corr_err':<10}", file=f)
            print(f" {Etot: 10.8f} {Etot_err: 10.8f} {Ehubb: 10.8f} {Ehubb_err: 10.8f} {Eonebody: 10.8f} {Eonebody_err: 10.8f} {ntot: 10.8f} {ntot_err: 10.8f} {np.mean(nn_corr).real: 10.8f} {nn_corr_err.real: 10.8f} {np.mean(szsz_corr).real: 10.8f} {szsz_corr_err.real: 10.8f}", file=f)

    else:
        if(mu is not None):
            if(print_header):
                print(f" {'mu':<10} {'Etot':<10}  {'Etot_err':<10}  {'E_U':<10}  {'E_U_err':<10}  {'E_K':<10}  {'E_K_err':<10}", file=f)
            print(f" {mu:10.8f} {Etot: 10.8f} {Etot_err: 10.8f} {Ehubb: 10.8f} {Ehubb_err: 10.8f} {Eonebody: 10.8f} {Eonebody_err: 10.8f}", file=f)
        else:
            if(print_header):
                print(f" {'Etot':<10}  {'Etot_err':<10}  {'E_U':<10}  {'E_U_err':<10}  {'E_K':<10}  {'E_K_err':<10}", file=f)
            print(f" {Etot: 10.8f} {Etot_err: 10.8f} {Ehubb: 10.8f} {Ehubb_err: 10.8f} {Eonebody: 10.8f} {Eonebody_err: 10.8f}", file=f)

def main():
    outfile = "props_vs_mu"
    dirs = [d for d in Path(".").glob("mu_*") if d.is_dir()]
    print_header = True
    with open(outfile, "w") as f:
        for direc in sorted(dirs, key=mu_value):
            run_analysis(direc,f,print_header=print_header)
            print_header=False

if __name__ == "__main__":
    main()
