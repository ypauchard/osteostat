#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from paper2026 import Osteostat, Mechanostat
from paper2026 import get_time_vectors, get_default_initial_conditions, get_drop_force_vector, get_day_index


# In[2]:


from pathlib import Path
import os
test_path = 'test_files/'
path = Path(test_path)


# In[3]:


number_of_failed_tests = 0


# ### Long-time unload/reload osteostat and mechanostat

# In[4]:


# Time-related parameters
n_years = 80
t_stop = n_years*365. # in days
t_inc = 1.0 # in days

F0 = 0.12 # [kN] Initial force

f_bm0 = 0.85 # [-] initial bone volume fraction

n_ramp = 1 # number of increments to change between forces (up or down ramp)

solver='dopri5'

sim_f_fac = [0.3, 0.65]
sim_strs = ['70pct', '35pct']


# In[5]:


t_eval, t_force = get_time_vectors(t_stop=t_stop, t_inc=t_inc, t_inc_force=t_inc)


# In[6]:


ostats = {'osteostat': Osteostat(f_bm0=f_bm0),
           'mechanostat': Mechanostat(f_bm0=f_bm0)}


# In[7]:


y0s=[]
for key in ostats:
    ostat = ostats[key]
    
    y0s.append(get_default_initial_conditions(ostat, F0, f_bm0))
    print(y0s[-1])


# In[8]:


# mid-point
# mid = t_force.size//2
mid = get_day_index(30*365, t_inc)

# solve all model-force situations
for i in range(len(sim_f_fac)):
    
    force = get_drop_force_vector(F0=F0, 
                                  F_drop=sim_f_fac[i]*F0,
                                 idx_up=mid,
                                 force_length=t_force.size,
                                 n_ramp=n_ramp)
    
    for j, key in enumerate(ostats):
        ostat = ostats[key]
        ostat.solve(t_eval, y0s[j], force , t_force, sim_strs[i], solver=solver)


# In[9]:


for name in ostats:
    ostat = ostats[name]
    print(f"\n*** RUNNING TESTS FOR {name} ***")
    test_file_prefix = 'test_lt_'+name+"_"
    for results_key in ostat.results.keys():
        print(f"--- {results_key} ---")
        results = ostat.results[results_key]
        for var in ['t', 'f_bm', 'Psi_0', 'Ot', 'Psi', 'mu', 'kform_OB', 'kform_OB_mech', 'kres_OC', 'kres_OC_mech', 'Ot_bar', 'F']:
            filename = path / Path(test_file_prefix + results_key+"_" + solver + '_' + var + '.npy')
            np_array = np.load(filename)
            if np.allclose(results[var], np_array):
                print(f'{filename}: OK')
            else:
                print(f'xxxx PROBLEM: {filename}: ERROR')
                number_of_failed_tests += 1


# ### Aging experiments 

# In[10]:


n_years = 40
t_stop = n_years*365. # in days
t_inc = 0.2 # in days

F0 = 0.12 # [kN] Initial force
f_bm0 = 0.85 # [-] initial bone volume fraction

ostat = Osteostat()

y0 = get_default_initial_conditions(ostat, F0, f_bm0)
t_eval, t_force = get_time_vectors(t_stop=t_stop, t_inc=t_inc, t_inc_force=t_inc)
force = F0 * np.ones(t_force.size) 


# In[11]:


A_factor = 2 # double A
year_to_change = 5 # when to apply the change to A

exp_str = f"{A_factor:.0f}xA" # label for experiement in results

# construct events dictionary to double A at the 5 year point
events = {get_day_index(year_to_change*365, t_inc): {'A': A_factor * ostat.A}}

# solve the ODEs
ostat.solve(t_eval, y0, force , t_force, exp_str, events=events, solver=solver)


# In[12]:


name='osteostat'
print(f"\n*** RUNNING TESTS FOR {name} ***")
test_file_prefix = 'test_ageing_'+name+"_"
for results_key in ostat.results.keys():
    print(f"--- {results_key} ---")
    results = ostat.results[results_key]
    for var in ['t', 'f_bm', 'Psi_0', 'Ot', 'Psi', 'mu', 'kform_OB', 'kform_OB_mech', 'kres_OC', 'kres_OC_mech', 'Ot_bar', 'F']:
        filename = path / Path(test_file_prefix + results_key+"_" + solver + '_' + var + '.npy')
        np_array = np.load(filename)
        if np.allclose(results[var], np_array):
            print(f'{filename}: OK')
        else:
            print(f'xxxx PROBLEM: {filename}: ERROR')
            number_of_failed_tests += 1


# In[13]:


print("-"*80)
if number_of_failed_tests == 0:
    print("SUCCESS: All tests passed")
else:
    print(f"FAILURE: A total of {number_of_failed_tests} tests failed.")
    print("Check output above for details:")
    print("    - When tests pass, the line ends with 'OK'.")
    print("    - When tests fail, the line starts with 'xxxx PROBLEM' and ends with 'ERROR'.")
print("-"*80)

