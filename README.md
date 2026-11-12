# Osteostat 2026
[![DOI](https://zenodo.org/badge/724765298.svg)](https://doi.org/10.5281/zenodo.24166004)
## Installation
Use [miniconda](https://docs.conda.io/en/latest/miniconda.html) and install packages in a conda environment with versions used in paper:
```
conda create --name osteosim python=3.8
conda activate osteosim
(osteosim) pip install -r requirements.txt
```

Alternatively, use more recent versions:
```
conda create --name osteosim python=3.14
conda activate osteosim
(osteosim) pip install jupyterlab nbdime pandas scipy seaborn tqdm
```
## Notebooks
### Long-time unloading and reloading
The notebook [paper2026_unload_reload.ipynb](paper2026_unload_reload.ipynb) shows simulations similar to Section 4.2 Long-time unloading and reloading in the paper.

### Mechanical regulation under osteocyte disruption
The notebook [paper2026_mechanical_regulation.ipynb](paper2026_mechanical_regulation.ipynb) shows simulations similar to Section 4.3 Mechanical regulation under osteocyte disruption in the paper.

## Tests
The `test_files` folder contains saved numpy arrays for simulations in the `test_osteostat.ipynb` notebook. Run these tests in the terminal with:
```
conda activate osteosim
(osteosim) python test_osteostat.py
```
Examine the output:
- Total number number of tests that failed is printed at the end
- When tests pass, the line ends with `OK`.
- When tests fail, the line starts with `xxxx PROBLEM` and ends with `ERROR`.

Tests pass with Python 3.8.17 and:
```
jupyterlab==3.6.3
nbdime==3.2.1
pandas==1.5.3
scipy==1.10.1
seaborn==0.12.2
tqdm==4.65.0
```
Tests pass with Python 3.14.1 and :
```
jupyterlab==4.5.0
nbdime==4.0.2
pandas==2.3.3
scipy==1.16.3
seaborn==0.13.2
tqdm==4.67.1
```


## Organization of `paper2026.py` 
- Uses scipy ode solver
- Osteostat and Mechanostat classes hold equations as functions and parameters as instance attributes
- Methods starting with an underscore, e.g. `_g()` are internal methods, other methods are part of the public interface to be used in notebooks.
- With parameters set, a simulation can be computed by calling the `solve()` method
- The `solve()` method passes the `_g()` function to `solve_model()` function which is dispatching to scipy's `solve_ivp()` or `ode()` functions depending on the solver selected.
- Every call to `solve()` will create an entry in the `results` instance variable dictionary using `results_key_str`as the key. The results object stored is a dictionary with osteostat variables as keys. Use `model.results.keys()` to find out what results are available. Use `model.results['simulation_key_string'].keys()` to list all variables that were calculated.
- The `plot_results()` method can be used to visualize variables of a given simulation. For more control, access the results arrays directly (see above).
- Optionally, "events" can be set that modify any of the models parameters during simulation.
- Events are passed directly to the `solve()` method of the object as a dictionary where the key is the index of the time step, and the value is a dictionary of parameter (as a string) as the key and the new value of the parameter as the value.
- See `paper2026_mechanical_regulation.ipynb` for an example
