import sys  # have progress bar print to stdout
import time  # measure execution time
from typing import Union

import matplotlib as mpl  # plot configuration
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # plot configuration
from scipy.integrate import ode, solve_ivp
from scipy.optimize import OptimizeResult
from tqdm import tqdm  # progress bar

supported_ivp_solvers = ["RK45", "RK23", "DOP853", "BDF", "Radau", "LSODA"]
supported_ode_solvers = ["dopri5", "lsoda"]


class OsteosimResult(OptimizeResult):
    pass


class OsteostatResult(OptimizeResult):
    pass


class Osteostat:
    def __init__(self, f_bm0=0.85, Ot_f=30000, occupancy=0.8) -> None:
        # parameters from Lerebours and Buenzli 2016
        alpha = 2e-4
        beta_ob = 0.5  # 0.5*1 mm cell/day
        beta_oc = 0.175  # 0.175 mm cell/day

        # parameters calculated
        self.alpha_ob = self.alpha_oc = alpha
        self.Ot_f = Ot_f
        self.A = (
            ((1 / occupancy) - 1) * (self._S_V(f_bm0) * (1 - f_bm0) * alpha) / f_bm0
        )
        self.gamma_ob = beta_ob / self.get_Ot_bar(f_bm0)
        self.gamma_oc = beta_oc / self.get_Ot_bar(f_bm0)

        # parameters from literature (see Table 2)
        self.C: float = 1.23e5  # [1/day] Cell desensitization rate
        self.m: float = 3e-4  # [-] Biochemical transduction parameter
        self.D_S: float = 100.0  # [1/day] Degradation rate of S
        self.D_R: float = 2.0  # [1/day] Degradation rate of R
        self.P_R: float = 1000.0  # [1 / (cell day)] Production rate of R
        self.gamma_K: float = 34e4  # [1/GPa] Parameter in K(\Psi_0^cell)
        self.f_0: float = 20.5  # [1/day] Mechanical transduction parameter
        self.S_0: float = 100.0  # [1/cell] Equilibrium number of S
        self.L2: float = 4.0  # [mm^2]
        self.c_bm: float = 0.0482  # [1/GPa] Component [1,1] of onverse stiffness tensor

        self.results: Union[dict, None] = (
            None  # once model is solved, results are augmented and stored here
        )
        self.label: str = "Osteostat"  # A descriptive label

    # -------------------------------------------
    #   internal methods to calculate derivatives
    # -------------------------------------------
    def _S_V(self, f_bm):
        """Eq. 10"""
        a1 = 14.1  # [1/mm]
        a2 = -10.5  # [1/mm]
        a3 = -17.8  # [1/mm]
        a4 = 43.0  # [1/mm]
        a5 = -28.8  # [1/mm]

        return a1 * f_bm + a2 * f_bm**2 + a3 * f_bm**3 + a4 * f_bm**4 + a5 * f_bm**5

    def _f_geom(self, f_bm):
        return self._S_V(f_bm) * (1 - f_bm)

    def _get_psi(self, F_at_t, f_bm):
        """Eq. 34"""
        return 1 / 2.0 * self.c_bm * (F_at_t / (self.L2 * f_bm)) ** 2

    def _mu_plus(self, mu):
        """mechanical stimulus mu for osteoblasts"""
        if mu > 0:
            return mu
        else:
            return 0

    def _mu_minus(self, mu):
        """mechanical stimulus mu for osteoclasts"""
        if mu < 0:
            return abs(mu)
        else:
            return 0

    def _get_mu(self, Psi, Psi_0):
        """Eq. 25: steady-state mechanical stimulus mu"""
        # eq 38 (osteostat_v1.6.pdf)
        a = (self.D_R * self.D_S * self.S_0) / (self.f_0 * self.P_R)
        b = (self.m * self.C * self.S_0) / self.P_R
        num = self.m * self.S_0 * self.gamma_K * (1 - a) * (Psi - Psi_0)
        den = (
            (1 - a) * (1 + b)
            + self.gamma_K * (Psi_0 + b * Psi)
            + self.gamma_K * a * (Psi - Psi_0)
        )

        return num / den

    def _get_kform_OB(self, f_bm, Ot, mu, mech_only=False):
        """Eq.2: Bone volume fraction formed"""
        if mech_only:
            return self._f_geom(f_bm) * (self.gamma_ob * f_bm * Ot * self._mu_plus(mu))
        else:
            return self._f_geom(f_bm) * (
                self.alpha_ob + self.gamma_ob * f_bm * Ot * self._mu_plus(mu)
            )

    def _get_kres_OC(self, f_bm, Ot, mu, mech_only=False):
        """Eq.3: Bone volume fraction resorbed"""
        if mech_only:
            return self._f_geom(f_bm) * (self.gamma_oc * f_bm * Ot * self._mu_minus(mu))
        else:
            return self._f_geom(f_bm) * (
                self.alpha_oc + self.gamma_oc * f_bm * Ot * self._mu_minus(mu)
            )

    # -------------------------------------------
    #   internal method called by the ode solver
    # -------------------------------------------
    def _g(self, t, y, params):
        """Calculating the derivatives of the ODEs dy / dt = func(t, y)
        t: time to evaluate ODEs
        y: variable vector (f_bm, Psi_0, Ot)
        params: tuple containing vectors for force and corresponding time, as well as events
                e.g. (F, t_force, events)
                F and t_force must be the same length

        returns derivatives for (f_bm, Psi_0, Ot)
        """

        # unpack current values of y
        f_bm, Psi_0, Ot = y

        # unpack parameters
        F, t_force, events = params

        # get current time index
        if t <= t_force[-1]:
            (i,) = np.where(t_force >= t)
        else:
            # TODO: should probably inform the user with a warning
            # if we are out of bounds, return last index
            # this happens sometimes with ode lsoda
            i = np.array([len(t_force) - 1])

        # update attributes according to events
        self._process_events(events, i[0])

        # get the current force
        F_at_t = F[i[0]]

        # get strain energy density
        Psi = self._get_psi(F_at_t, f_bm)

        # get mechanical stimulus
        mu = self._get_mu(Psi, Psi_0)

        # get bone fraction formed
        kform_OB = self._get_kform_OB(f_bm, Ot, mu)

        # get bone fraction resorbed
        kres_OC = self._get_kres_OC(f_bm, Ot, mu)

        # Eq. 5
        dPsi_0 = 1 / f_bm * kform_OB * (Psi - Psi_0)

        # Eq. 1
        df_bm = kform_OB - kres_OC

        # Eq 6
        dOt = 1 / f_bm * kform_OB * (self.Ot_f - Ot) - self.A * Ot

        # list of dy/dt=f functions
        derivs = [df_bm, dPsi_0, dOt]

        return derivs

    # -----------------------------------------
    #   internal method to create results entry
    # -----------------------------------------
    def _add_result(self, key_str, soln, params):
        """Takes an OstesimResult object,
        adds other variables not computed during model solving
        and stores in results dictionary with key_str as key

        Osteostat can have multiple results

        Essentially _g() without derivative calculation

        """

        F, t_force, events = params

        # Create and initialize results object
        result = OsteostatResult(
            t=np.copy(soln.t),
            f_bm=np.copy(soln.y[0, :]),
            Psi_0=np.copy(soln.y[1, :]),
            Ot=np.copy(soln.y[2, :]),
            Psi=np.zeros(len(soln.t)),
            mu=np.zeros(len(soln.t)),
            kform_OB=np.zeros(len(soln.t)),
            kform_OB_mech=np.zeros(len(soln.t)),
            kres_OC=np.zeros(len(soln.t)),
            kres_OC_mech=np.zeros(len(soln.t)),
            Ot_bar=np.zeros(len(soln.t)),
            F=np.zeros(len(soln.t)),
            solver=soln.solver,
            status=soln.status,
            message=soln.message,
        )

        # add additional result vectors for changed attributes
        if events is not None:
            # unique attributes
            event_keys = set([key for d in events.values() for key in d])
            for key in event_keys:
                setattr(result, key, getattr(self, key) * np.ones(len(soln.t)))

        # calculate missing signal points
        for k in tqdm(range(len(soln.t)), file=sys.stdout):
            # get current time index
            if result.t[k] <= t_force[-1]:
                (i,) = np.where(t_force >= result.t[k])
            else:
                # TODO: should probably inform the user with a warning
                # if we are out of bounds, return last index
                # this happens sometimes with ode lsoda
                i = np.array([len(t_force) - 1])

            # update attributes according to events
            self._process_events(events, i[0])

            # update event vector
            if events is not None:
                e = events.get(i[0])
                if e is not None:  # We have an event
                    for key in e:
                        getattr(result, key)[k:] = e[key]

            F_at_t = F[i[0]]

            result.F[k] = F_at_t

            result.Psi[k] = self._get_psi(F_at_t, result.f_bm[k])

            result.mu[k] = self._get_mu(result.Psi[k], result.Psi_0[k])

            result.kform_OB[k] = self._get_kform_OB(
                result.f_bm[k], result.Ot[k], result.mu[k], mech_only=False
            )

            result.kform_OB_mech[k] = self._get_kform_OB(
                result.f_bm[k], result.Ot[k], result.mu[k], mech_only=True
            )

            result.kres_OC[k] = self._get_kres_OC(
                result.f_bm[k], result.Ot[k], result.mu[k], mech_only=False
            )
            # same as above but only mechanical term (for setpoint curve)
            result.kres_OC_mech[k] = self._get_kres_OC(
                result.f_bm[k], result.Ot[k], result.mu[k], mech_only=True
            )

            result.Ot_bar[k] = self.get_Ot_bar(result.f_bm[k])

        # create a new dictionary if there is none yet
        if self.results is None:
            self.results = {key_str: result}
        else:
            self.results[key_str] = result

    # --------------------------------------------------------------
    #   internal methods related to handling parameter change events
    # --------------------------------------------------------------
    def _get_attributes(self, events):
        if events is None:
            return None
        else:
            event_keys = set([key for d in events.values() for key in d])
            # will raise and AttributeError if key is not an attribute of self
            original_attributes = {k: getattr(self, k) for k in event_keys}
            return original_attributes

    def _restore_attributes(self, original_attributes):
        if original_attributes is not None:
            for key, value in original_attributes.items():
                setattr(self, key, value)

    def _process_events(self, events, idx):
        """Changes ostat attributes according to events at index"""
        # check for events
        if events is not None:
            e = events.get(idx)
            if e is not None:  # We have an event
                for key in e:
                    # Ignore entries that are not valid attributes
                    if getattr(self, key, None) is not None:
                        setattr(self, key, e[key])
                        print(f"EVENT: {idx, key, e[key]}")

    # ----------------------------------------
    #   public methods
    # ----------------------------------------
    def get_Ot_bar(self, f_bm0):
        """Calculate osteocyte density steady state for given BV"""
        kfOB_bar = self._f_geom(f_bm0) * self.alpha_ob
        return self.Ot_f / (1 + (f_bm0 * self.A) / kfOB_bar)

    def plot_result(
        self,
        key_str,
        variables,
        compare_results=None,
        axs=None,
        plot_years=True,
        start_idx=None,
        end_idx=None,
        add_legend=True,
    ):
        """Plots desired variables from results with key key_str stored in self.results

        optionally, provide a dict of other_results
        (could be from other Osteostat objects)
        for comparison
        """
        # check if we have results
        if self.results is None:
            return

        # get the result desired
        result = self.results.get(key_str)
        if result is None:
            return

        # check that variables are valid
        for var_str in variables:
            variable = result.get(var_str)
            if variable is None:
                print("Could not find {}".format(var_str))
                print("Options are {}".format(result.keys()))
                return

        # we are using the compare_results dictionary to iterate over later
        if compare_results is None:
            compare_results = dict()
        # add the self result to this dictionary
        compare_results[key_str] = result

        # change time from days to years
        if plot_years:
            t_scale = 1.0 / 365.0
            xlabel_str = "time (years)"
        else:
            t_scale = 1.0
            xlabel_str = "time (days)"

        if start_idx is None:
            start_idx = 0  # set to first index, i.e. plot all

        if end_idx is None:
            end_idx = -1  # set to last index, i.e. plot all

        if axs is None:
            fig = plt.figure(1, figsize=(6, 3 * len(variables)))
            axs = fig.subplots(nrows=len(variables), ncols=1)

        # if single variable desired, only one axis created flatten fails.
        # create a list with single axes as element for loop to work.
        try:
            axs = axs.flatten()
        except AttributeError:
            axs = [axs]

        # assumes a flattened list of axes
        for j, ax in enumerate(axs):

            # iterate over all results to be plotted
            for label, res in compare_results.items():
                variable = res.get(variables[j])
                ax.plot(
                    res.t[start_idx:end_idx] * t_scale,
                    variable[start_idx:end_idx],
                    label=label,
                )
            ax.set_xlabel(xlabel_str)
            ax.set_ylabel(variables[j])
            if add_legend:
                ax.legend()
            ax.grid(visible=True)
        plt.tight_layout()

    def solve(
        self,
        t_eval,
        y0,
        force,
        t_force,
        results_key_str,
        events=None,
        solver="dopri5",
    ):
        """Solve the model and add to results dictionary

        events: (optional) Dict[int:Dict[str:float]]
                A dictionary of index into t_eval
                containint a dictionary with ostat attributes and values
                to set at this time. E.g. {2: {'alpha_ob': 1e-4}}
                Sets ostat.alpha_ob = 1e-4 priot to calculating t_eval[2]
        """

        params = (force, t_force, events)

        # Collect origianl attributes that will be changed
        original_attributes = self._get_attributes(events)

        soln = solve_model(self._g, t_eval, y0, params, solver)

        # _g() called by solve_model may change attributes: Restore here
        self._restore_attributes(original_attributes)

        print(soln.message)

        if soln.status < 0:
            print("something went wrong - not adding result")
        else:

            print("adding results with key {}".format(results_key_str))
            self._add_result(results_key_str, soln, params)

            # add_result may change attributes: Restore here
            self._restore_attributes(original_attributes)


class Mechanostat(Osteostat):
    def __init__(self, f_bm0=0.85, Ot_f=30000, occupancy=0.8) -> None:
        super().__init__(f_bm0, Ot_f, occupancy)
        self.gamma_ob *= 1.8

    def _g(self, t, y, params):
        """Calculating the derivatives of the ODEs dy / dt = func(t, y)
        t: time to evaluate ODEs
        y: variable vector (f_bm, Psi_0, Ot)
        params: tuple containing vectors for force and corresponding time, as well as events
                e.g. (F, t_force, events)
                F and t_force must be the same length

        returns derivatives for (f_bm, Psi_0, Ot)
        """
        df_bm, _, _ = super()._g(t, y, params)
        derivs = [df_bm, 0, 0]

        return derivs


def solve_model(slope_func, t_eval, y0, params, solver="dopri5"):
    """wrapper to call scipy ode solvers with either solve_ivp() or ode() interface

    This function is not specific to osteostat

    slope_func: Osteostat._g(t, y, params)
    t_eval: vector of times to be evaluated
    y0: initial conditions (f_bm, S, R, Psi_0, Ot)
    params: [force, t_force]
    solver: see `supported_ivp_solvers` and `supported_ode_solvers`

    returns an OsteosimResult object

    """
    # solve_ivp solvers
    if solver in supported_ivp_solvers:
        print("using solve_ivp() with {}".format(solver))
        # start timers
        t1_start = time.perf_counter()
        t2_start = time.process_time()

        # call solver
        soln = solve_ivp(
            lambda t, y: slope_func(t, y, params),
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method=solver,
            rtol=1e-6,
            atol=1e-12,
        )
        t = soln.t
        y = soln.y
        status = soln.status
        message = soln.message

        # stop timers
        t1_stop = time.perf_counter()
        t2_stop = time.process_time()
        m1, s1 = divmod(t1_stop - t1_start, 60)
        m2, s2 = divmod(t2_stop - t2_start, 60)
        print("Elapsed time: {:02.0f}:{:06.3f}".format(m1, s1))
        print("CPU process time: {:02.0f}:{:06.3f}".format(m2, s2))

    elif solver in supported_ode_solvers:
        print("using ode() with {}".format(solver))
        # total number of steps
        num_steps = len(t_eval)

        # Set the ODE solver
        r = ode(slope_func).set_integrator(solver)
        r.set_initial_value(y0, t_eval[0]).set_f_params(params)

        # prepare output vectors
        soln = np.zeros((len(y0), len(t_eval)))
        # initialize
        soln[:, 0] = y0
        k = 1

        # TODO: tqdm does not behave well when interrupted, call in try-except
        pbar = tqdm(total=num_steps, file=sys.stdout)
        while r.successful() and r.t < t_eval[-1]:
            r.integrate(t_eval[k])
            soln[:, k] = r.y
            k += 1
            pbar.update()
        pbar.close()

        t = t_eval
        y = soln
        status = r.get_return_code()
        if status > 0:
            message = "Success"
        else:
            message = "Something went wrong, check documentation on status."

    else:
        print("Error: solver {} not found in supported solvers".format(solver))
        print(
            "options are {} (solve_ivp()) or {} (ode())".format(
                supported_ivp_solvers, supported_ode_solvers
            )
        )
        # TODO: should probably throw an exception here
        return None

    return OsteosimResult(t=t, y=y, solver=solver, status=status, message=message)


def get_default_initial_conditions(ostat, F0, f_bm0):

    # initial mechanical set point = Psi(t=0) from Eq. 35
    Psi_00 = 1 / 2.0 * (ostat.c_bm * (F0 / (ostat.L2 * f_bm0)) ** 2)

    # inital osteocyte density per BV
    Ot0 = ostat.get_Ot_bar(f_bm0)

    # Bundle initial conditions for ODE solver
    y0 = [f_bm0, Psi_00, Ot0]
    return y0


def get_day_index(day, time_inc):
    """Given a time increment, get the index of a specific day"""
    return int(day * 1 / time_inc)


def get_time_vectors(t_stop, t_inc, t_inc_force):
    # t_eval is in days
    t_eval = np.arange(0.0, t_stop + t_inc, t_inc)

    # time vector to define force
    t_force = np.arange(0.0, t_stop + t_inc_force, t_inc_force)
    return t_eval, t_force


def get_drop_force_vector(
    F0: float, F_drop: float, idx_up: int, force_length: int, n_ramp: int
):
    """
          n_ramp      n_ramp
    F0       \\          //-------
              \\        //
    F_drop     \\______//
                   idx_up

    """

    assert F0 > F_drop, f"initial force {F0} not larger than drop force {F_drop}"
    assert (
        force_length > 2 * n_ramp
    ), f"length {force_length} not larger than twice n_ramp"
    assert idx_up < force_length, f"idx_up needs to be within [0, {force_length})"

    def create_ramp(start_val, end_val, n):
        if start_val > end_val:
            # down ramp
            s = 1
            e = 0
            level = end_val
        else:
            # up ramp
            s = 0
            e = 1
            level = start_val

        return abs(start_val - end_val) * np.linspace(s, e, n) + level

    # drop load
    force_drop = F_drop * np.ones(force_length)
    # ramp down
    force_drop[:n_ramp] = create_ramp(start_val=F0, end_val=F_drop, n=n_ramp)
    # ramp up
    force_drop[idx_up : idx_up + n_ramp] = create_ramp(
        start_val=F_drop, end_val=F0, n=n_ramp
    )
    # full load
    force_drop[idx_up + n_ramp :] = F0
    return force_drop


def configure_plotting_for_paper():
    # reference: https://github.com/maksimt/empirical_privacy/blob/master/src/notebook_context/__init__.py
    cm = 1 / 2.54
    # /textwidth is 16cm, one figure is half of this, slightly smaller
    figsize = (7.8 * cm, 6 * cm)  # width, height

    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["figure.dpi"] = 140
    mpl.rcParams["savefig.dpi"] = 300
    sns.set()
    sns.set_style("whitegrid")
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"
    sns.set_context("paper", font_scale=3.0, rc=mpl.rcParams)

    return figsize
