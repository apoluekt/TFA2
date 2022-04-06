# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import amplitf.interface as atfi

from iminuit import Minuit

from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from timeit import default_timer as timer


class FitParameter:
    def __init__(self, name, init_value, lower_limit, upper_limit, step_size=1e-6):
        self.var = ResourceVariable(
            init_value, shape=(), name=name, dtype=atfi.fptype(), trainable=True
        )
        self.init_value = init_value
        self.name = name
        self.step_size = step_size
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.prev_value = None
        self.fixed = False
        self.error = 0.0
        self.positive_error = 0.0
        self.negative_error = 0.0
        self.fitted_value = init_value

    def update(self, value):
        if value != self.prev_value:
            self.var.assign(value)
            self.prev_value = value

    def __call__(self):
        return self.var

    def fix(self):
        self.fixed = True

    def float(self):
        self.fixed = False

    def setFixed(self, fixed):
        self.fixed = fixed

    def floating(self):
        """
        Return True if the parameter is floating and step size>0
        """
        return self.step_size > 0 and not self.fixed

    def numpy(self):
        return self.var.numpy()


def run_minuit(nll, pars, use_gradient=True, use_hesse = False, use_minos = False, get_covariance = False, print_level = 0):
    """
    Run IMinuit to minimise NLL function

    nll  : python callable representing the negative log likelihood to be minimised
    pars : list of FitParameters
    use_gradient : if True, use analytic gradient
    use_hesse : if True, uses HESSE for error estimation
    use_minos : if True, use MINOS for asymmetric error estimation
    get_covariance: if True, get the covariance matrix from the fit

    returns the dictionary with the values and errors of the fit parameters
    """

    float_pars = [p for p in pars if p.floating()]
    fixed_pars = [p for p in pars if not p.floating()]

    def func(par):
        for i, p in enumerate(float_pars):
            p.update(par[i])
        kwargs = {p.name: p() for p in float_pars + fixed_pars}
        func.n += 1
        nll_val = nll(kwargs).numpy()
        if func.n % 10 == 0:
            print(func.n, nll_val, par)
        return nll_val

    def gradient(par):
        for i, p in enumerate(float_pars):
            p.update(par[i])
        kwargs = {p.name: p() for p in float_pars + fixed_pars}
        float_vars = [i() for i in float_pars]
        gradient.n += 1
        with tf.GradientTape() as gt:
            gt.watch(float_vars)
            nll_val = nll(kwargs)
        g = gt.gradient(
            nll_val, float_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        g_val = [i.numpy() for i in g]
        return g_val

    func.n = 0
    gradient.n = 0

    start = [p.init_value for p in float_pars]
    error = [p.step_size for p in float_pars]
    limit = [(p.lower_limit, p.upper_limit) for p in float_pars]
    name = [p.name for p in float_pars]

    if use_gradient:
        minuit = Minuit(func,start,grad=gradient,name=name)
    else:
        minuit = Minuit(func,start,name=name)

    minuit.errordef=Minuit.LIKELIHOOD
    minuit.errors = error
    minuit.limits = limit

    initlh = func(start)
    starttime = timer()
    minuit.migrad()
    if use_hesse:
        minuit.hesse()

    if use_minos:
        minuit.minos()

    minuit.print_level = print_level

    endtime = timer()

    par_states = minuit.params
    f_min = minuit.fmin
    #print the nice tables of fit results
    print(f_min)
    print(par_states)
    if f_min.is_valid: print(minuit.covariance.correlation())

    results = {"params": {}}  # Get fit results and update parameters
    for n, p in enumerate(float_pars):
        p.update(par_states[n].value)
        p.fitted_value = par_states[n].value
        p.error = par_states[n].error
        results["params"][p.name] = (p.fitted_value, p.error)
    for p in fixed_pars:
        results["params"][p.name] = (p.numpy(), 0.0)

    # return fit results
    results["initlh"] = initlh
    results["loglh"] = f_min.fval
    results["iterations"] = f_min.nfcn
    results["func_calls"] = func.n
    results["grad_calls"] = gradient.n
    results["time"] = endtime - starttime
    #results["covariance"] = [(k, v) for k, v in minuit.covariance.items()]
    #is_valid == (has_valid_parameters & !has_reached_call_limit & !is_above_max_edm)
    results["is_valid"] = int(f_min.is_valid) 
    results["has_parameters_at_limit"] = int(f_min.has_parameters_at_limit)
    results["has_accurate_covar"] = int(f_min.has_accurate_covar)
    results["has_posdef_covar"] = int(f_min.has_posdef_covar)
    results["has_made_posdef_covar"] = int(f_min.has_made_posdef_covar)
    results["has_reached_call_limit"] = int(f_min.has_reached_call_limit)

    #store covariance matrix of parameters
    if get_covariance:
        covarmatrix = {}
        for p1 in float_pars:
            covarmatrix[p1.name] = {}
            for p2 in float_pars:
                covarmatrix[p1.name][p2.name] = minuit.covariance[p1.name, p2.name]

        results["covmatrix"] = covarmatrix

    return results


def calculate_fit_fractions(pdf, norm_sample):
    """
    Calculate fit fractions for PDF components
      pdf : PDF function or two parameters:
      norm_sample : normalisation sample.

    PDF should be the function of 2 parameter:
      pdf(x, switches = 4*[1])
    where the 1st positional parameter is the data sample,
    and the named parameter "switches" should default to the
    list of "1". The size of the list defines the number of components.
    """
    import inspect

    args, varargs, keywords, defaults = inspect.getargspec(pdf)
    num_switches = 0
    if defaults:
        default_dict = dict(zip(args[-len(defaults) :], defaults))
        if "switches" in default_dict:
            num_switches = len(default_dict["switches"])

    @tf.function
    def pdf_components(d):
        result = []
        for i in range(num_switches):
            switches = num_switches * [0]
            switches[i] = 1
            result += [pdf(d, switches=switches)]
        return result

    total_int = atfi.reduce_sum(pdf(norm_sample))
    return [
        (atfi.reduce_sum(i) / total_int).numpy() for i in pdf_components(norm_sample)
    ]


def write_fit_results(pars, results, filename, store_covariance = False):
    """
    Write the dictionary of fit results to text file
      pars     : list of FitParameter objects
      results  : fit results as returned by MinuitFit
      filename : file name
    """
    f = open(filename, "w")
    for p in pars:
        if not p.name in results["params"]:
            continue
        s = "%s " % p.name
        for i in results["params"][p.name]:
            s += "%f " % i
        f.write(s + "\n")
    s = "loglh %f %f" % (results["loglh"], results["initlh"])
    f.write(s + "\n")
    f.close()

    if store_covariance:
        covmatrix = results['covmatrix']
        fcov = open(filename.replace('.txt','_cov.txt'), "w")
        for k1 in list(covmatrix.keys()):
            for k2 in list(covmatrix[k1].keys()):
                s = "%s %s %f" % (k1, k2, covmatrix[k1][k2])
                fcov.write(s + "\n")
        fcov.close()
