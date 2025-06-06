# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt
from scipy.optimize import differential_evolution
from numba import jit, float64, prange
import numpy as np
from argparse import ArgumentParser
import csv
import sys

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=1, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-n', default='Super1', type=str, required=False, help='node=Super1')
parser.add_argument('-w', default=1, type=int, required=False, help='Number of islands in differential evolution (i.e. workers)')
parser.add_argument('-steps', default=1, type=int, required=False, help='Number of steps in capacity expansion')
args = parser.parse_args()

scenario = args.s
node = args.n
steps = args.steps

from Input import *
from Simulation import Reliability
from Network import Transmission

def check_limits():   
    test = np.array([PVBuildRateLimit] * pzones + [WindBuildRateLimit]  * wzones + [50.] * nodes * steps + [50.], dtype=np.float64)

    S = Solution(test)

    Deficit = Reliability(S, flexible=np.ones(intervals, dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    Deficit_sum = Deficit.sum() * resolution
    if Deficit_sum > 0: 
        print('Not possible to match load  with current build limits')
        print('Ending Optimisation...')
        sys.exit(0)
    else: 
        print('Optimisation possible with specified build limits and growth multiplier')
        return


def get_initial_guess():
    while True:
        response = input("Would you like to import the results of the previous optimisation as the initial guess for this run? All variables should be kept the same as previous (y/n): ").strip().lower()
        if response in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'.")

    if response == 'y':
        print("Importing previous as initial guess")
        try:
            with open('Results/Optimisation_resultx{}{}.csv'.format(args.n, args.i), 'r') as f:
                lines = f.readlines()
                last_line = lines[-1].strip()
                initial_guess = [float(x) for x in last_line.split(',')]  
        except Exception as e:
            print(f"An error occurred: {e}")
            initial_guess = None
    else:
        print("Continuing. Initial guess is null")
        initial_guess = None

    return initial_guess

@jit(parallel=True)
def parallel_object_wrapper(xs):
    result = np.empty(xs.shape[1], dtype=np.float64)
    for i in prange(xs.shape[1]):
        result[i] = F(xs[:,i])
    return result

@jit(nopython=True)
def F(x):
    """This is the objective function."""

    #Objective Function starts here
    S = Solution(x)

    Deficit = Reliability(S, flexible=np.zeros(intervals, dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    Deficit = Reliability(S, flexible=np.ones(intervals, dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    Deficit_sum = Deficit.sum() * resolution
    PenDeficit = max(0, Deficit_sum) # MWh


    TDC = Transmission(S) if 'Super' in node else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW
    TDC_abs = np.abs(TDC)

    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for i in range(0,intervals):
        for j in range(0,len(DCloss)):
            if TDC_abs[i][j] > CDC[j]:
                CDC[j] = TDC_abs[i][j]
    CDC = CDC * 0.001 # CDC(k), MW to GW


    cost = factor *  np.concatenate((np.array([S.CPV[int(len(S.CPV) * (1 - 1/steps)):int(len(S.CPV))].sum(), S.CWind[int(len(S.CWind) * (1 - 1/steps)):int(len(S.CWind))].sum(), S.CPHP[int(len(S.CPHP) * (1 - 1/steps)):int(len(S.CPHP))].sum(), S.CPHS]), CDC, np.array([S.CPV[int(len(S.CPV) * (1 - 1/steps)):int(len(S.CPV))].sum(), S.CWind[int(len(S.CWind) * (1 - 1/steps)):int(len(S.CWind))].sum(), Hydro * 0.000001, -1.0, -1.0])))
    cost = cost.sum()

    loss = TDC_abs.sum(axis=0) * DCloss
    loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss)

    Func = LCOE + PenDeficit + PenHydro 
    
    return Func


def main():
    initial_guess = get_initial_guess() #Confirm whether to use results of previous optimisation

    check_limits() #Check if optimisation is possible under current build limits

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
 
    lb = []
    ub = []

    lb = [0.]  * pzones + [0.]   * wzones + contingency * steps + [0.]
    ub = [PVBuildRateLimit] * pzones + [WindBuildRateLimit]  * wzones + [50.] * (nodes * steps) + [5000.]

    result = differential_evolution(
        x0=initial_guess, #The idea of an intial guess is fun. Maybe add back later. Initial guess starts with result from last run
        func=parallel_object_wrapper, 
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=(0.2,args.m), 
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred',
        callback=None, 
        workers=1,
        vectorized=True,
        )

    # Print the best solution and its objective function value
    print("Best solution:", result.x)
    print("Value of the objective function:", result.fun)

    with open('Results/Optimisation_resultx{}{}.csv'.format(args.n, args.i), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    if args.steps > 1:
        with open('Results/LCOE_resultx{}{}.csv'.format(args.n, args.i), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([result.fun])

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

if __name__=='__main__':
    main()
