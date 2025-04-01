import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from argparse import ArgumentParser
import csv
import importlib
import datetime as dt

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
runCount = 0

from Input import *
from Simulation import Reliability
from Network import Transmission

def growth_constaint(x):
    num_diffs = int(len(PVl) + len(Windl) + 2)
    diffs = np.empty(num_diffs, dtype=x.dtype) 
    for i in range(num_diffs):
        diffs[i] = x[(num_diffs)+i] - x[i]
    return diffs


def F(x):
    """This is the objective function."""
    Func = 0
    length = int(len(PVl) + len(Windl) + 2)

    #Calculate cost to build
    #build_cost = ( (x[:pidx]).sum() * 10 + (x[pidx: widx]).sum() * 10
    #           + (x[17: pidx+17]).sum() * 20 + (x[pidx+17:widx+17]).sum() * 20 ) 
    
    #Calculate LCOE
    for i in range(2):
        S = Solution(x[i*length:length*(i+1)])
        demand_multiplier = 1+i

        S.MLoad = MLoad * demand_multiplier

        Deficit = Reliability(S, flexible=np.zeros(intervals, dtype=np.float64)) # Sj-EDE(t, j), MW
        Flexible = Deficit.sum() * resolution / years / efficiency # MWh p.a.
        Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
        PenHydro = max(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

        TDC = Transmission(S) if 'Super' in node else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW
        TDC_abs = np.abs(TDC)

        Deficit = Reliability(S, flexible=np.ones(intervals, dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
        Deficit_sum = Deficit.sum() * resolution
        PenDeficit = max(0, Deficit_sum) # MWh

        CDC = np.zeros(len(DCloss), dtype=np.float64)
        for i in range(0,intervals):
            for j in range(0,len(DCloss)):
                if TDC_abs[i][j] > CDC[j]:
                    CDC[j] = TDC_abs[i][j]
        CDC = CDC * 0.001 # CDC(k), MW to GW

        cost = factor *  np.concatenate((np.array([S.CPV.sum(), S.CWind.sum(), S.CPHP.sum(), S.CPHS]), CDC, np.array([S.CPV.sum(), S.CWind.sum(), Hydro * 0.000001, -1.0, -1.0])))
        cost = cost.sum()

        loss = TDC_abs.sum(axis=0) * DCloss
        loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
        LCOE = cost / abs(energy - loss)

        Func = Func + LCOE + PenDeficit + PenHydro
  

    return Func

class GrowthConstraint:
    def __init__(self):
        self.lb = np.zeros(int(len(PVl) + len(Windl) + 2))
        self.ub = np.full(int(len(PVl) + len(Windl) + 2), np.inf)

    def __call__(self, x):
        return growth_constaint(x)
    
balancing_constraint = GrowthConstraint() 
monotonicity_growth_constraint = NonlinearConstraint(
                                fun=balancing_constraint,
                                lb=balancing_constraint.lb,
                                ub=balancing_constraint.ub,
                            )

PVBuildRateLimit = 5 #GW/step/generator
WindBuildRateLimit = 5 #GW/year/generator

lb = ([0.]  * pzones + [0.]   * wzones + contingency   + [0.])*2
ub = ([PVBuildRateLimit] * pzones + [WindBuildRateLimit]  * wzones + [50.] * nodes + [5000.] +
        [PVBuildRateLimit*2] * pzones + [WindBuildRateLimit*2]  * wzones + [50.] * nodes + [5000.])    


def main():

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    result = differential_evolution(
            func=F, 
            bounds=list(zip(lb, ub)), 
            constraints=(monotonicity_growth_constraint,), 
            tol=0,
            maxiter=args.i, 
            popsize=args.p, 
            mutation=args.m, 
            recombination=args.r,
            disp=True, 
            polish=False, 
            updating='deferred',
            workers=-1,
            vectorized=False,
            )

    with open('Results/Optimisation_resultx{}{}.csv'.format(args.n, args.i), 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([result.fun])
            writer.writerow(result.x[0:int(len(result.x)/2)])
            writer.writerow(result.x[int(len(result.x)/2):len(result.x)])

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    return result

if __name__ == "__main__":
    main()