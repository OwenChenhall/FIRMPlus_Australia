#### SUDO CODE ####
# Code will take existing inputs plus a current network model as inputs
# call optimization.py main function to compute optimal capacities
# return capacities as .cvs file
# update network model with new capacities to assume they have been built
# rerun optimization.py main function
# repeat as neccesary
# finally return LCOE value and write capacities to .csv file

# test before running optimisation on max build rates

from argparse import ArgumentParser
import datetime
import Optimisation
import Input
import importlib
from Optimisation import args

starttime = datetime.datetime.now()

Optimisation.main()


for x in range(args.steps - 1):
    importlib.reload(Input)
    importlib.reload(Optimisation)
    Optimisation.main()

endtime = datetime.datetime.now()
print("Overal Expansion took", endtime - starttime)
