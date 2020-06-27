#!/usr/bin/env python
# Note: might need to add the model directory to PYTHONPATH, 
# or just move this file to the grid_cell_model dir


from default_params import defaultParameters
from common         import GenericSubmitter, ArgumentCreator

import logging as lg
lg.basicConfig(level=lg.DEBUG)


parameters = defaultParameters

parameters['time']              = 1293e3    # ms
parameters['ndumps']            = 1

parameters['placeT']            = 10e3      # ms

parameters['stateMonDur']       = 10e3

parameters['bumpCurrentSlope']  = 1.175     # pA/(cm/s), !! this will depend on prefDirC !!
parameters['gridSep']           = 60        # cm, grid field inter-peak distance

parameters['ratVelFName'] 	= './kg_trajectory.mat'

startJobNum = 0
numRepeat = 1

# Workstation parameters
programName         = 'python simulation_basic_grids.py'
blocking            = True

ac = ArgumentCreator(parameters)
submitter = GenericSubmitter(ac, programName, blocking=blocking)
submitter.submitAll(startJobNum, numRepeat, dry_run=False)
