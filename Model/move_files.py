""" Temporary file to move files around """

import numpy as np
# import sys
import os
import glob

if __name__ == '__main__':

    # Relevant directories
    dir1 = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Model\\Output\\"
    dir2 = dir1 + "StraightLine_PeriodicFlow\\BendingElasticity_Clamped_VaryingBendingViscosity\\MoreData\\"
    dir3 = dir1 + "StraightLine_PeriodicFlow\\ShearElasticity_Clamped_VaryingShearViscosity\\"

    # Get relevant files from dir1
    dir12_files = glob.glob(dir1 + "*bool_EI_True*")
    # Move files from dir1 to dir2
    for file in dir12_files:
        newfile = file.replace(dir1, dir2)
        # os.replace(file, newfile)

    # Get relevant files from dir3
    dir13_files = glob.glob(dir1 + "*bool_EI_False*")
    # Move files from dir1 to dir3
    for file in dir13_files:
        newfile = file.replace(dir1, dir3)
        # os.replace(file, newfile)        

    

