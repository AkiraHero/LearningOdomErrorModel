'''
 Find the best scale for cov from traditional method
'''

import sys,os
script_path = os.path.abspath(sys.argv[0])
proj_path = os.path.join('/', *script_path.split('/')[:-2])
sys.path.append(proj_path)
from Test.TestManager import TestManager
from Training.ErrorModelLearner import ErrorModelLearner
from Dataset.DatasetLoader.generateLocalizationDataSet import config_loc_dataset

if __name__ == '__main__':
    tm = TestManager()
    # Step 1, set data
    dataset = config_loc_dataset(tm.configuration, 'test-acc')

    tm.set_data(*dataset)

    # Step 2, run testing
    scalelist = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    tm.find_best_covscale(scalelist)
