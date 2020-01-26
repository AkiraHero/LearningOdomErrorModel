'''
 Test trained network by comparing localization accuracy
'''

import os
import sys

script_path = os.path.abspath(sys.argv[0])
proj_path = os.path.join('/', *script_path.split('/')[:-2])
sys.path.append(proj_path)
from Test.TestManager import TestManager
from Training.ErrorModelLearner import ErrorModelLearner
from Dataset.DatasetLoader.generateLocalizationDataSet import config_loc_dataset

if __name__ == '__main__':
    tm = TestManager()
    # Step 1, set data
    dataset = config_loc_dataset(tm.configuration, 'test-pre', allfrm=True)

    tm.set_data(*dataset)

    # Step 2, set learner
    learner = ErrorModelLearner()
    tm.set_learner(learner)

    # Step 3, run testing
    fileout = tm.model_prediction()

    infofile = fileout + '.info.csv'

    tm.transNetPrediction2Mat(fileout, infofile, type='info')

    # Step 1, set data
    dataset = config_loc_dataset(tm.configuration, 'test-acc', modelprediction_infomatfile=infofile)

    tm.set_data(*dataset)

    # Step 2, set learner, get setting again
    tm.set_learner(learner)

    # Step 3, run testing
    tm.compare_accuracy(gen_sampletraj=1, withNetPre=True)
