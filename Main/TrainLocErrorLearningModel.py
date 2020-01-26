'''
Train the error mapping model.
'''
import sys,os
script_path = os.path.abspath(sys.argv[0])
proj_path = os.path.join('/', *script_path.split('/')[:-2])
sys.path.append(proj_path)
from Training.TrainingManager import TrainingManager
from Training.ErrorModelLearner import ErrorModelLearner
from Dataset.DatasetLoader.generateLocalizationDataSet import config_loc_dataset

if __name__ == '__main__':
    tm = TrainingManager()
    # Step 1, set data
    dataset = config_loc_dataset(tm.configuration, 'train')

    tm.set_data(*dataset)

    # Step 2, set learner
    learner = ErrorModelLearner()
    tm.set_learner(learner)

    # Step 3, run training
    tm.train()
