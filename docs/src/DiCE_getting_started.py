## This file shows how to generate Diverse Counterfactual Explanations.

## python libraries
import numpy as np
import pandas as pd
import os

import dice_ml
from dice_ml import dice # dice interface
from dice_ml import data # data interface
from dice_ml import model # model interface
from dice_ml import utils
from dice_ml.utils import helpers

# loading sample dataset (adult-income:https://archive.ics.uci.edu/ml/datasets/adult )
#dataset = pd.read_csv(os.path.join('dice_ml', 'utils', 'sample_data', 'adult.csv'))
dataset = helpers.load_adult_income_dataset()

# a data interface to access all required information about the data for DiCE
d = data.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')

# loading trained ML model
pkg_path = dice_ml.__path__[0]
m = model.Model(model_path= os.path.join(pkg_path, 'utils', 'sample_trained_models', 'adult.h5'))

# initiating DiCE
exp = dice.Dice(d, m)

# generating counterfactuals
test_instance = [22, 'Private', 'HS-grad', 'Single', 'Service', 'White', 'Female', 45]
dice_exp = exp.generate_counterfactuals(test_instance, total_CFs=4, desired_class="opposite")

# printing counterfactuals
dice_exp.visualize_as_list()
