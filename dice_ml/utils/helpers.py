"""
This module containts helper functions to load data and get meta deta.
"""
import numpy as np
import pandas as pd
import shutil
import os

import dice_ml

# for data transformations
import sklearn
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

def load_adult_income_dataset(only_train=True):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', delimiter=', ', dtype=str)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)


    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                           'Exec-managerial':'White-Collar','Farming-fishing':'Blue-Collar',
                                            'Handlers-cleaners':'Blue-Collar',
                                            'Machine-op-inspct':'Blue-Collar','Other-service':'Service',
                                            'Priv-house-serv':'Service',
                                           'Prof-specialty':'Professional','Protective-serv':'Service',
                                            'Tech-support':'Service',
                                           'Transport-moving':'Blue-Collar','Unknown':'Other/Unknown',
                                            'Armed-Forces':'Other/Unknown','?':'Other/Unknown'}})

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married','Never-married':'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                           'Amer-Indian-Eskimo':'Other'}})

    adult_data = adult_data[['age','workclass','education','marital-status','occupation','race','gender',
                     'hours-per-week','income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                           '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                          '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if only_train:
        train, _ = train_test_split(adult_data, test_size=0.2, random_state=17)
        adult_data = train.reset_index(drop=True)

    # Remove the downloaded dataset
    if os.path.isdir('archive.ics.uci.edu'):
        entire_path = os.path.abspath('archive.ics.uci.edu')
        shutil.rmtree(entire_path)

    return adult_data

def load_custom_testing_dataset():
    data = [['a', 10, 0], ['b', 10000, 0], ['c', 14, 0], ['a', 88, 0], ['c', 14, 0]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])

def load_custom_testing_dataset_binary():
    data = [['a', 1, 0], ['b', 5, 1], ['c', 2, 0], ['a', 3, 0], ['c', 4, 1]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])

def load_custom_testing_dataset_multiclass():
    data = [['a', 10, 1], ['b', 20, 2], ['c', 14, 1], ['a', 23, 2], ['c', 7, 0]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])

def load_custom_testing_dataset_regression():
    data = [['a', 10, 1], ['b', 21, 2.1], ['c', 14, 1.4], ['a', 23, 2.3], ['c', 7, 0.7]]
    return pd.DataFrame(data, columns=['Categorical', 'Numerical', 'Outcome'])

def get_adult_income_modelpath(backend='TF1'):
    pkg_path = dice_ml.__path__[0]
    model_ext = '.h5' if 'TF' in backend else '.pth'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'adult'+model_ext)
    return modelpath

def get_custom_dataset_modelpath_pipeline():
    pkg_path = dice_ml.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom'+model_ext)
    return modelpath

def get_custom_dataset_modelpath_pipeline_binary():
    pkg_path = dice_ml.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_binary'+model_ext)
    return modelpath

def get_custom_dataset_modelpath_pipeline_multiclass():
    pkg_path = dice_ml.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_multiclass'+model_ext)
    return modelpath

def get_custom_dataset_modelpath_pipeline_regression():
    pkg_path = dice_ml.__path__[0]
    model_ext = '.sav'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'custom_regression'+model_ext)
    return modelpath

def get_adult_data_info():
    feature_description = {'age':'age',
                        'workclass': 'type of industry (Government, Other/Unknown, Private, Self-Employed)',
                        'education': 'education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)',
                        'marital_status': 'marital status (Divorced, Married, Separated, Single, Widowed)',
                        'occupation': 'occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)',
                        'race': 'white or other race?',
                        'gender': 'male or female?',
                        'hours_per_week': 'total work hours per week',
                        'income': '0 (<=50K) vs 1 (>50K)'}
    return feature_description

def get_base_gen_cf_initialization(data_interface, encoded_size, cont_minx, cont_maxx, margin, validity_reg, epochs, wm1, wm2, wm3, learning_rate ):

    #Dice Imports - TODO: keep this method for VAE as a spearate module or move it to feasible_base_vae.py. Check dependencies.
    from dice_ml.utils.sample_architecture.vae_model import CF_VAE

    #Pytorch
    import torch
    import torch.utils.data
    from torch import nn, optim
    from torch.nn import functional as F
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from torch.autograd import Variable

    # Dataset for training Variational Encoder Decoder model for CF Generation
    df = data_interface.normalize_data(data_interface.one_hot_encoded_data)
    encoded_data= df[data_interface.ohe_encoded_feature_names + [data_interface.outcome_name]]
    dataset = encoded_data.to_numpy()
    print('Dataset Shape:',  encoded_data.shape)
    print('Datasets Columns:', encoded_data.columns)

    #Normalise_Weights
    normalise_weights={}
    for idx in range(len(cont_minx)):
        _max= cont_maxx[idx]
        _min= cont_minx[idx]
        normalise_weights[idx]=[_min, _max]

    #Train, Val, Test Splits
    np.random.shuffle(dataset)
    test_fraction =0.2
    test_size= int(test_fraction*len(data_interface.data_df)) #TODO:create an input parameter for data interface
    vae_test_dataset= dataset[:test_size]
    dataset= dataset[test_size:]
    vae_val_dataset= dataset[:test_size]
    vae_train_dataset= dataset[test_size:]

    #BaseGenCF Model
    cf_vae = CF_VAE(data_interface, encoded_size)

    #Optimizer
    cf_vae_optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
        {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()),'weight_decay': wm2},
        {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3},
        ], lr=learning_rate
    )

    # Check: If base_obj was passsed via reference and it mutable; might not need to have a return value at all
    return vae_train_dataset, vae_val_dataset, vae_test_dataset, normalise_weights, cf_vae, cf_vae_optimizer

def ohe_min_max_transformation(data, data_interface):
    """the data is one-hot-encoded and min-max normalized and fed to the ML model"""
    return data_interface.get_ohe_min_max_normalized_data(data).values

class DataTransfomer:
    """A class to transform data based on user-defined function to get predicted outcomes. This class calls FunctionTransformer of scikit-learn internally (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)."""

    def __init__(self, func=None, kw_args=None):
        self.func = func
        self.kw_args = kw_args

    def feed_data_params(self, data_interface):
        if self.kw_args is not None:
            self.kw_args['data_interface'] = data_interface
        else:
            self.kw_args = {'data_interface': data_interface}

    def initialize_transform_func(self):
        if self.func == 'ohe-min-max':
            self.data_transformer = FunctionTransformer(func=ohe_min_max_transformation, kw_args=self.kw_args, validate=False)
        elif self.func == None: # identity transformation
            self.data_transformer = FunctionTransformer(func=self.func, kw_args=None, validate=False) # add more ready-to-use transformers (such as label-encoding) in elif loops.
        else:
            self.data_transformer = FunctionTransformer(func=self.func, kw_args=self.kw_args, validate=False) # add more ready-to-use transformers (such as label-encoding) in elif loops.

    def transform(self, data):
        return self.data_transformer.transform(data) # should return a numpy array

    def inverse_transform(self, data):
        return self.data_transformer.inverse_transform(data) # should return a numpy array
