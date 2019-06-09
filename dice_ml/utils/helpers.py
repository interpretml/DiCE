"""
This file containts all helper functions for plotting, printing results etc.
"""
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras

import data # should be from dice import data
import model # should be from dice import model

def load_compas():
    dataset = pd.read_csv('utils'+ os.sep + 'sample_data' + os.sep +'compas.csv')
    dataset = dataset[['priors_count', 'sex', 'race', 'age_cat', 'c_charge_degree', 'two_year_recid']]

    feature_description = {'priors_count': 'Number of prior criminal records',
                            'sex':'Male/Female',
                            'race': 'African-American/Caucasian/Hispanic/Asian/Native American/Other)',
                            'age_cat': '<25/25-45/>45',
                            'c_charge_degree': 'Felony(F)/Misdemeanor(M)',
                            'two_year_recid': 'Will commit another crime within 2 years?'}

    d = data.Data(dataframe=dataset, continuous_feature_indexes=[0], feature_names=['priors_count', 'sex', 'race', 'age_cat', 'c_charge_degree'],target_name='two_year_recid',feature_description=feature_description)

    pre_trained_model = keras.models.load_model('utils'+ os.sep + 'sample_trained_models' + os.sep + 'compas.h5')
    m = model.Model(pre_trained_model)

    return d, m

def load_adult():
    dataset = pd.read_csv('sample_data' + os.sep + 'adult.csv')
    dataset = dataset[['age', 'hours_per_week', 'education', 'workclass', 'marital_status', 'occupation', 'race', 'sex', 'income']] # remove this and change the column orders in .csv itself

    feature_description = {'age':'age',
                            'hours_per_week': 'total work hours per week',
                            'education': 'education level',
                            'workclass': 'type of industry',
                            'marital_status': 'marital status',
                            'occupation': 'occupation',
                            'race': 'white or other race?',
                            'sex': 'male or female?'}

    d = Data(dataset=dataset, continuous_vars_indexes=[0,1], feature_names=['age', 'hours_per_week', 'education', 'workclass', 'marital_status', 'occupation', 'race', 'sex'], target_name='income',feature_description=feature_description)

    pre_trained_model = keras.models.load_model('sample_trained_models' + os.sep + 'adult.h5')
    m = Model(pre_trained_model)

    return d, m

def load_lending():
    dataset = pd.read_csv('sample_data' + os.sep + 'lending.csv')
    dataset = dataset[['emp_length', 'annual_inc', 'open_acc', 'credit_years', 'grade', 'home_ownership', 'purpose', 'addr_state', 'loan_status']]

    feature_description = {'emp_length':'Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years',
                            'annual_inc': 'The self-reported annual income provided by the borrower during registration',
                            'open_acc': 'The number of open credit lines in the borrower credit file',
                            'credit_years': 'how many years old is the credit history of a borrower?',
                            'grade': 'LC assigned loan grade',
                            'home_ownership': 'The home ownership status provided by the borrower during registration',
                            'purpose': 'A category provided by the borrower for the loan request',
                            'addr_state': 'The state where loan was provided - either one of low_default regions-["IL" "TX" "CA" "GA"] or one of high_default regions-["FL" "NY"] or Other states',
                            'loan_status': '0: default, 1: fully paid' }

    column_names = dataset.columns.tolist()
    feature_names = [i for i in column_names if i!='loan_status']

    d = Data(dataset=dataset, continuous_vars_indexes=[i for i in range(4,8)], feature_names=['priors_count', 'sex', 'race', 'age_cat', 'c_charge_degree'],target_name='loan_status',feature_description=feature_description)

    pre_trained_model = keras.models.load_model('sample_trained_models' + os.sep + 'lending.h5')
    m = Model(pre_trained_model)

    return d, m
