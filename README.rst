|BuildStatus|_ |PyPiVersion|_ |PythonSupport|_ |Downloads|_ |CondaVersion|_

.. |BuildStatus| image:: https://github.com/interpretml/DiCE/actions/workflows/python-package.yml/badge.svg?branch=main
.. _BuildStatus: https://github.com/interpretml/DiCE/actions/workflows/python-package.yml?query=workflow%3A%22Python+package%22

.. |PyPiVersion| image:: https://img.shields.io/pypi/v/dice-ml
.. _PyPiVersion: https://pypi.org/project/dice-ml/

.. |Downloads| image:: https://static.pepy.tech/personalized-badge/dice-ml?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
.. _Downloads: https://pepy.tech/project/dice-ml

.. |PythonSupport| image:: https://img.shields.io/pypi/pyversions/dice-ml
.. _PythonSupport: https://pypi.org/project/dice-ml/

.. |CondaVersion| image:: https://anaconda.org/conda-forge/dice-ml/badges/version.svg
.. _CondaVersion: https://anaconda.org/conda-forge/dice-ml

Diverse Counterfactual Explanations (DiCE) for ML
======================================================================

*How to explain a machine learning model such that the explanation is truthful to the model and yet interpretable to people?*

`Ramaravind K. Mothilal <https://raam93.github.io/>`_, `Amit Sharma <http://www.amitsharma.in/>`_, `Chenhao Tan <https://chenhaot.com/>`_
  
`FAT* '20 paper <https://arxiv.org/abs/1905.07697>`_ | `Docs <https://interpretml.github.io/DiCE/>`_ | `Example Notebooks <https://github.com/interpretml/DiCE/tree/master/docs/source/notebooks>`_ | Live Jupyter notebook |Binder|_

.. |Binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder:  https://mybinder.org/v2/gh/interpretML/DiCE/master?filepath=docs/source/notebooks

 **Blog Post**: `Explanation for ML using diverse counterfactuals <https://www.microsoft.com/en-us/research/blog/open-source-library-provides-explanation-for-machine-learning-through-diverse-counterfactuals/>`_
 
 **Case Studies**: `Towards Data Science <https://towardsdatascience.com/dice-diverse-counterfactual-explanations-for-hotel-cancellations-762c311b2c64>`_ (Hotel Bookings) | `Analytics Vidhya <https://medium.com/analytics-vidhya/dice-ml-models-with-counterfactual-explanations-for-the-sunk-titanic-30aa035056e0>`_ (Titanic Dataset)
 
.. image:: https://www.microsoft.com/en-us/research/uploads/prod/2020/01/MSR-Amit_1400x788-v3-1blog.gif
  :align: center
  :alt: Visualizing a counterfactual explanation
  
Explanations are critical for machine learning, especially as machine learning-based systems are being used to inform decisions in societally critical domains such as finance, healthcare, education, and criminal justice.
However, most explanation methods depend on an approximation of the ML model to
create an interpretable explanation. For example,
consider a person who applied for a loan and was rejected by the loan distribution algorithm of a financial company. Typically, the company may provide an explanation on why the loan was rejected, for example, due to "poor credit history". However, such an explanation does not help the person decide *what they do should next* to improve their chances of being approved in the future. Critically, the most important feature may not be enough to flip the decision of the algorithm, and in practice, may not even be changeable such as gender and race.


DiCE implements `counterfactual (CF) explanations <https://arxiv.org/abs/1711.00399>`_  that provide this information by showing feature-perturbed versions of the same person who would have received the loan, e.g., ``you would have received the loan if your income was higher by $10,000``. In other words, it provides "what-if" explanations for model output and can be a useful complement to other explanation methods, both for end-users and model developers.

Barring simple linear models, however, it is difficult to generate CF examples that work for any machine learning model. DiCE is based on `recent research <https://arxiv.org/abs/1905.07697>`_ that generates CF explanations for any ML model. The core idea is to setup finding such explanations as an optimization problem, similar to finding adversarial examples. The critical difference is that for explanations, we need perturbations that change the output of a machine learning model, but are also diverse and feasible to change. Therefore, DiCE supports generating a set of counterfactual explanations  and has tunable parameters for diversity and proximity of the explanations to the original input. It also supports simple constraints on features to ensure feasibility of the generated counterfactual examples.


Installing DICE
-----------------
DiCE supports Python 3+. The stable version of DiCE is available on `PyPI <https://pypi.org/project/dice-ml/>`_.

.. code:: bash

    pip install dice-ml

DiCE is also available on `conda-forge <https://anaconda.org/conda-forge/dice-ml>`_. 

.. code:: bash

    conda install -c conda-forge dice-ml

To install the latest (dev) version of DiCE and its dependencies, clone this repo and run `pip install` from the top-most folder of the repo:

.. code:: bash

    pip install -e .

If you face any problems, try installing dependencies manually.

.. code:: bash

    pip install -r requirements.txt
    # Additional dependendies for deep learning models
    pip install -r requirements-deeplearning.txt
    # For running unit tests
    pip install -r requirements-test.txt


Getting started with DiCE
-------------------------
With DiCE, generating explanations is a simple three-step  process: set up a dataset, train a model, and then invoke DiCE to generate counterfactual examples for any input. DiCE can also work with pre-trained models, with or without their original training data. 


.. code:: python

    import dice_ml
    from dice_ml.utils import helpers # helper functions
    from sklearn.model_selection import train_test_split

    dataset = helpers.load_adult_income_dataset()
    target = dataset["income"] # outcome variable 
    train_dataset, test_dataset, _, _ = train_test_split(dataset,
                                                         target,
                                                         test_size=0.2,
                                                         random_state=0,
                                                         stratify=target)
    # Dataset for training an ML model
    d = dice_ml.Data(dataframe=train_dataset,
                     continuous_features=['age', 'hours_per_week'],
                     outcome_name='income')
    
    # Pre-trained ML model
    m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
                      backend='TF2', func="ohe-min-max")
    # DiCE explanation instance
    exp = dice_ml.Dice(d,m)

For any given input, we can now generate counterfactual explanations. For
example, the following input leads to class 0 (low income) and we would like to know what minimal changes would lead to a prediction of 1 (high income).

.. code:: python
    
    # Generate counterfactual examples
    query_instance = test_dataset.drop(columns="income")[0:1]
    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
    # Visualize counterfactual explanation
    dice_exp.visualize_as_dataframe()

.. image:: https://raw.githubusercontent.com/interpretml/DiCE/master/docs/_static/getting_started_updated.png 
  :width: 400
  :alt: List of counterfactual examples

You can save the generated counterfactual examples in the following way.

.. code:: python

    # Save generated counterfactual examples to disk
    dice_exp.cf_examples_list[0].final_cfs_df.to_csv(path_or_buf='counterfactuals.csv', index=False)


For more details, check out the `docs/source/notebooks <https://github.com/interpretml/DiCE/tree/master/docs/source/notebooks>`_ folder. Here are some example notebooks:

* `Getting Started <https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_getting_started.ipynb>`_: Generate CF examples for a `sklearn`, `tensorflow` or `pytorch` binary classifier and compute feature importance scores.
* `Explaining Multi-class Classifiers and Regressors
  <https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_multiclass_classification_and_regression.ipynb>`_: Generate CF explanations for a multi-class classifier or regressor.
* `Local and Global Feature Importance <https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_feature_importances.ipynb>`_: Estimate local and global feature importance scores using generated counterfactuals.
* `Providing Constraints on Counterfactual Generation
  <https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_model_agnostic_CFs.ipynb>`_: Specifying which features to vary and their permissible ranges for valid counterfactual examples.

Supported methods for generating counterfactuals
------------------------------------------------
DiCE can generate counterfactual examples using the following methods.

**Model-agnostic methods**

* Randomized sampling 
* KD-Tree (for counterfactuals within the training data)
* Genetic algorithm 

See `model-agnostic notebook
<https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_model_agnostic_CFs.ipynb>`_ for code examples on using these methods.

**Gradient-based methods**

* An explicit loss-based method described in `Mothilal et al. (2020) <https://arxiv.org/abs/1905.07697>`_ (Default for deep learning models).
* A Variational AutoEncoder (VAE)-based method described in `Mahajan et al. (2019) <https://arxiv.org/abs/1912.03277>`_ (see the BaseVAE `notebook <https://github.com/interpretml/DiCE/blob/master/docs/notebooks/DiCE_getting_started_feasible.ipynb>`_).

The last two methods require a differentiable model, such as a neural network. If you are interested in a specific method, do raise an issue `here <https://github.com/interpretml/DiCE/issues>`_.

Supported use-cases
-------------------
**Data**

DiCE does not need access to the full dataset. It only requires metadata properties for each feature (min, max for continuous features and levels for categorical features). Thus, for sensitive data, the dataset can be provided as:

.. code:: python

    d = data.Data(features={
                       'age':[17, 90],
                       'workclass': ['Government', 'Other/Unknown', 'Private', 'Self-Employed'],
                       'education': ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college'],
                       'marital_status': ['Divorced', 'Married', 'Separated', 'Single', 'Widowed'],
                       'occupation':['Blue-Collar', 'Other/Unknown', 'Professional', 'Sales', 'Service', 'White-Collar'],
                       'race': ['Other', 'White'],
                       'gender':['Female', 'Male'],
                       'hours_per_week': [1, 99]},
             outcome_name='income')

**Model**

We support pre-trained models as well as training a model. Here's a simple example using Tensorflow. 

.. code:: python

    sess = tf.InteractiveSession()
    # Generating train and test data
    train, _ = d.split_data(d.normalize_data(d.one_hot_encoded_data))
    X_train = train.loc[:, train.columns != 'income']
    y_train = train.loc[:, train.columns == 'income']
    # Fitting a dense neural network model
    ann_model = keras.Sequential()
    ann_model.add(keras.layers.Dense(20, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l1(0.001), activation=tf.nn.relu))
    ann_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    ann_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])
    ann_model.fit(X_train, y_train, validation_split=0.20, epochs=100, verbose=0, class_weight={0:1,1:2})

    # Generate the DiCE model for explanation
    m = model.Model(model=ann_model)

Check out the `Getting Started <https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_getting_started.ipynb>`_ notebook to see code examples on using DiCE with sklearn and PyTorch models.

**Explanations**

We visualize explanations through a table highlighting the change in features. We plan to support an English language explanation too!

Feasibility of counterfactual explanations
-------------------------------------------
We acknowledge that not all counterfactual explanations may be feasible for a
user. In general, counterfactuals closer to an individual's profile will be
more feasible. Diversity is also important to help an individual choose between
multiple possible options.

DiCE provides tunable parameters for diversity and proximity to generate
different kinds of explanations.

.. code:: python

    dice_exp = exp.generate_counterfactuals(query_instance,
                    total_CFs=4, desired_class="opposite",
                    proximity_weight=1.5, diversity_weight=1.0)

Additionally, it may be the case that some features are harder to change than
others (e.g., education level is harder to change than working hours per week). DiCE allows input of relative difficulty in changing a feature through specifying *feature weights*. A higher feature weight means that the feature is harder to change than others. For instance, one way is to use the mean absolute deviation from the median as a measure of relative difficulty of changing a continuous feature. By default, DiCE computes this internally and divides the distance between continuous features by the MAD of the feature's values in the training set. We can also assign different values through the *feature_weights* parameter. 

.. code:: python

    # assigning new weights
    feature_weights = {'age': 10, 'hours_per_week': 5}
    # Now generating explanations using the new feature weights
    dice_exp = exp.generate_counterfactuals(query_instance,
                    total_CFs=4, desired_class="opposite",
                    feature_weights=feature_weights)

Finally, some features are impossible to change such as one's age or race. Therefore, DiCE also allows inputting a
list of features to vary.

.. code:: python

    dice_exp = exp.generate_counterfactuals(query_instance,
                    total_CFs=4, desired_class="opposite",
                    features_to_vary=['age','workclass','education','occupation','hours_per_week'])

It also supports simple constraints on
features that reflect practical constraints (e.g., working hours per week
should be between 10 and 50 using the ``permitted_range`` parameter).

For more details, check out `this <https://github.com/interpretml/DiCE/blob/master/docs/source/notebooks/DiCE_model_agnostic_CFs.ipynb>`_ notebook.

The promise of counterfactual explanations
-------------------------------------------
Being truthful to the model, counterfactual explanations can be useful to all stakeholders for a decision made by a machine learning model that makes decisions.

* **Decision subjects**: Counterfactual explanations can be used to explore actionable recourse for a person based on a decision received by a ML model. DiCE shows decision outcomes with *actionable* alternative profiles, to help people understand what they could have done to change their model outcome.

* **ML model developers**: Counterfactual explanations are also useful for model developers to debug their model for potential problems. DiCE can be used to show CF explanations for a selection of inputs that can uncover if there are any problematic (in)dependences on some features (e.g., for 95% of inputs, changing features X and Y change the outcome, but not for the other 5%). We aim to support aggregate metrics to help developers debug ML models.

* **Decision makers**: Counterfactual explanations may be useful to
  decision-makers such as doctors or judges who may use ML models to make decisions. For a particular individual, DiCE allows probing the ML model to see the possible changes that lead to a different ML outcome, thus enabling decision-makers to assess their trust in the prediction.

* **Decision evaluators**: Finally, counterfactual explanations can be useful
  to decision evaluators who may be interested in fairness or other desirable
  properties of an ML model. We plan to add support for this in the future.


Roadmap
-------
Ideally, counterfactual explanations should balance between a wide range of suggested changes (*diversity*), and the relative ease of adopting those changes (*proximity* to the original input), and also follow the causal laws of the world, e.g., one can hardly lower their educational degree or change their race.

We are working on adding the following features to DiCE:

* Support for using DiCE for debugging machine learning models
* Constructed English phrases (e.g., ``desired outcome if feature was changed``) and other ways to output the counterfactual examples
* Evaluating feature attribution methods like LIME and SHAP on necessity and sufficiency metrics using counterfactuals (see `this paper <https://arxiv.org/abs/2011.04917>`_)
* Support for Bayesian optimization and other algorithms for generating counterfactual explanations
* Better feasibility constraints for counterfactual generation 

Citing
-------
If you find DiCE useful for your research work, please cite it as follows.

Ramaravind K. Mothilal, Amit Sharma, and Chenhao Tan (2020). **Explaining machine learning classifiers through diverse counterfactual explanations**. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*. 

Bibtex::

	@inproceedings{mothilal2020dice,
  		title={Explaining machine learning classifiers through diverse counterfactual explanations},
  		author={Mothilal, Ramaravind K and Sharma, Amit and Tan, Chenhao},
  		booktitle={Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency},
  		pages={607--617},
  		year={2020}
	}


Contributing
------------

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the `Microsoft Open Source Code of Conduct <https://opensource.microsoft.com/codeofconduct/>`_.
For more information see the `Code of Conduct FAQ <https://opensource.microsoft.com/codeofconduct/faq/>`_ or
contact `opencode@microsoft.com <mailto:opencode@microsoft.com>`_ with any additional questions or comments.
