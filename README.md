# Diverse Counterfactual Explanations (DiCE) for Machine Learning Models 
 
*How to explain a complex machine learning model such that the explanation is truthful to the model and yet interpretable to people?*

Arxiv paper | Documentation | Live Jupyter notebook 

Explanations are critical for machine learning, especially as machine learning-based systems are being used to inform decisions in societally critical domains such as finance, healthcare, education, and criminal justice. Since many machine learning algorithms are black boxes to end users and do not provide guarantees on input-output relationship, explanations serve a useful role to inspect these models.
Besides helping to debug ML models, explanations are hypothesized to improve the interpretability and trustworthiness of algorithmic decisions and enhance human decision making.

However, most explanation methods depend on an approximation of the ML model to
create an interpretable explanation. These are either global approximations
(e.g., ) or local approximations (e.g., LIME). For example,   
consider a person who applied for a loan and was rejected by the loan distribution algorithm of a financial company. Typically, the company may provide an explanation on why the loan was rejected, for example, due to ``poor credit history''. However, such an explanation does not help the person decide *what they do should next* to improve their chances of being approved in the future. Critically, the most important feature may not be enough to flip the decision of the algorithm, and in practice, may not even be changeable such as gender and race.


\emph{Counterfactual} explanations~\cite{wachter2017counterfactual} provide this information, by showing feature-perturbed versions of the same person who would have received the loan, e.g., ``you would have received the loan if your income was higher by $\$10,000$''. In other words, they provide ``what-if'' explanations for model output. Unlike explanation methods that depend on approximating the classifier's decision boundary~\cite{ribeiro2016should},
counterfactual (CF) explanations have the advantage that they are always truthful w.r.t. the underlying model by giving direct outputs of the algorithm.  Moreover, counterfactual examples may also be human-interpretable~\cite{wachter2017counterfactual} by allowing users to explore ``what-if'' scenarios,
similar to how children learn through counterfactual examples~\cite{weisberg2013pretense,beck2009relating,buchsbaum2012power}. 

# The promise of counterfactual explanations
Being truthful to the model, counterfactual explanations can be useful to all stakeholders for a decision made by a machine learning model that makes decisions.

* **Decision subjects**: Counterfactual explanations can be used to explore actionable recourse based on a decision received by a ML model. CF explanations can show decision outcomes from the algorithm 
with \emph{actionable} alternative profiles, to help people understand what they could have done to change their loan decision. 
Similar to the loan example above, such explanations are useful for a range of scenarios involving decision-making on an individual's outcome, such as deciding admission to a university~\cite{waters2014grade}, screening job applicants \cite{rockoff2011can}, disbursing government aid \cite{andini2017targeting,athey2017beyond}, and identifying people at high risk of a future disease \cite{dai2015prediction}. In all these cases, knowing reasons for a bad outcome is not enough; it is important to know what to do to obtain a better outcome in the future. 

* ML Model developers:  
* Decision makers:

* Decision evaluators: 


# Generating Counterfactual Explanations 
There is no free lunch, however. Barring simple linear models~\cite{russell2019efficient}, however, it is difficult to generate CF examples that work for any machine learning model. DiCE is based on recent research [link] that generates CF explanations for any ML model. The core idea to setup finding such explanations as an optimization problem, similar to finding adversarial examples. The critical difference is that for explanations, we need perturbations that change the output of a machine learning model, but are also diverse and feasible to change.

Therefore, DiCE supports generating a set of counterfactual explanations  and has tunable parameters for diversity and proximity of the explanations to the original input. It also supports simple constraints on features to ensure feasibility of the generated counterfactual examples. 

Here's the optimization problem that DiCE solves. 

Add equation.


# Getting started with DiCE


# Supported use-cases


# Roadmap
Ideally, counterfactual explanations should balance between a wide range of suggested changes (\emph{diversity}), and the relative ease of adopting those changes (\emph{proximity} to the original input), and also follow the causal laws of the world, e.g., one can hardly lower their educational degree or change their race. 

We are working on adding the following features to DiCE:
* Support for using DiCE for debugging machine learning models
* Support for other algorithms for generating counterfactual explanations 
* Incorporating causal constraints when generating counterfactual explanations


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
