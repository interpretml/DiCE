from dice_ml.counterfactual_explanations import CounterfactualExplanations


def json_converter(obj):
    """ Helper function to convert CounterfactualExplanations object to json.
    """
    if isinstance(obj, CounterfactualExplanations):
        return obj.__dict__
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__

