"""Constants for dice-ml package."""


class BackEndTypes:
    Sklearn = 'sklearn'
    Tensorflow1 = 'TF1'
    Tensorflow2 = 'TF2'
    Pytorch = 'PYT'

    ALL = [Sklearn, Tensorflow1, Tensorflow2, Pytorch]


class SamplingStrategy:
    Random = 'random'
    Genetic = 'genetic'
    KdTree = 'kdtree'


class ModelTypes:
    Classifier = 'classifier'
    Regressor = 'regressor'

    ALL = [Classifier, Regressor]


class _SchemaVersions:
    V1 = '1.0'
    V2 = '2.0'
    CURRENT_VERSION = V2

    ALL_VERSIONS = [V1, V2]
