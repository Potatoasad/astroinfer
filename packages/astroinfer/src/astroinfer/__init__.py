"""astroinfer: typed forward models and hierarchical inference in JAX.

Layout (each layer only imports the ones above it):

    display       HTML / LaTeX rendering for notebooks (RichDisplay)
    parameters    @parameter_dataclass: batched JAX pytrees for theta and d
    latent        LatentSpace: per-field bijections to the real line
    distributions ConditionalDistribution: p(x | c) from unbatched sample/logpdf
    posterior     PosteriorData: posterior draws + sample stats + metadata
    generative    differentiable population models p(theta | Lambda)
    population    PopulationModel: the ConditionalDistribution p(theta | Lambda)
    inference     likelihoods and gravpop-backed PE / population sampling
    modeling      sample / factor / infer: typed distributions inside NumPyro models
    measurement   MeasurementProcess: the ConditionalDistribution L(d | theta)
    representations  Fit/Fitter protocol, registry, the built-in raw representation
    catalog       Catalog of per-event posteriors and their representations
    detection     detection triggers P(detected | d) and injection tables
"""
from . import detection, generative, representations
from .catalog import Catalog
from .display import RichDisplay
from .distributions import (
    BoundConditionalDistribution,
    ConditionalDistribution,
    MixtureDistribution,
    ProductDistribution,
    StructuredDistribution,
    conditional,
    mixture,
    product,
)
from .latent import LatentSpace
from .inference import (
    FixedDataLikelihood,
    JointPopulationLikelihood,
    PopulationLikelihood,
    joint_population_inference,
    parameter_estimation,
    population_inference,
    run_sampler,
)
from .measurement import MeasurementProcess, Proposal
from .modeling import factor, infer, sample
from .parameters import (
    Parameters,
    ParameterSpec,
    batched_method,
    coerce,
    parameter,
    parameter_dataclass,
)
from .population import PopulationModel
from .posterior import PosteriorData
from .representations import RawFit, RawSamples, register

__all__ = [
    "factor",
    "infer",
    "sample",
    "MixtureDistribution",
    "ProductDistribution",
    "StructuredDistribution",
    "conditional",
    "mixture",
    "product",
    "BoundConditionalDistribution",
    "Catalog",
    "ConditionalDistribution",
    "FixedDataLikelihood",
    "JointPopulationLikelihood",
    "LatentSpace",
    "MeasurementProcess",
    "ParameterSpec",
    "Parameters",
    "PopulationLikelihood",
    "PopulationModel",
    "PosteriorData",
    "Proposal",
    "RawFit",
    "RawSamples",
    "RichDisplay",
    "batched_method",
    "coerce",
    "detection",
    "generative",
    "joint_population_inference",
    "parameter",
    "parameter_dataclass",
    "parameter_estimation",
    "register",
    "representations",
    "population_inference",
    "run_sampler",
]
