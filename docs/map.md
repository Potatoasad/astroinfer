# A map of astroinfer

Most of astroinfer's classes are probability distributions, and the rest hold their inputs and
outputs or run inference with them. This page places every public class on one picture, a
hierarchical model, and gives the mathematical object each one stands for. The notation is that
of the rest of the documentation:

| symbol | meaning |
|---|---|
| $\Lambda$ | hyperparameters of a population |
| $\theta$ | parameters of one object, or event |
| $d$ | data of one event |
| $\pi(\theta\mid\Lambda)$ | population model |
| $\mathcal L(d\mid\theta)$ | likelihood, defined by a measurement process |
| $\pi(\theta)$ | prior used for parameter estimation (the PE prior) |
| $p(\theta\mid d)$ | posterior of one event |
| $q_\Lambda(\theta\mid d)$ | proposal for importance sampling |
| $P(\mathrm{det}\mid d)$ | probability that data are detected |
| $\alpha(\Lambda)$ | fraction of the population that is detected |

## The hierarchical model

A population draws the parameters of each event, a measurement process draws data from the
parameters, and a detection trigger decides which data are kept. Each arrow is one class:

```{mermaid}
flowchart LR
    L["Λ<br/>hyperparameters"]
    T["θ<br/>parameters"]
    D["d<br/>data"]
    K["detected"]
    L -->|"π(θ ∣ Λ)<br/>PopulationModel, StructuredDistribution,<br/>GenerativeModel"| T
    T -->|"ℒ(d ∣ θ)<br/>MeasurementProcess"| D
    D -->|"P(det ∣ d)<br/>DetectionTrigger"| K
    D -.->|"q_Λ(θ ∣ d)<br/>Proposal"| T
```

The dashed arrow runs backwards: a proposal draws parameters from data, to estimate the integral
over $\theta$ by importance sampling. $\theta$, $d$ and, optionally, $\Lambda$ are objects of
*parameter types*, classes made with `@parameter_dataclass`.

## Distributions

Every distribution in astroinfer is a subclass of `ConditionalDistribution`, a distribution
$p(x\mid c)$ of an output $x$ given conditions $c$. A subclass writes `sample_single` and
`logpdf_single` for one $x$ and one $c$; the base class batches them, binds conditions and draws
the display. The subclasses differ in what $x$ and $c$ are:

```{mermaid}
flowchart TD
    CD["ConditionalDistribution<br/>p(x | c)"]
    MP["MeasurementProcess<br/>ℒ(d | θ)"]
    PR["Proposal<br/>q_Λ(θ | d)"]
    PM["PopulationModel<br/>π(θ | Λ)"]
    SD["StructuredDistribution<br/>∏_f p(x_f | x_earlier, c)"]
    PD["ProductDistribution<br/>p_a(x_a | c) p_b(x_b | c)"]
    MD["MixtureDistribution<br/>Σ_k w_k(c) p_k(x | c)"]
    BD["BoundConditionalDistribution<br/>p(x | c fixed)"]
    CD --> MP
    CD --> PR
    CD --> PM
    CD --> SD
    CD --> PD
    CD --> MD
    CD --> BD
    GM["GenerativeModel<br/>π(θ | Λ) by inverse transform"]
    GM -.->|"PopulationModel.from_generative"| PM
    SD -.->|"as_generative"| GM
```

| class | distribution | output $x$ | conditions $c$ | made by |
|---|---|---|---|---|
| `ConditionalDistribution` | $p(x\mid c)$ | any parameter type | any, or none | subclassing |
| `MeasurementProcess` | $\mathcal L(d\mid\theta)$ | data $d$ | parameters $\theta$ | subclassing |
| `Proposal` | $q_\Lambda(\theta\mid d)$ | parameters $\theta$ | data $d$, with $\Lambda$ passed whole | subclassing |
| `PopulationModel` | $\pi(\theta\mid\Lambda)$ | parameters $\theta$ | hyperparameters $\Lambda$ | subclassing, or `PopulationModel.from_generative` |
| `StructuredDistribution` | $\prod_f p(x_f\mid x_{\text{earlier}}, c)$, one factor per field | a parameter type | none, or set with `.given(type)` | `Type.distribution(field=...)` |
| `ProductDistribution` | $p_a(x_a\mid c)\,p_b(x_b\mid c)$ | a tuple or dict of outputs | the union of the components' | `ai.product(...)` |
| `MixtureDistribution` | $\sum_k w_k(c)\,p_k(x\mid c)$ | one output type | the components', and the weights' | `ai.mixture(...)` |
| `BoundConditionalDistribution` | $p(x\mid c)$ with some of $c$ fixed | the parent's | the parent's, minus the bound ones | `.condition(...)` |

Two kinds of distribution sit outside this tree. A `GenerativeModel` (in `astroinfer.generative`)
is a population written as leaves (`Uniform`, `Beta`, `TruncatedNormal`, `NumPyroLeaf`) combined
with `product`, `mixture` and `conditional`; it draws by inverse transform from uniform *seeds*,
so its draws are smooth functions of $\Lambda$. `PopulationModel.from_generative` wraps it as a
conditional distribution, and `StructuredDistribution.as_generative` converts the other way. A
NumPyro distribution, such as `dist.Normal`, describes one field: it is a field of a structured
distribution or a plain site in a NumPyro model.

A `DetectionTrigger` (in `astroinfer.detection`) is not a distribution of data. It gives
$P(\mathrm{det}\mid d)$ with `detection_probability(d)`, draws detections with `detect`, and
makes injection tables with `injections(measurement, draws, rng)`. `ThresholdTrigger` detects data
whose selection statistic exceeds a threshold.

## Objects that hold values

| class | what it holds |
|---|---|
| `Parameters` | one or a batch of $\theta$, $d$ or $\Lambda$: the base class of every parameter type, a JAX pytree with batch axes and named core axes |
| `ParameterSpec` | the declaration of one field: name, LaTeX, core axes, support |
| `LatentSpace` | a bijection, field by field, between a parameter type and unconstrained coordinates $z\in\mathbb R^n$ |
| `PosteriorData` | draws $\theta_j \sim p(\theta\mid d)$ of one event, with $\log\mathcal L(d\mid\theta_j)$ and $\pi(\theta_j)$ in `sample_stats`, derived quantities in `extras` and $d$ in `observed_data` |
| `Catalog` | one `PosteriorData` per event, and the representations fitted to them |
| `Fit`, `Fitter` | a *representation* of one event's posterior, such as a Gaussian mixture fitted to it, and the object that makes one fit per event; `RawFit` and `RawSamples` are the built-in representation, the draws themselves |
| `Graph` | a directed acyclic graph of a distribution (`.graph()`) or of a model function (`ai.model_graph`) |

## Inference

Three routes lead from data to posteriors, and each ends in a `PosteriorData`:

```{mermaid}
flowchart TD
    subgraph one ["One event at a time"]
        D1["data d of one event"]
        PE["MeasurementProcess.PE<br/>p(θ | d) ∝ ℒ(d | θ) π(θ)"]
        PD["PosteriorData<br/>θ_j ~ p(θ | d)"]
        CAT["Catalog<br/>one posterior per event"]
        DP["data product<br/>draws or fits, events along the first axis"]
        D1 --> PE --> PD --> CAT -->|"Fitter"| DP
    end
    subgraph pop ["A population from the data of all events"]
        DE["data d_1 … d_E"]
        PL["PopulationLikelihood<br/>ℒ(Λ) = ∏_i Z_i(Λ)"]
        POP["PosteriorData over Λ"]
        DE --> PL -->|"population_inference"| POP
        PRO["Proposal q_Λ(θ | d)"] -.-> PL
        INJ["injections, α(Λ)"] -.-> PL
    end
    subgraph np ["Any NumPyro model"]
        M["model function<br/>ai.sample, ai.factor"]
        INF["ai.infer"]
        POST["PosteriorData<br/>one field per root"]
        MG["marginalize=True<br/>∫ p(a | ·) ℒ(d | a, ·) da"]
        M --> INF --> POST
        MG -.-> M
    end
```

The first route samples each event's posterior with its PE prior and collects the posteriors in a
catalog, whose data product is the input of population analyses that work from posterior draws,
such as gravpop's. The second integrates each event's parameters against the population directly,
through the measurement process, by Monte Carlo. Both use gravpop's sampler. The third is any
model you write in NumPyro, with typed objects at its sites, sampled by NUTS or nested sampling.

| object | quantity | role |
|---|---|---|
| `FixedDataLikelihood` | $\theta \mapsto \mathcal L(d\mid\theta)$ for fixed $d$ | parameter estimation of one event (`parameter_estimation`, `MeasurementProcess.PE`), giving $p(\theta\mid d) \propto \mathcal L(d\mid\theta)\,\pi(\theta)$ |
| `PopulationLikelihood` | $\mathcal L(\Lambda) = \prod_i Z_i(\Lambda)$, with $Z_i(\Lambda) = \mathbb E_{\theta\sim\pi(\cdot\mid\Lambda)}\,\mathcal L(d_i\mid\theta)$ | population inference (`population_inference`); with a proposal, $Z_i \approx \frac1N\sum_j \mathcal L(d_i\mid\theta_{ij})\,\pi(\theta_{ij}\mid\Lambda)/q_\Lambda(\theta_{ij}\mid d_i)$; with selection effects, each factor is divided by $\alpha(\Lambda)$, estimated from injections |
| `JointPopulationLikelihood` | $\prod_i \mathcal L(d_i\mid\theta_i)\,\pi(\theta_i\mid\Lambda)$ | inference of $\Lambda$ and every $\theta_i$ together (`joint_population_inference`) |
| `run_sampler` | | runs gravpop's sampler on any of the three |
| `ai.sample(name, distribution, ...)` | one site per random field, or one site per object | puts a distribution in a NumPyro model; with `obs=`, adds $\log\mathcal L$ of the data |
| `ai.factor(name, value)` | adds `value` to $\log p$ | extra terms, such as $-E\log\alpha(\Lambda)$ |
| `ai.infer(model, ...)` | samples $p(\text{latent sites}\mid\text{observed sites})$ | NUTS or nested sampling on the model |
| `ai.marginal` families and pair rules | $\log \int p(a\mid\cdot)\,\mathcal L(d\mid a, \cdot)\,da$, and draws of $a$ given $d$ | integrating a conjugate value out of a model (`marginalize=True`); see [Integrating values out of a model](user-guide/marginalization.md) |

A measurement process also has convenience methods for the first two routes: `PE(d, priors)` runs
parameter estimation for one event and returns a `PosteriorData`, and
`population_inference(data, population, priors)` and `joint_population_inference(...)` take the
data of all events as one batched object.

## Where each class is explained

| class | page |
|---|---|
| parameter types, `Parameters`, `LatentSpace` | [Parameter types](user-guide/parameters.md) |
| `MeasurementProcess`, `PosteriorData`, PE | [Measurement processes](user-guide/measurement-processes.md) |
| `ConditionalDistribution`, `StructuredDistribution`, products, mixtures, `Graph` | [Distributions over parameter types](user-guide/distributions.md) |
| `ai.sample`, `ai.factor`, `ai.infer`, `ai.model_graph` | [NumPyro models](user-guide/numpyro-models.md) |
| `ai.marginal` | [Integrating values out of a model](user-guide/marginalization.md) |
| `GenerativeModel`, `PopulationModel` | [Population models](user-guide/population-models.md) |
| `PopulationLikelihood`, `Proposal`, `DetectionTrigger`, injections | [Population inference](user-guide/population-inference.md) |
| `Catalog`, `Fit`, `Fitter`, representations | [Catalogs](user-guide/catalogs.md) |
