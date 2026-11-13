"""Corner plots of posteriors and their TGMM fits (needs ``plot_helper``:
install ``astroinfer-tgmm[plot]``)."""
import numpy as np
import pandas as pd


def _ph():
    try:
        import plot_helper
    except ImportError as exc:
        raise ImportError("plotting needs plot_helper: install astroinfer-tgmm[plot]") from exc
    return plot_helper


def _labels(columns, labels):
    import matplotlib

    labels = dict(labels or {})
    escape = matplotlib.rcParams["text.usetex"]
    return [labels.get(c, c.replace("_", r"\_") if escape else c) for c in columns]


def _posterior_frame(posterior, columns, n, seed):
    df = pd.DataFrame({k: np.asarray(v) for k, v in posterior.variables(columns).items()})
    return df.sample(min(n, len(df)), random_state=seed)


def plot_posterior(catalog, eventname, columns=None, *, name="tgmm", n=8000, labels=None, **corner_kwargs):
    """Corner plot of an event's posterior; columns default to the TGMM config's."""
    if columns is None:
        columns = list(catalog.fitter(name).config_for(eventname).columns)
    corner_kwargs.setdefault("scatter_kwargs", dict(alpha=0.3, s=2))
    fig, _ = _ph().make_corner_plot(all_data=[_posterior_frame(catalog.posterior(eventname), columns, n, 0)],
                                    model_labels=["posterior"], variables=columns,
                                    variable_labels=_labels(columns, labels), **corner_kwargs)
    return fig


def plot_comparison(catalog, eventname, *, name="tgmm", n=8000, labels=None, **corner_kwargs):
    """Posterior draws against draws from the event's ``name`` fit."""
    fit = catalog.get_fit(eventname, name)
    if catalog.is_stale(eventname, name):
        import warnings

        warnings.warn(f"the {name!r} fit of {eventname} is stale", stacklevel=2)
    columns = list(fit.config.columns)
    df_post = _posterior_frame(catalog.posterior(eventname), columns, n, fit.config.seed)
    df_fit = fit.sample(len(df_post), seed=fit.config.seed)
    corner_kwargs.setdefault("scatter_kwargs", dict(alpha=0.3, s=2))
    fig, _ = _ph().make_corner_plot(all_data=[df_post[columns], df_fit[columns]],
                                    model_labels=["posterior", "TGMM fit"], variables=columns,
                                    variable_labels=_labels(columns, labels), **corner_kwargs)
    return fig
