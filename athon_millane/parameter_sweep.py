from stop_loss_policy import StopLossPolicy
import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from statsmodels.tsa.arima_process import ArmaProcess
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="white")


class ParameterSweep:
    """
    This class implements a number of parameter sweeps that may be desired
    to explore the results of applying a stop loss policy.

    1 - sweep_policy: Sweep across all parameters of interest for a policy
                      on a single (real) portfolio. Experiment visualised
                      as a grid of heatmaps covering all 4 parameters.

    2 - sweep_ar1:    Sweep across rho values of an AR(1) autoregressive
                      process.

    3 - sweep_paper:  Sweep across strategy application timeframe, stopping
                      window and stopping threshold as designed in Section
                      5.1 of the paper.
    """

    def __init__(self, returns_df: pd.DataFrame = None):
        """
        Constructor. Initalises the parameter sweep either with a returns df
        or as None if using a simulated returns process.

        :param returns_df (pd.DataFrame): Returns df to run experiments on.
        """

        if returns_df is not None:
            self.returns_df = returns_df

        else:
            if not simulate:
                raise Exception(
                    "Please provide a returns df or set simulate=True")

    def _heatmap_grid(self, experiment_df: pd.DataFrame, metric: str):
        """
        Helper method to plot a facet grid of heatmaps allowing for performance
        visualisation across all parameters of the stop loss policy.

        :param experiment_df: (pd.DataFrame): Experiment df.
        :param metric: (str): Performance measure to plot.
        """

        def draw_heatmap(*args, **kwargs):
            """
            Draw single heatmap.
            """
            data = kwargs.pop('data')
            d = data.pivot(columns=args[0], index=args[1],
                           values=args[2])
            ann = pd.DataFrame(d.values >= d.values.max()
                               ).replace({True: "X", False: ""})
            sns.heatmap(d, annot=ann, fmt="s", **kwargs)

        # Create a facetgrid and map the heatmap to that grid, with nice colour
        # mapping.
        grid = sns.FacetGrid(experiment_df, row='j', col='kappa')
        cbar_ax = grid.fig.add_axes([1.03, .15, .03, .7])
        cmap = sns.color_palette("Spectral", as_cmap=True)
        grid.map_dataframe(
            draw_heatmap,
            'gamma',
            'delta',
            metric,
            cmap=cmap,
            cbar_ax=cbar_ax,
            cbar_kws={
                'label': metric})

    def sweep_policy(
            self,
            sweep_params: dict,
            metric: str = "Stopping Premium",
            plot: bool = True) -> pd.DataFrame:
        """
        Sweep across all parameters of interest for a policy on a single
        (real) portfolio. Experiment results are visualised as a grid
        of heatmaps covering all parameters of the stop loss process.

        :param sweep_params: (dict): Dictionary of sweep parameters.
        :param metric: (str): Performance measure for sweep.
        :param plot: (bool): Flag for plotting results.
        :return (pd.DataFrame): Returns results table if plot=False.
        """

        # Construct parameter grid search
        c = [p for p in sweep_params.values()]
        experiment_df = pd.DataFrame(
            list(
                itertools.product(
                    *c)),
            columns=sweep_params.keys()).astype(float)

        # Iterate through parameters
        results = []
        for _, row in tqdm(experiment_df.iterrows(),
                           total=experiment_df.shape[0]):
            sl = StopLossPolicy(self.returns_df)

            params = {"gamma": row["gamma"],
                      "delta": row["delta"],
                      "j": row["j"],
                      "kappa": row["kappa"],
                      "rolling_sd": True}

            sl.calculate(**params)
            sl.apply(r_f=0)

            # Calculate performance measures and append to results dataframe.
            results.append(
                sl.calculate_performance_measures()
                  .set_index(0).loc[metric][1])

        experiment_df[metric] = results

        # Report best policy
        print(f"Best policy for metric {metric}:")
        best_policy = experiment_df.iloc[experiment_df[metric]
                                         .argmax()].rename("Value").to_frame()
        display(best_policy)

        if plot:
            self._heatmap_grid(experiment_df, metric)

        # Return optimal param dict
        best_params = best_policy["Value"][:-1].to_dict()
        best_params.update({"rolling_sd": True})
        return best_params

    def sweep_ar1(
            self,
            rhos: list,
            scale_factor: int = 100,
            n_samples: int = 100,
            policy_params: dict = None,
            plot: bool = True) -> pd.DataFrame:
        """
        This method sweeps across rho values of an AR(1)
        autoregressive process.

        :param rhos: (list): List of rho values to iterate over.
        :param scale_factor: (int): Scaling factor for process values.
        :param n_samples: (int): Number of samples from each process.
        :param policy_params: (dict): Parameters of the StopLossPolicy.
        :param plot: (bool): Flag for plotting results.
        :return (pd.DataFrame) : Returns results table if plot=False.
        """

        if policy_params is None:
            params = {"gamma": -0.5,
                      "delta": 0,
                      "j": 20,
                      "rolling_sd": True}
        else:
            params = policy_params

        results = []
        for rho in tqdm(rhos):
            # Initialise AR1 process.
            ar = AR1(scale_factor, rho)

            # Sample from the process.
            for s in range(n_samples):

                # Generate sample from AR1 process.
                sample = ar.sample()
                sample_df = pd.DataFrame(sample).rename({0: "r_t"}, axis=1)

                # Instantiate stop loss policy.
                sl = StopLossPolicy(sample_df)
                sl.calculate(**params)
                sl.apply(r_f=0)

                # Calculate performance measures and append to results
                # dataframe.
                res = sl.calculate_performance_measures().set_index(
                    [0]).T.reset_index(
                    drop=True) .drop(
                    "Sharpe Difference", axis=1)
                res["rho"] = np.around(rho, 3)
                res["sample"] = s
                res = res.set_index(["rho", "sample"])
                results.append(res)

        results = pd.concat(results)

        if plot:
            fig, ax = plt.subplots(figsize=[12, 8])
            outcomes = pd.melt(results.reset_index(), id_vars=[
                               "rho", "sample"]).rename({0: "metric"}, axis=1)

            # Plot the responses for different events and regions
            _ = sns.lineplot(
                data=outcomes,
                x="rho",
                y="value",
                hue="metric",
                style="metric",
                ci="sd")

        else:
            return results

    def sweep_paper(
            self,
            sweep_params: dict,
            plot: bool = True) -> pd.DataFrame:
        """
        Sweep across strategy application timeframe, stopping window
        and stopping threshold as designed in Section 5.1 of the paper.

        :param sweep_params (dict): Dictionary of sweep parameters.
        :param plot (bool): Flag for plotting results.
        :return (pd.DataFrame): Returns results table if plot=False.
        """

        # Calculate combinations from resample and J parameters.
        combinations = [(m, f) for f in sweep_params["resample"]
                        for m in sweep_params["J"]]

        # Iterate through parameter sweep.
        results = []
        for thresh in tqdm(sweep_params["gamma"]):

            experiments = []
            for window, freq in combinations:

                # Resample.
                resampled_df = self.returns_df.resample(f"{freq}D").sum()

                # Initialise policy.
                sl = StopLossPolicy(resampled_df)

                # Calculate policy.
                sl.calculate(
                    gamma=thresh,
                    delta=sweep_params["delta"][0],
                    j=window,
                    rolling_sd=True)

                # Apply policy.
                sl.apply()

                experiments.append(sl.calculate_performance_measures()
                                   .set_index(0)
                                   .rename({1: f"{freq},{window}"}, axis=1))

            # Process out results
            exp_df = pd.concat([e for e in experiments], axis=1)
            exp_df["gamma"] = thresh
            results.append(exp_df)

        results = pd.concat([r for r in results])\
                    .rename_axis("measure").reset_index()\
                    .melt(id_vars=["measure", "gamma"], var_name="Strategy")

        if plot:
            _ = sns.relplot(
                data=results, x="Strategy", y="value",
                row="measure", hue="gamma", style="gamma",
                kind="line", facet_kws=dict(sharey=False), height=2.5, aspect=3
            )
        else:
            return results.sort_values(["measure"])


class AR1:
    """
    Order 1 autoregressive process.
    """

    def __init__(self, s: int, rho: int):
        """
        Constructor for process.

        :param s: (int): Scale factor.
        :param rho: (int): Autoregressive parameter.
        """
        self.ar = np.array([s, -s * rho])
        self.ma = np.array([1])
        self.process = ArmaProcess(self.ar, self.ma)

    def sample(self, N: int = 1000):
        """
        Sample from the process.

        :param N: (int): Number of samples.
        :return (np.array): Samples from process.
        """
        sample = self.process.generate_sample(nsample=N)
        return sample
