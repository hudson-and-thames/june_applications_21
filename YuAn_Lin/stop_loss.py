import pandas as pd
import numpy as np
from typing import Union, Callable


import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource, Panel, Tabs, ColorBar, LinearColorMapper
from bokeh.transform import transform
from bokeh.layouts import row, column, gridplot
from bokeh.palettes import Spectral10, Reds
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn


class StopLoss:
    """Implementation of Kaminski, Kathryn M., and Andrew W. Lo. (2013).
    
    Paper Link
    ----------
    https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf

    Parameters
    ----------
    strategy_return : Pandas Series
        Series contains returns of strategy.

    rf_return : Pandas Series or float
        It can be a series contains returns of risk-free asset or a constant risk free rate.
        
    Returns
    -------
    StopLoss Object
    """

    
    def __init__(self, strategy_return: pd.Series, rf_return: Union[pd.Series, float]):
        self.__check_index_type(strategy_return)
        self.strategy_return = strategy_return        
        
        if isinstance(rf_return, float):
            self.rf_return = self.strategy_return.copy()
            self.rf_return.iloc[:] = rf_return
            
        elif isinstance(rf_return, pd.core.series.Series):
            self.rf_return = rf_return
            
            #Check whether two series have same lengths.
            if len(self.rf_return) != len(self.strategy_return):
                raise ValueError("strategy_return and rf_return need to have same lengths.")
                
            self.__check_index_type(rf_return)
            self.rf_return = rf_return
            
        else:
            raise TypeError("rf_return has to be Pandas Series or float.")
            
        # Initialize dictionaries for calculated results
        self.custom_policy_dict = {}
        self.st_dict = {}
        self.performance_dict = {}
        self.rst_dict = {}
        
        # Initialize strategy performance without stopping policy
        self.__strategy_performance()
        
        
    # Private Methods
    def __check_index_type(self, series: pd.Series):
        """Check whether the index has a correct type."""
        
        if not isinstance(series.index, pd.core.indexes.datetimes.DatetimeIndex): 
            raise TypeError("Index's type must be DatetimeIndex.")

    
    def __strategy_performance(self):
        # Calculate metrics
        # Expected Return
        e_rt = self.strategy_return.mean()
        
        # Standard Deviation
        std_rt = self.strategy_return.std()
        
        # Sharpe Ratio
        sr_rt = (self.strategy_return - self.rf_return).mean() / std_rt
        
        self.origin_strategy_performance = {
            "Expected Return without Stopping Policy" : e_rt,
            "Standard Deviation without Stopping Policy" : std_rt,
            "Sharpe Ratio without Stopping Policy" : sr_rt,
        }
        
    
    # Public Methods
    def simple_stop_loss_policy(self, gamma: float, delta: float, J: int, compounding: bool = False):
        """Implementation of Definition 1.
        
        Method:
        st ≡ 0 if Rt−1(J) < −gamma and st−1 = 1 (exit)
             1 if rt−1 ≥ delta and st−1 = 0 (re-enter)
             1 if Rt−1(J) ≥ −gamma and st−1 = 1 (stay in)
             0 if rt−1 < delta and st−1 = 0 (stay out)
             
        rst ≡ st*rt + (1 − st)*rf
        
        where st denotes proportion of assets allocated in time t.
              rst denotes the return of combinaton of portfolio strategy P and the stop-loss policy S.

        Parameters
        ----------
        gamma : float
            Loss threshold

        delta : float
            Re-entry threshold
            
        J : int
            Cumulative-return window
            
        compounding : bool, default False
            Whether returns are reinvested back into the account.
            
        Returns
        -------
        Series contains proportions of assets allocated.
        """
        
        # Calculate cumulative returns.
        if compounding:
            cum_return = (self.strategy_return + 1).rolling(J).apply(np.prod, raw=True) - 1
        else:
            cum_return = self.strategy_return.rolling(J).sum()
        
        # Create Series contains proportions of assets allocated.
        st = self.strategy_return.copy()
        st[:] = 1
        
        cr_condition = (cum_return >= -gamma)
        r_condition = (self.strategy_return >= delta)
        
        """
        Change the method into Boolean algebra format:
        F = 0 if A = 0 and C = 1
            1 if B = 1 and C = 0 
            1 if A = 1 and C = 1
            0 if B = 0 and C = 0
        We can get F = AC + BC', where A = Rt−1(J) ≥ −gamma, B = rt−1 ≥ delta, C = st−1 
        """
        for i in range(J, len(st)):
            st[i] = (cr_condition[i - 1] and st[i - 1]) or (r_condition[i - 1] and (not st[i - 1]))
        
        return st
    
    
    def stopping_policy_performance(self, st: pd.Series, kappa: float = 0):
        """Implementation of Definition 2, 3.
        
        Method:
        delta_mean = E[rst] - E[rt]
        delta_sigma = Var[rst]^(1/2) - Var[rt]^(1/2)
        delta_sr = (E[rst] - rf)/[Var[rst]^(1/2)] - (E[rt] - rf)/[Var[rt]^(1/2)]
        stopping_ratio = delta_mean / p0
        
        where p0 denotes Prob(st = 0)

        Parameters
        ----------
        st : Series
            Series contains proportions of assets allocated.
        
        kappa : float
            One-way transactions cost of a stop-loss event
            
        Returns
        -------
        1. Dictionary contains performance metrics
        2. rst
        """
        
        # Get origin strategy performance
        e_rt = self.origin_strategy_performance["Expected Return without Stopping Policy"]
        std_rt = self.origin_strategy_performance["Standard Deviation without Stopping Policy"]
        sr_rt = self.origin_strategy_performance["Sharpe Ratio without Stopping Policy"]
        
        # Calculate returns with the stop-loss policy.
        rst = st*self.strategy_return + (1 - st)*self.rf_return - kappa*(st - st.shift(1).fillna(0))
        
        # Calculate metrics
        # Expected Return
        e_rst = rst.mean()
        
        # Standard Deviation
        std_rst = rst.std()
        
        # Sharpe Ratio
        sr_rst = (rst - self.rf_return).mean() / std_rst
        
        # Stopping Ratio
        p0 = (1 - st).sum() / len(st)
        stopping_ratio = (e_rst - e_rt) / p0
        
        performance = {"Expected Return with Stopping Policy" : e_rst,
                       "Diff. b/w Expected Return" : e_rst - e_rt,
                       
                       "Standard Deviation with Stopping Policy" : std_rst,
                       "Diff. b/w Standard Deviation" : std_rst - std_rt,
                       
                       "Variance with Stopping Policy" : std_rst**2,
                       "Diff. b/w Variance" : std_rst**2 - std_rt**2,
                       
                       "Sharpe Ratio with Stopping Policy" : sr_rst,
                       "Diff. b/w Sharpe Ratio" : sr_rst - sr_rt,
                       
                       "p0" : p0,
                       "Stopping Ratio" : stopping_ratio
                      }
        
        return performance, rst
    
    

    def random_walk_theoretical_performance(self, st: pd.Series):
        """Implementation of Proposition 1.

        Parameters
        ----------
        st : Series
            Series contains proportions of assets allocated.
            
        Returns
        -------
        Dictionary contains performance metrics
        """

        # Calculate Random Walk Hypothesis theoretical performance
        pi = (self.strategy_return - self.rf_return).mean()
        std = self.strategy_return.std()
        p0 = (1 - st).sum() / len(st)

        performance = {"Diff. b/w Expected Return" : -p0*pi,
                       "Diff. b/w Variance" : -p0*(std**2) + p0*(1 - p0)*(pi**2),
                       "Diff. b/w Sharpe Ratio" : -pi/std + (-p0*pi + pi)/(np.sqrt(-p0*(std**2) + p0*(1 - p0)*(pi**2) + (std**2))),
                       "Stopping Ratio" : -pi
                      }

        return performance



    def regime_switching_theoretical_performance(self, st: pd.Series, It: pd.Series, mean_1: float, mean_2: float):
        """Implementation of Proposition 1.

        Parameters
        ----------
        st : Series
            Series contains proportions of assets allocated.
        
        It : Series
            Series contains states.

        mean_1 : float
            Mean of r1
        
        mean_2 : float
            Mean of r2

        Returns
        -------
        Dictionary contains performance metrics
        """

        # Calculate regime-switching theoretical performance
        p0 = (1 - st).sum() / len(st)
        p01 = ((st == 0)&(It == 1)).sum() / len(st)
        p02 = ((st == 0)&(It == 0)).sum() / len(st)
        p02_hat = p02 / p0

        pi1 = (mean_1 - self.rf_return).mean()
        pi2 = (mean_2 - self.rf_return).mean()

        performance = {"Diff. b/w Expected Return" : p01*(-pi1) + p02*(-pi2),
                       "Diff. b/w Sharpe Ratio" : (1 - p02_hat)*(-pi1) + p02_hat*(-pi2),
                      }

        return performance


    def add_custom_policy(self, policy: Callable):
        """Add a custom policy to the object
        
        Parameters
        ----------
        policy : Callable
            Function with returns Series as the first parameter and proportions Series as the output.
            
        Returns
        -------
        None
        """
        
        function_name = policy.__name__
        self.custom_policy_dict[function_name] = policy
        
        
    def evaluate(self, policy_name: str = None, config: dict = None, kappa: float = 0):
        """Caculate performance of the stopping policy.
        
        Parameters
        ----------
        policy_name : str
            Name of the policy
            
        config : dict
            Dictionary contains the policy function parameters except the first parameter. 
            
        kappa : float
            One-way transactions cost of a stop-loss event
            
        Returns
        -------
        1. Dictionary contains performance metrics
        2. rst
        """
        
        #Calculate st
        if policy_name == "simple_stop_loss_policy":
            st = self.simple_stop_loss_policy(**config)
        else:
            st = self.custom_policy_dict[policy_name](self.strategy_return, **config)
            
        self.st_dict[policy_name] = st
        
        #Calculate stopping policy performance
        performance, rst = self.stopping_policy_performance(st, kappa)
        self.performance_dict[policy_name] = performance
        self.rst_dict[policy_name] = rst
        
        return performance, rst
    
    
    # Get method
    def get_performance_result(self):
        """Return a dictionary containing all calculated performances."""
        return self.performance_dict

    def get_rst_result(self):
        """Return a dictionary containing all calculated returns series with the stop-loss policy."""
        return self.rst_dict



    # Visualization
    def trend_chart(self, policy_names: list, compounding: bool = False, height: int = 350, width: int = 800):
        """Trend chart of the result using Bokeh.
        
        Parameters
        ----------
        policy_names : list
            List of selected policy names
            
        compounding : bool, default False
            Whether returns are reinvested back into the account.
            
        height : int
            Height of the plot
            
        width : int
            Width of the plot
            
        Returns
        -------
        None
        """

        selected_rst_dict = {key: self.rst_dict[key] for key in policy_names}
        data = pd.DataFrame(selected_rst_dict)
        data["strategy_return"] = self.strategy_return
        
        if compounding:
            cum = (data + 1).cumprod()
        else:
            cum = data.cumsum() + 1
            
        if compounding:
            mdd = (cum/cum.cummax() - 1)
        else: 
            mdd = cum - cum.cummax()
            
        source = ColumnDataSource(data = cum)
        source_mdd = ColumnDataSource(data = mdd)
        
        p = figure(x_axis_type="datetime", title="Trend Line", plot_height=height, plot_width=width)
        p.xgrid.grid_line_color=None
        p.ygrid.grid_line_alpha=0.5
        p.xaxis.axis_label = 'Time'
        p.yaxis.axis_label = 'Total Return'
        
        p_mdd = figure(x_axis_type="datetime", title="Max Drawdown", plot_height=height, plot_width=width, x_range=p.x_range)
        p_mdd.xgrid.grid_line_color=None
        p_mdd.ygrid.grid_line_alpha=0.5
        p_mdd.xaxis.axis_label = 'Time'
        p_mdd.yaxis.axis_label = 'MDD'
        
        lines = []
        for i in range(len(cum.columns)):
            lines.append(p.line("Date", cum.columns[i], source=source, line_width=2, line_alpha=0.8, line_color = Spectral10[i%10], legend_label = cum.columns[i], muted_color = Spectral10[i%10], muted_alpha = 0.1))
            
        lines_mdd = []
        for i in range(len(mdd.columns)):
            lines_mdd.append(p_mdd.line("Date", mdd.columns[i], source=source_mdd, line_width=2, line_alpha=0.8, line_color = Spectral10[i%10], legend_label = mdd.columns[i], muted_color = Spectral10[i%10], muted_alpha = 0.1))
        
        p.legend.location = "top_left"
        p.legend.click_policy = "mute"
        
        p_mdd.legend.location = "bottom_left"
        p_mdd.legend.click_policy = "mute"

        LABELS = list(cum.columns)
        checkbox_group = CheckboxGroup(labels=LABELS)
        checkbox_group.active = list(range(len(LABELS)))

        code = """ for (var i = 0; i < lines.length; i++) {
                        lines[i].visible = false;
                        if (cb_obj.active.includes(i)){lines[i].visible = true;}
                    }
               """
        callback = CustomJS(code = code, args = {'lines': lines})
        checkbox_group.js_on_click(callback)
        
        callback = CustomJS(code = code, args = {'lines': lines_mdd})
        checkbox_group.js_on_click(callback)
        
        grid = gridplot([[p, checkbox_group], [p_mdd]])
        show(grid)
        
        
    def performace_table(self, policy_names: list, decimals: int, height: int = 600, width: int = 900):
        """Table of the result using Bokeh.
        
        Parameters
        ----------
        policy_names : list
            List of selected policy names

        decimals : int
            Number of decimal places to round

        height : int
            Height of the table
            
        width : int
            Width of the table
            
        Returns
        -------
        None
        """
        
        selected_performance_dict = {key: self.performance_dict[key] for key in policy_names}
        table = pd.concat([pd.DataFrame(v, index = [k]).T for k, v in selected_performance_dict.items()], axis=1)
        table.reset_index(inplace = True)
        table.rename(columns = {"index" : "metric"}, inplace = True)
        table = table.round(decimals)
        
        Columns = [TableColumn(field=Ci, title=Ci) for Ci in table.columns] # bokeh columns
        data_table = DataTable(columns=Columns, source=ColumnDataSource(table), fit_columns=False, height = height, width = width) # bokeh table
        data_table.index_position = None
        show(data_table)