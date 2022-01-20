import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource, Panel, Tabs, ColorBar, LinearColorMapper
from bokeh.transform import transform
from bokeh.layouts import row, column
from bokeh.palettes import Spectral10, Reds
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn


def generate_date(start_date: str, end_date: str, step:int = 1, ignore_weekends: bool = True):
    """Generate list of datetimes

    Parameters
    ----------
    start_date : str in %Y%m%d format
        Start date
        
    end_date : str in %Y%m%d format
        End date
        
    step : int
        Interval between two dates
        
    ignore_weekends : bool, default True
        Whether the process ignores weekends.
        
    Returns
    -------
    List of datetimes
    """
    
    # Generate sequence of dates
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d').date()
    end_date = datetime.datetime.strptime(end_date, '%Y%m%d').date()
    number_of_days = ((end_date - start_date).days + 1)
    dates = [start_date + datetime.timedelta(days = i) for i in range(number_of_days)]
    
    if ignore_weekends:
        dates = [d for d in dates if not d.isoweekday() in [6,7]]
    
    dates = [dates[i] for i in range(len(dates)) if i%step == 0]
    
    return dates


def random_walk_generater(mean: float, std: float, start_date: str, end_date: str, step:int = 1, ignore_weekends: bool = True):
    """Generate Series contains returns through Random Walk Hypothesis
    
    Method:
    rt = µ + et, et ∼ White Noise(0, σ^2),

    Parameters
    ----------
    mean : float
        Mean of the process

    std : float
        Standard deviation of the white noise
        
    start_date : str in %Y%m%d format
        Start date of the process
        
    end_date : str in %Y%m%d format
        End date of the process
        
    step : int
        Interval between two dates
        
    ignore_weekends : bool, default True
        Whether the process ignores weekends.
        
    Returns
    -------
    Series contains returns
    """
    
    # Generate sequence of dates
    dates = generate_date(start_date, end_date, step, ignore_weekends)
    
    # Generate white noise
    white_noise = np.random.normal(0, std, size = len(dates))
    
    # Generate returns Series
    rt = pd.Series(mean + white_noise, index = dates, name = 'random_walk')
    rt.index = pd.to_datetime(rt.index)
    rt.index.name = "Date"

    return rt


def ar_1_return_generater(mean: float, std: float, rho: float, r0: float, start_date: str, end_date: str, step:int = 1, ignore_weekends: bool = True):
    """Generate Series contains returns through the process described in Kaminski, Kathryn M., and Andrew W. Lo. (2013) equation 14.
    
    Method:
    rt = µ + ρ(rt-1 - µ) + et, et ∼ White Noise(0, σ^2),

    Parameters
    ----------
    mean : float
        Mean of the process

    std : float
        Standard deviation of the white noise
        
    rho : float
        ρ ∈ (−1, 1)
    
    r0 : float
        Initial value of the series
        
    start_date : str in %Y%m%d format
        Start date of the process
        
    end_date : str in %Y%m%d format
        End date of the process
        
    step : int
        Interval between two dates
        
    ignore_weekends : bool, default True
        Whether the process ignores weekends.
        
    Returns
    -------
    Series contains returns
    """
    
    # Generate sequence of dates
    dates = generate_date(start_date, end_date, step, ignore_weekends)
    
    # Generate white noise
    white_noise = np.random.normal(0, std, size = len(dates))
    
    # Generate returns Series
    rt = pd.Series(r0, index = dates, name = 'ar_1')
    for i in range(1, len(rt)):
        rt.iloc[i] = mean + rho*(rt.iloc[i - 1] - mean) + white_noise[i]
        
    rt.index = pd.to_datetime(rt.index)
    rt.index.name = "Date"

    return rt


def regime_switching_return_generater(mean_1: float, std_1: float, mean_2: float, std_2: float, I0: float, trans_prob_matrix: np.array, 
                                      start_date: str, end_date: str, step:int = 1, ignore_weekends: bool = True):
    """Generate Series contains returns through the process described in Kaminski, Kathryn M., and Andrew W. Lo. (2013) equation 19.
    
    Method:
    rt = It*r1t + (1 − It)*r2t, rit ~ N(µi, σi^2)
    
                It+1 = 1     It+1 = 0
    A ≡ It = 1  [p11          p12]
        
        It = 0  [p21          p22]
        
    where A is the Markov transition probabilities matrix that governs the transitions between the two states.
    
    Parameters
    ----------
    mean_1 : float
        Mean of r1

    std_1 : float
        Standard deviation of r1
        
    mean_2 : float
        Mean of r2

    std_2 : float
        Standard deviation of r2
        
    I0 : float
        Initial value of the state
        
    trans_prob_matrix : 2x2 np.array
        Markov transition probabilities matrix
        
    start_date : str in %Y%m%d format
        Start date of the process
        
    end_date : str in %Y%m%d format
        End date of the process
        
    step : int
        Interval between two dates
        
    ignore_weekends : bool, default True
        Whether the process ignores weekends.
        
    Returns
    -------
    1. Series contains returns
    2. Series contains states
    """
    
    # Generate sequence of dates
    dates = generate_date(start_date, end_date, step, ignore_weekends)
    
    # Generate r1, r2
    r1 = np.random.normal(mean_1, std_1, size = len(dates))
    r1 = pd.Series(r1, index = dates)
    
    r2 = np.random.normal(mean_2, std_2, size = len(dates))
    r2 = pd.Series(r2, index = dates)
    
    # Generate returns Series
    It = pd.Series(I0, index = dates)
    for i in range(1, len(It)):
        if It.iloc[i - 1] == 1:
            It.iloc[i] = np.random.choice([1, 0], 1, p = trans_prob_matrix[0])[0]
        else:
            It.iloc[i] = np.random.choice([1, 0], 1, p = trans_prob_matrix[1])[0]
            
    rt = It*r1 + (1 - It)*r2
    rt.index = pd.to_datetime(rt.index)
    rt.index.name = "Date"
    rt.name = 'regime_switching'

    return rt, It


def trend_chart(returns_series, compounding: bool = False, height: int = 350, width: int = 800):
    """Trend chart of the result using Bokeh.
    
    Parameters
    ----------
    returns_series : Pandas Series
        Series contains returns.
    
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

    data = returns_series
    
    if compounding:
        cum = (data + 1).cumprod()
    else:
        cum = data.cumsum() + 1
        
    cum = pd.DataFrame(cum)
    source = ColumnDataSource(data = cum)

    p = figure(x_axis_type="datetime", title="Trend Line", plot_height=height, plot_width=width)
    p.xgrid.grid_line_color=None
    p.ygrid.grid_line_alpha=0.5
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Total Return'
    lines = []
    for i in range(len(cum.columns)):
        lines.append(p.line("Date", cum.columns[i], source=source, line_width=2, line_alpha=0.8, line_color = Spectral10[i%10], legend_label = cum.columns[i], muted_color = Spectral10[i%10], muted_alpha = 0.1))

    p.legend.location = "top_left"
    p.legend.click_policy="mute"

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

    layout = row(p, checkbox_group)
    show(layout)