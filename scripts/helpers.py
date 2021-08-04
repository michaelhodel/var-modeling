import sys
import os
import datetime
from subprocess import run
from pathlib import Path
import pylatex
import pandas as pd
import numpy as np
import yfinance as yf

from scripts import plotting


def log_version_info():
    """ which packages are used, which versions were used for development and for running """
    import pandas, numpy, scipy, matplotlib, sklearn, pylatex, yfinance, seaborn, QuantLib
    packages = [pandas, numpy, scipy, matplotlib, sklearn, pylatex, yfinance, seaborn, QuantLib]
    names = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'sklearn',
        'pylatex', 'yfinance', 'seaborn', 'QuantLib'
    ]
    versions = ['1.0.1', '1.18.0', '1.4.1', '3.1.2', '0.22.1', '1.4.1', '0.1.62', '0.10.0', '1.22']
    cols = columns = ['Version Developed in', 'Version Running']
    versions_info = pd.DataFrame({'Version Developed in': versions}, index=names, columns=cols)
    for package in packages:
        versions_info.loc[package.__name__, 'Version Running'] = package.__version__
    python_version = sys.version.split(' ')[0]
    print('Developed in Python v. 3.7.0, running using Python {}.\n'.format(python_version))
    print(versions_info.to_string())


# specify the data type for each parameter
data_type_mapper = {
    'start':                str,
    'end':                  str,
    'confidence':           float,
    'VaR_horizon':          int,
    'lmbda':                float,
    'rf_estim_method' :     str,
    'vol_period':           int,
    'corr_period':          int,
    'n_mc_scenarios':       int,
    'hs_period':            int,
    'n_bootstrap':          int,
    'lsmc_simulations':     int,
    'lsmc_granularity':     int,
    'amer_opt_p_method':    str,
    'crr_tree_height':      int,
    'option_pricing':       str,
    'bt_horizon':           lambda x: int(x) if x.isnumeric() else str(x),
    'estimation_window':    int,
    'riskfree_rate':        float,
    'jarque_bera_alpha':    float,
    'cov_estim_method':     str,
    'risk_factors_path':    lambda x: None if x.upper() == 'NONE' else str(x),
    'ptf_spec_path':        str,
    'nan_threshold':        float,
    'load_from_csv':        lambda x: True if x.upper() == 'TRUE' else False
}

# specify a sanity check for each parameter
checks_mapper = {
    'start':                lambda x: len(x) == 10 and x[4] == x[7] == '-',
    'end':                  lambda x: len(x) == 10 and x[4] == x[7] == '-',
    'confidence':           lambda x: 0 < x < 1,
    'VaR_horizon':          lambda x: 0 < x,
    'lmbda':                lambda x: 0 < x < 1,
    'rf_estim_method':      lambda x: x in ['sample', 'ledoit_wolf', 'garch', 'ewma'],
    'vol_period':           lambda x: 0 < x,
    'corr_period':          lambda x: 0 < x,
    'n_mc_scenarios':       lambda x: 0 < x,
    'hs_period':            lambda x: 0 < x,
    'n_bootstrap':          lambda x: 0 < x,
    'lsmc_simulations':     lambda x: 0 < x,
    'lsmc_granularity':     lambda x: 0 < x,
    'amer_opt_p_method':    lambda x: x in ['crr', 'lsmc'],
    'crr_tree_height':      lambda x: 0 < x,
    'option_pricing':       lambda x: x in ['quantlib', 'custom'],
    'bt_horizon':           lambda x: x > 1 if isinstance(x, int) else x in ['B', 'W', 'M'],
    'estimation_window':    lambda x: x > 3,
    'riskfree_rate':        lambda x: True,
    'jarque_bera_alpha':    lambda x: 0 < x < 1,
    'cov_estim_method':     lambda x: x in ['shrinkage', 'ledoit_wolf', 'sample'],
    'risk_factors_path':    lambda x: True,
    'ptf_spec_path':        lambda x: True,
    'nan_threshold':        lambda x: 0 <= x <= 1,
    'load_from_csv':        lambda x: isinstance(x, bool)
}

def read_models_config(path, report):
    """ read the models configuration file """
    # read the configuration file to a dataframe
    if path.endswith('.csv'):
        # read .csv file
        config_df = pd.read_csv(path, index_col=0, header=None).T
    elif path.endswith('.xlsx'):
        # read .xlsx file
        config_df = pd.read_excel(path, engine='openpyxl', index_col=0, header=None).T
    else:
        raise TypeError("Expected one of .csv or .xlsx, but found {}".format(path.split('.')[1]))
    
    # iterate over each parameter
    for parameter in config_df:
        # convert to the required data type
        parameter_value = data_type_mapper[parameter](config_df[parameter].item())
        # ensure the value fulfils the check
        assert checks_mapper[parameter](parameter_value)
        config_df.loc[:, parameter] = [parameter_value]
    
    config = {k: v[1] for k, v in config_df.to_dict().items()}
    print('\n'.join(['{}{}{}'.format(k, (20 - len(k)) * ' ', v) for k, v in config.items()]))
    
    if 'lmbda' not in config:
        config['lmbda'] = None
    
    # dictionary with parameter(s) for american option pricing
    amer_opt_params = {
        'option_pricing': config['option_pricing'],
        'amer_opt_p_method': config['amer_opt_p_method']
    }
    if config['amer_opt_p_method'] == 'crr':
        amer_opt_params['steps'] = config['crr_tree_height']
    elif config['amer_opt_p_method'] == 'lsmc':
        amer_opt_params['timeSteps'] = config['lsmc_granularity']
        amer_opt_params['requiredSamples'] = config['lsmc_simulations']
    
    # print and log configuration
    config_for_logging = pd.DataFrame([config.copy()], index=['Value']).T
    report.write_table(config_for_logging, 'Model Configuration', index_colname='Parameter')
    
    return config, amer_opt_params

    
def get_total_shares_outstanding(tickers):
    """ returns the current shares outstanding """
    total_shares_outstanding = pd.Series(index=tickers, dtype='float64')
    for ticker in tickers:
        total_shares_outstanding[ticker] = yf.Ticker(ticker).info['sharesOutstanding']
    return total_shares_outstanding


def float_formatter(v, lower_precision=False):
    """ round floats and use abbreviations for large numbers """
    sub = 2 if lower_precision else 0
    if abs(v) < 100:
        # round small floats to some decimals
        v = round(v, 4 - sub)
    elif 100 <= abs(v) < 1e6:
        # round larger floats to no decimals
        v = round(v)
    # denote very large values in millions (M), billions (B) or trillions (T)
    elif 1e6 <= abs(v) < 1e9:
        v = '{} M'.format(round(v / 1e6, 2 - sub))
    elif 1e9 <= abs(v) < 1e12:
        v = '{} B'.format(round(v / 1e9, 2 - sub))
    else:
        v = '{} T'.format(round(v / 1e12, 2 - sub))
    return v


class Report():
    """ creating a pdf report for tables and figures via LaTeX """
    def __init__(self, name):
        self.name = name
        self.doc = pylatex.Document('basic')
        self.doc.packages.append(pylatex.Package('placeins', options=['section']))
        self.doc.packages.append(pylatex.Package('geometry', options=['hmargin=0cm']))
        self.doc.packages.append(pylatex.Package('geometry', options=['vmargin=2cm']))
        self.doc.append(pylatex.Command('centering'))
        self.position_names = []
    
    def write(self, text):
        """ add text to the report """
        self.doc.append(pylatex.position.HorizontalSpace('1cm'))
        self.doc.append(text + '\n')
        self.doc.append(pylatex.position.HorizontalSpace('2cm'))
    
    def chapter(self, title):
        """ create a new section """
        self.doc.append(pylatex.NoEscape(r'\clearpage'))
        self.doc.append(pylatex.NoEscape(r'\section{' + title + '}'))
    
    def float_handler(self, df):
        """ handle float values """
        for col in df.columns:
            for idx in df.index:
                v = df.loc[idx, col]
                if isinstance(v, float):
                    # use empty strings for nans
                    if np.isnan(v):
                        v = ''
                    else:
                        v = float_formatter(v)
                df.loc[idx, col] = str(v)
        return df
    
    
    def write_table(self, dataframe, caption, head=False, index_colname='', index=True):
        """ add a dataframe to the report """
        # ensure deep copy
        dataframe_copy = dataframe.copy()
        df = pd.DataFrame(dataframe_copy.values)
        df.columns = dataframe_copy.columns.copy().tolist()
        df.index = dataframe_copy.index.copy()
        
        # handle datetime indices
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y-%m-%d')
        
        # use only first few rows if desired
        if head:
            df = df.head()
                
        # separate words by spaces instead of underscores
        df.index = list(map(lambda x: ' '.join(x.split('_')) if isinstance(x, str) else x, df.index))
        
        # use numbering instead of long asset names to save space
        for ax in [df.columns, df.index]:
            for i, c in enumerate(ax):
                if c in self.position_names:
                    ax.values[i] = 'Asset {}'.format(self.position_names.index(c))
        
        # handle floats (nans, rounding, etc.)
        df = self.float_handler(df)
        
        # create table
        table = pylatex.Table(position="h")
        table.append(pylatex.Command('centering'))
        tabular = pylatex.Tabular(('|c|' if index else '|') + df.shape[1] * 'c|')
        tabular.add_hline()
        
        # add header
        header = ([index_colname] if index else []) + df.columns.tolist()
        tabular.add_row(header, mapper=pylatex.utils.bold)
        
        # add rows
        for i, row in df.iterrows():
            tabular.add_hline()
            if index:
                lab = str(i).replace('%', '\%').replace('Correlation', 'Corr.')
                idx = r'\textbf{' + lab + r'}'
            line = [v.replace('_', '\_') if isinstance(v, str) else v for v in row.tolist()]
            r = '&'.join(([idx] if index else []) + line) + r'\\%'
            tabular.append(pylatex.NoEscape(r))
        
        # add extra row with cells with dots to indicate continuation
        if head:
            tabular.add_hline()
            tabular.add_row((df.shape[1] + (1 if index else 0)) * ['...'])
        
        tabular.add_hline()
        table.append(tabular)
        table.add_caption(caption=caption)
        self.doc.append(pylatex.NoEscape(r'\FloatBarrier'))
        self.doc.append(pylatex.Command('centering'))
        self.doc.append(table)
    
    
    def write_plot(self, caption):
        """ add a plot to the report """
        self.doc.append(pylatex.NoEscape(r'\FloatBarrier'))
        with self.doc.create(pylatex.Figure(position = 'h')) as figure:
            figure.add_plot(width=pylatex.NoEscape(r'1.0\linewidth'))
            figure.add_caption(caption=caption)
    
    
    def save(self, overwrite_existing=True):
        """ generate report as pdf """
        # generate .tex file
        self.doc.generate_tex('temp')
        # generate .pdf file
        _ = run(['pdflatex', '-interaction=nonstopmode', 'temp.tex'])
        # move file
        name = self.name
        # add timestamp to prevent overwriting
        if not overwrite_existing:
            name += datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        # rename report
        Path('temp.pdf').rename('outputs/{}.pdf'.format(name))
        # remove temporary files
        for suffix in ['aux', 'log', 'tex']:
            os.remove('temp.' + suffix)
