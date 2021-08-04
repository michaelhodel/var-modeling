import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.covariance import ledoit_wolf
import mgarch
import yfinance as yf

from scripts import plotting


class RiskFactorsAndPositions():
    """ loading position specifications and risk factor prices time series """
    def __init__(self, ptf_spec_path, load_from_csv, risk_factors_path,
                 start, end, nan_threshold, report):
        self.ptf_spec_path = ptf_spec_path
        self.load_from_csv = load_from_csv
        self.risk_factors_path = risk_factors_path
        self.start = start
        self.end = end
        self.nan_threshold = nan_threshold
        self.report = report
    
    
    def run(self):
        """ loading of position configurations, loading and cleaning of risk factors, returns """
        # read and display all position configurations
        self.read_positions_configurations()

        # get all necessary tickers
        self.get_necessary_tickers()
        
        # load risk factor prices
        self.load_risk_factor_prices()
        
        # organize risk factor prices
        self.organize_risk_factor_prices()
        
        # remove all assets from configurations for which underlyers are not available
        self.remove_assets()
        
        # plot price time series, computation and plotting of log returns
        self.prices_and_returns_time_series()
        
        return self.rf, self.rf_rets, self.position_configurations, self.tickers
    
    
    def read_positions_configurations(self):
        """ get a list of all positions configurations """
        # read the configuration file to a dataframe
        if self.ptf_spec_path.endswith('.csv'):
            # read .csv file
            self.position_configurations = pd.read_csv(self.ptf_spec_path)
        elif self.ptf_spec_path.endswith('.xlsx'):
            # read .xlsx file
            self.position_configurations = pd.read_excel(self.ptf_spec_path, engine='openpyxl')
        else:
            msg = 'Expected one of .csv or .xlsx, but found {}'
            raise TypeError(msg.format(self.ptf_spec_path.split('.')[1]))

        print('\nPosition Configurations:')
        print(self.position_configurations.to_string())
    
    
    def get_necessary_tickers(self):
        """ get all necessary risk factor tickers """
        # a list containing all tickers (risk factors) required
        all_tickers = []

        # iterate over all position configurations
        for i, pos_config in self.position_configurations.iterrows():
            if pos_config['type'] in ['Equity', 'Future']:
                # add the ticker for the price if the position is an equity
                all_tickers.append((pos_config['price_ticker'], 'price'))
            elif pos_config['type'] == 'Option':
                # add the tickers for the price, rate and volatility if the position is an option
                all_tickers.append((pos_config['price_ticker'], 'price'))
                all_tickers.append((pos_config['rate_ticker'], 'rate'))
                all_tickers.append((pos_config['vol_ticker'], 'vol'))
            elif pos_config['type'] == 'Bond':
                # add the ticker for the yield if the position is an equity
                all_tickers.append((pos_config['yield_ticker'], 'yield'))

        # remove duplicates
        self.tickers = list(set(all_tickers))

        # print all required tickers / risk factors and their "types"
        print('\nRisk Factors:')
        print(self.tickers)
    
    
    def remove_assets(self):
        """ remove assets with no price time series available for underlyers """
        # list of available tickers
        available_tickers = self.rf.columns.tolist()
        
        # list of position indices for which underliers are missing
        unavailable_idx = []
        
        # iterate over all position configurations
        for i, pos_config in self.position_configurations.iterrows():
            # determine the tickers for the required underlying levels
            required_tickers = []
            if pos_config['type'] in ['Equity', 'Future']:
                required_tickers.append(pos_config['price_ticker'])
            elif pos_config['type'] == 'Option':
                required_tickers.append(pos_config['price_ticker'])
                required_tickers.append(pos_config['rate_ticker'])
                required_tickers.append(pos_config['vol_ticker'])
            elif pos_config['type'] == 'Bond':
                required_tickers.append(pos_config['yield_ticker'])
            # iterate over each required ticker
            for required_ticker in required_tickers:
                # add to list of position configuration indices for which levels are missing
                if required_ticker not in available_tickers:
                    unavailable_idx.append(i)
                    # break loop to not add index twice in case of multiple missing tickers
                    break
        
        # drop position configurations for which levels are missing
        self.position_configurations = self.position_configurations.drop(unavailable_idx)
        # reset index
        self.position_configurations.index = list(range(self.position_configurations.shape[0]))
        # log info in case positions had to be dropped
        if len(unavailable_idx) > 0:
            n = len(unavailable_idx)
            msg = '\nRemoved {} assets due to unavailable underlying prices.'.format(n)
            print(msg)
            self.report.write(msg)
    
    
    def load_risk_factor_prices(self):
        """ load risk factor prices either from csv of yfinance """
        if self.load_from_csv:
            # read a csv with a column containing the time series for each risk factor
            if self.risk_factors_path.endswith('.csv'):
                self.rf = pd.read_csv(self.risk_factors_path, index_col=0)
            elif self.risk_factors_path.endswith('.xlsx'):
                self.rf = pd.read_excel(self.risk_factors_path, engine='openpyxl', index_col=0)
            else:
                msg = 'Expected one of .csv or .xlsx, but found {}'
                raise TypeError(msg.format(self.risk_factors_path.split('.')[1]))
        else:
            # download the time series from yfinance, use adjusted closing price
            tickers = [t[0] for t in self.tickers]
            df = yf.download(tickers=tickers, start=self.start, end=self.end, adjusted=True)

            # use closing price and reorder time series by descending date
            self.rf = df['Adj Close'].reindex(index=df.index[::-1])
    
        
    def organize_risk_factor_prices(self):
        """ removal of bad quality risk facotrs, interpolation, filling and sorting """
        # indicate for each risk factor whether or not share of NaNs exceeds threshold
        indicator = self.rf.isna().sum() <= self.rf.shape[0] * self.nan_threshold

        if not indicator.all():
            # remove all risk factors with more than nan_threshold of missing values
            self.rf = self.rf.loc[:, indicator.loc[indicator].index]

            # create a list of risk factors that were removed
            removed_rfs = indicator.loc[indicator == False].index.tolist()
            msg = '\nRemoved {} due to more than {}% missing values.\n'
            msg = msg.format(removed_rfs, self.nan_threshold * 100)
            print(msg)
            self.report.write(msg)

        # linearly interpolate missing values between valid values
        self.rf.interpolate(inplace=True, limit_direction='both', limit_area='inside')

        # forward- and backward-fill missing values not enclosed within valid values
        self.rf = self.rf.ffill().bfill()

        # sort risk factors by order of provided tickers
        tickers = [tup[0] for tup in self.tickers]
        self.rf = self.rf[[t for t in tickers if t in self.rf.columns.tolist()]]

        # display the prices for the first few dates and log to report
        print('\nRisk Factors Prices:')
        print(self.rf.head().to_string())
        self.report.write_table(self.rf, 'Risk Factor Prices', head=True, index_colname='Date')
    
    
    def prices_and_returns_time_series(self):
        """ plot price time series, computation and plotting of log returns """
        # plot the price time series
        plotting.plot_time_series(self.rf, 'Prices', 'Adj Close Price', self.report)

        # compute logarithmic returns and display first few
        self.rf_rets = np.log(self.rf / self.rf.shift(-1))[:-1]
        print('Risk Factor Log-Returns:')
        print(self.rf_rets.head().to_string())
        self.report.write_table(
            self.rf_rets, 'Risk Factor Log Returns', head=True, index_colname='Date')

        # plot the log return time series
        plotting.plot_time_series(self.rf_rets, 'Returns', 'Log-Return', self.report)


class CorrelationAndVolatilityEstimators():
    """ volatility and correlation estimation methods for the risk factor log returns """
    def __init__(self, rf_estim_method, vol_period, corr_period, rf_rets, report, lmbda=None):
        self.rf_estim_method = rf_estim_method
        self.vol_period = vol_period
        self.corr_period = corr_period
        self.rets = rf_rets
        self.lmbda = lmbda
        self.report = report
    
    
    def correlation_from_covariance(self, cov_mat):
        """ computes the correlation matrix from the covariance matrix """
        diag_sqrt_inv = np.diag(1 / np.sqrt(np.diag(cov_mat)))
        return np.dot(np.dot(diag_sqrt_inv, cov_mat), diag_sqrt_inv)
    
    
    def sample_estimator(self):
        """ sample historical volatilities and sample correlation matrix """
        # compute the sample standard deviations
        self.volatilities['sample'] = np.std(self.rets[:self.vol_period], ddof=1)
        # compute the sample correlation matrix
        self.correlation_matrices['sample'] = self.rets[:self.corr_period].corr()
    
    
    def ledoit_wolf_estimator(self):
        """ volatilities and correlation matrix from Ledoit-Wolf covariance matrix"""
        # compute the Ledoit-Wolf covariance matrix for volatility estimation
        vol_cov_mat = ledoit_wolf(self.rets[:self.vol_period])[0]
        
        # compute the volatilities
        self.volatilities['ledoit_wolf'] = pd.Series(np.diag(vol_cov_mat) ** 0.5, self.rets.columns)
        
        cov_mat = ledoit_wolf(self.rets[:self.corr_period])[0]

        # compute the correlation matrix from the Ledoit-Wolf covariance matrix
        corr_mat = self.correlation_from_covariance(cov_mat)
        ax = self.rets.columns
        self.correlation_matrices['ledoit_wolf'] = pd.DataFrame(corr_mat, ax, ax)
    
    
    def garch_estimator(self):
        """ volatilities and correlation matrix from GARCH covariance matrix """
        # DCC-GARCH model for volatilities
        rf_vol_rets = self.rets[:self.vol_period][::-1]
        vol_mgarch_model = mgarch.mgarch()
        _ = vol_mgarch_model.fit(rf_vol_rets)
        volatilities = np.sqrt(vol_mgarch_model.predict()['cov'].diagonal())
        self.volatilities['garch'] = pd.Series(volatilities, self.rets.columns)
        
        # DCC-GARCH model for correlation
        rf_corr_rets = self.rets[:self.corr_period][::-1]
        corr_mgarch_model = mgarch.mgarch()
        _ = corr_mgarch_model.fit(rf_corr_rets)
        correlations = self.correlation_from_covariance(corr_mgarch_model.predict()['cov'])
        ax = self.rets.columns
        self.correlation_matrices['garch'] = pd.DataFrame(correlations, ax, ax)
    
    
    def ewma_estimator(self):
        """ volatilities and correlation matrix from EWMA process """
        # fit two separate models for volatilities and correlation matrix estimation
        ewma = {'vol': self.vol_period, 'corr': self.corr_period}
        
        for t, p in ewma.items():
            # get the log returns of the risk factors in a numpy array
            y = self.rets[:p].values

            # create a numpy array for the EWMA RF variances
            rf_ewma_vars = np.full(y.shape, np.nan)

            # compute the initial covariance matrix and seed the array
            S = np.cov(self.rets[:p], rowvar = False)
            rf_ewma_vars[y.shape[0] - 1,] = np.diag(S)

            # iterate over the time series'
            lmbda_i = 1 - self.lmbda
            for i in range(y.shape[0] - 2, -1, -1):
                # compute the updated covariance matrix
                S = self.lmbda * S + lmbda_i * np.asmatrix(y[i + 1]).T * np.asmatrix(y[i + 1])
                # extract the variances
                rf_ewma_vars[i, ] = np.diag(S)
            
            if t == 'vol':
                # compute the ewma volatilities for the most recent date with available variances
                self.volatilities['ewma'] = pd.Series(rf_ewma_vars[0] ** 0.5, self.rets.columns)
            elif t == 'corr':
                # compute the correlation matrix from the covariance matrix S
                corr_mat = self.correlation_from_covariance(S)
                ax = self.rets.columns
                self.correlation_matrices['ewma'] = pd.DataFrame(corr_mat, ax, ax)
    
    
    def run(self):
        """ use each estimation method """
        # initialize dictionaries for the volatilities and correlation matrices
        self.volatilities = dict()
        self.correlation_matrices = dict()
        
        # apply each estimation method
        self.sample_estimator()
        self.ledoit_wolf_estimator()
        #self.garch_estimator()
        self.ewma_estimator()
        
        # print and log volatilities and correlation matrices to the report
        self.log()
        
        # extract volatilities and correlation matrix from chosen estimation method
        rf_hist_vols = self.volatilities[self.rf_estim_method]
        rf_corr_mat = self.correlation_matrices[self.rf_estim_method]
        
        return rf_hist_vols, rf_corr_mat
    
    
    def log(self):
        """ display results and save to report """
        # display volatilities
        vol_df = pd.DataFrame(self.volatilities)
        print('Risk Factor Volatilites:\n')
        print(vol_df.to_string())
        self.report.write_table(vol_df, 'Risk Factor Volatilities')

        # plot correlation matrices as heatmaps
        for method, matrix in self.correlation_matrices.items():
            name = ' '.join([w.capitalize() for w in method.split('_')])
            plotting.plot_corr_mat(matrix, '{} Correlation Matrix'.format(name), self.report)
