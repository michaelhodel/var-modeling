import pandas as pd
import numpy as np
from scipy.stats import norm, pearsonr

from scripts import pricing, plotting


class ParametricValueAtRisk():
    """ Parametric Value at Risk Model """
    def __init__(self, report, VaR_horizon, confidence, rf, rf_hist_vols,
                 rf_corr_mat, tickers, position_names, positions_df):
        self.report = report
        self.VaR_horizon = VaR_horizon
        self.confidence = confidence
        self.rf = rf
        self.rf_hist_vols = rf_hist_vols
        self.rf_corr_mat = rf_corr_mat
        self.tickers = tickers
        self.position_names = position_names
        self.positions_df = positions_df
    
    
    def run(self):
        """ calculate value at risk metrics """
        self.report.chapter('Parametric Value-at-Risk')
        self.calculate_rf_information()
        self.calcualte_rf_sensitivities()
        self.calculate_rf_volatilities()
        self.calcualte_asset_covariance_matrix()
        self.calculate_ptf_var()
        self.calculate_asset_metrics()
        self.calculate_component_var()
        self.calculate_incremental_var()
        self.calculate_diversification_metrics()
        self.plot_pdf()
        self.waterfall_plots()
    
    
    def log(self, df, title, head=False, index=True):
        """ print dataframe and save to report """
        print('\n\n{}:\n'.format(title))
        print(df.head().to_string() if head else df.to_string())
        self.report.write_table(df, title, head=head, index=index)
    
    
    def calculate_rf_information(self):    
        """ additional information on risk factors """
        self.rf_info = pd.DataFrame(columns=self.rf.columns)

        # use most recent data
        self.rf_info.loc['Current Level', :] = self.rf.iloc[0]

        # use historical volatilities
        self.rf_info.loc['Volatility (%)', :] = self.rf_hist_vols

        # adjust current levels for rate and yield levels
        for risk_factor, risk_factor_type in self.tickers:
            if risk_factor_type in ['rate', 'yield']:
                self.rf_info.loc['Current Level', risk_factor] *= 100

        # compute the volailities in units
        levels = self.rf_info.loc['Current Level', :]
        self.rf_info.loc['Volatility (units)', :] = self.rf_hist_vols * levels

        # adjust unit volatilities for rate and yield levels
        for risk_factor, risk_factor_type in self.tickers:
            if risk_factor_type in ['rate', 'yield']:
                self.rf_info.loc['Volatility (units)', risk_factor] /= 10000

        self.log(self.rf_info, 'Risk Factor Information')
    
    
    def calcualte_rf_sensitivities(self):
        """ asset price sensitivities to unit changes in risk factors """
        # create a dataframe, a row for each asset and a column for each risk factor
        self.rf_sensitivities = pd.DataFrame(None, self.position_names, self.rf.columns)

        # iterate over the positions (assets)
        for i, pos in self.positions_df.iteritems():
            # retrieve the product type, price ticker and position name
            product_type = pos.loc['position', '', 'product_type']
            price_ticker = pos.loc['position', '', 'price_ticker']
            asset = pos.loc['position', '', 'asset']

            # compute the sensitivities for equity positions
            if product_type in ['Equity', 'Future']:
                delta = pos.loc['price_sensitivities', 'pos', 'delta']
                self.rf_sensitivities.loc[asset, price_ticker] = delta

            # compute sensitivities for option positions
            if product_type == 'Option':
                rate_ticker = pos.loc['position', '', 'rate_ticker']
                vol_ticker = pos.loc['position', '', 'vol_ticker']
                gamma = pos.loc['price_sensitivities', 'pos', 'gamma']
                delta = pos.loc['price_sensitivities', 'pos', 'delta']
                pv01 = pos.loc['price_sensitivities', 'pos', 'pv01']
                vega = pos.loc['price_sensitivities', 'pos', 'vega']
                
                self.rf_sensitivities.loc[asset, price_ticker] = gamma + delta
                self.rf_sensitivities.loc[asset, rate_ticker] = pv01
                self.rf_sensitivities.loc[asset, vol_ticker] = vega

            # compute the sensitivities for bond positions
            if product_type == 'Bond':
                yield_ticker = pos.loc['position', '', 'yield_ticker']
                pv01 = pos['price_sensitivities', 'pos', 'pv01']
                self.rf_sensitivities.loc[asset, yield_ticker] = pv01

        # fill remaining values with zeros
        self.rf_sensitivities.fillna(value=0.0, inplace=True)

        # compute the aggregated sensitivities over each asset
        self.rf_sensitivities.loc['Portfolio', :] = self.rf_sensitivities.sum()
        self.log(self.rf_sensitivities, 'Risk Factor Sensitivities')
    
    
    def calculate_rf_volatilities(self):
        """ dollar volatilities """
        self.rf_volatilities = pd.DataFrame(None, self.position_names, self.rf.columns)

        # iterate over each position
        for i, pos in self.positions_df.iteritems():
            # iterate over each risk factor
            for risk_factor in self.rf:
                # get unit volatility, asset, product type and ticker
                level = self.rf_info.loc['Current Level', risk_factor]
                vol = self.rf_info.loc['Volatility (%)', risk_factor]
                level_vol_prod = level * vol
                    
                asset = pos.loc['position', '', 'asset']
                prod_type = pos.loc['position', '', 'product_type']
                price_ticker = pos.loc['position', '', 'price_ticker']

                # dollar volatility for options and price tickers
                if prod_type == 'Option' and risk_factor == price_ticker:
                    delta = pos.loc['price_sensitivities', 'pos', 'delta']
                    gamma = pos.loc['price_sensitivities', 'pos', 'gamma']
                    dollar_vol = delta * level_vol_prod + 0.5 * gamma * level_vol_prod ** 2
                    self.rf_volatilities.loc[asset, risk_factor] = dollar_vol

                # dollar volatility for bonds and yield tickers
                elif prod_type == 'Bond':
                    yield_ticker = pos.loc['position', '', 'yield_ticker']
                    if risk_factor == yield_ticker:
                        pv01 = pos.loc['price_sensitivities', 'pos', 'pv01']
                        cvex01 = pos.loc['price_sensitivities', 'pos', 'cvex01']
                        dollar_vol = pv01 * level_vol_prod + cvex01 * level_vol_prod ** 2
                        self.rf_volatilities.loc[asset, risk_factor] = dollar_vol

                # dollar volatilities for remaining (position, risk factor) combinations
                else:
                    dollar_vol = level_vol_prod * self.rf_sensitivities.loc[asset, risk_factor]
                    self.rf_volatilities.loc[asset, risk_factor] = dollar_vol
                        
        # fill remaining values with zeros
        self.rf_volatilities.fillna(value=0.0, inplace=True)

        # compute the aggregated volatilities over each asset
        self.rf_volatilities.loc['Net', :] = self.rf_volatilities.sum()
        self.log(self.rf_volatilities, 'Dollar Volatilities')
    
    
    def calcualte_asset_covariance_matrix(self):
        """ asset covariance matrix from correlation matrix and volatilities """
        D = self.rf_volatilities.drop('Net')
        asset_covariance_matrix = np.dot(np.dot(D, self.rf_corr_mat), D.T)
        ax = self.position_names
        self.asset_cov_mat = pd.DataFrame(asset_covariance_matrix, ax, ax)
        self.log(self.asset_cov_mat, 'Asset Covariance Matrix')
    
    
    def calculate_ptf_var(self):
        """ n-day portfolio VaR """
        # scalar to convert 1-day VaR measures to n-day VaR measures
        self.n_day_var_scalar = self.VaR_horizon ** 0.5
        
        # compute the volatility multiplier
        self.percent_point = norm.ppf(1 - self.confidence)
        
        # compute the portfolio volatility
        D = self.rf_volatilities.loc['Net', :]
        self.ptf_vol = np.sqrt(np.dot(np.dot(D, self.rf_corr_mat), D.T))
        
        self.ptf_var = self.percent_point * self.ptf_vol * self.n_day_var_scalar
        conf_pct = round(self.confidence * 100, 1)
        print('Portfolio {}% VaR ($): {}'.format(conf_pct, round(self.ptf_var, 4)))
    
    
    def calculate_asset_metrics(self):
        """ volatilities, correlation and individual VaR """
        # compute the volatilities
        volatilities = np.sqrt(np.diag(self.asset_cov_mat))

        # compute the correlations
        correlation = (self.asset_cov_mat.sum() / (volatilities * self.ptf_vol)).values

        # compute the n-day individual VaR
        individual_var = self.percent_point * volatilities * self.n_day_var_scalar
        
        # add values to dataframe
        self.asset_metrics = pd.DataFrame(columns=self.position_names)
        self.asset_metrics.loc['Volatility (%)', :] = volatilities
        self.asset_metrics.loc['Correlation', :] = correlation
        self.asset_metrics.loc['Individual VaR', :] = individual_var

        self.log(self.asset_metrics, 'Asset Metrics')
    
    
    def calculate_component_var(self):
        """ component VaR """
        self.component_var = pd.DataFrame(columns=self.position_names)

        # compute the n-day component VaR as product of correlation and individual VaR
        pct_vols = self.asset_metrics.drop(['Volatility (%)'])
        self.component_var.loc[''] = pct_vols.prod()

        # compute the aggregatd component VaR
        self.component_var['Sum'] = self.component_var.sum(axis=1)
        self.log(self.component_var, 'Component VaR', index=False)
    
    
    def calculate_incremental_var(self):
        """ incremental VaR """
        self.incremental_var = pd.DataFrame(columns=self.position_names)

        # computes the sum over all non-asset covariances
        cov_sum = lambda asset, cov_mat: cov_mat.drop(asset).drop(asset, axis=1).sum().sum()

        # compute the n-day incremental VaR for each asset
        cov_sums = np.array([cov_sum(asset, self.asset_cov_mat) for asset in self.position_names])
        asset_vars = self.percent_point * np.sqrt(cov_sums) * self.n_day_var_scalar
        self.incremental_var.loc['', :] = self.ptf_var - asset_vars
        self.log(self.incremental_var, 'Incremental VaR', index=False)
    
    
    def calculate_diversification_metrics(self):
        """ compute diversification metrics """
        # initialize a dataframe for diversification metrics
        cols = ['Undiv. VaR', 'Div. VaR', 'Div. Effect ($)', 'Div. Effect (%)', 'Div. Ratio']
        self.diversification = pd.DataFrame(None, [''], cols)

        # compute the undiversified VaR as sum of individual VaR
        undiv_var = self.asset_metrics.loc['Individual VaR', :].sum()
        self.diversification['Undiv. VaR'] = undiv_var

        # add the diversified VaR
        self.diversification['Div. VaR'] = self.ptf_var

        # compute the diversification effects ($)
        self.diversification['Div. Effect ($)'] = undiv_var - self.ptf_var

        # compute the diversification effects (%)
        self.diversification['Div. Effect (%)'] = (undiv_var - self.ptf_var) / undiv_var
        
        # compute the diversification ratio
        self.diversification['Div. Ratio'] = self.ptf_var / undiv_var

        self.log(self.diversification, 'Diversification Measures', index=False)
    
    
    def plot_pdf(self):
        """ P&L probability density function """
        title = 'P&L Probability Density Function'
        plt_lgnd = '{}% {}-Day VaR: {} $'.format(
            self.confidence * 100, self.VaR_horizon, round(self.ptf_var, 2)
        )
        plotting.plot_pnl_pdf(
            title, self.ptf_vol, self.ptf_var, plt_lgnd, self.report
        )
    
    
    def waterfall_plots(self):
        """ waterfall plots """
        # plot diversification effects
        pvar_div_effs = self.diversification.drop(columns=['Div. Ratio'])
        n_day_label = '{}-Day'.format(self.VaR_horizon)
        ptf_market_value = self.positions_df.loc['position', '', 'market_value'].sum()
        plotting.div_effect_plots(
            pvar_div_effs, self.asset_metrics, ptf_market_value,
            self.position_names, n_day_label, self.report
        )
        
        # plot VaR decomposition
        plotting.var_decomp_plots(
            self.component_var, ptf_market_value, self.position_names,
            n_day_label, self.report
        )


class MonteCarloValueAtRisk():
    """ Monte Carlo Value at Risk Model """
    def __init__(self, report, VaR_horizon, confidence, rf_hist_vols,
                 rf_corr_mat, position_names, n_mc_scenarios, rf,
                 positions_df, ptf_market_value, amer_opt_params):
        self.report = report
        self.VaR_horizon = VaR_horizon
        self.confidence = confidence
        self.rf_hist_vols = rf_hist_vols
        self.rf_corr_mat = rf_corr_mat
        self.position_names = position_names
        self.n_mc_scenarios = n_mc_scenarios
        self.rf = rf
        self.positions_df = positions_df
        self.ptf_market_value = ptf_market_value
        self.amer_opt_params = amer_opt_params
    
    
    def run(self):
        """ calculate value at risk metricsc """
        # set seed for reproducibility
        np.random.seed(42)
        self.report.chapter('Monte Carlo Value-at-Risk')
        self.compute_cholesky_matrix()
        self.calculate_shifted_levels()
        self.get_full_revaluation_pnl()
        self.get_ptf_pnl()
        self.calculate_var()
        self.calculate_sub_ptf_pnl()
        self.calculate_sub_ptf_var()
        self.calculate_component_var()
        self.calculate_incremental_var()
        self.calculate_diversification_metrics()
        self.plot_hist()
        self.waterfall_plots()
    
    
    def log(self, df, title, head=False, index=True):
        """ print dataframe and save to report """
        print('\n\n{}:\n'.format(title))
        print(df.head().to_string() if head else df.to_string())
        self.report.write_table(df, title, head=head, index=index)
    
    
    def compute_cholesky_matrix(self):
        """ compute the cholesky decomposition and get triangular matrix """
        self.cholesky_mat = np.linalg.cholesky(self.rf_corr_mat).T
    
    
    def calculate_shifted_levels(self):
        """ run the Monte Carlo simulation """
        self.rng = range(1, self.n_mc_scenarios + 1)
        
        # get levels risk factors
        levels = self.rf.iloc[0, :]

        # create a dataframe of shifted risk factor levels
        self.shifted_levels = pd.DataFrame(columns=self.rf.columns, index=self.rng)

        # perform the number of specified iterations
        for i in self.rng:
            # random numbers from standard normal distribution for each time step and risk factor
            M = np.random.normal(size=(self.VaR_horizon, len(levels)))

            # convert independent random sample to correlated random sample
            M = np.apply_along_axis(lambda gs: np.dot(gs, self.cholesky_mat), 1, M)

            # calculate the total log return over the n-day period
            n_day_rets = M.sum(axis=0)

            # simulate future price levels (shifted levels)
            exponent = -0.5 * self.rf_hist_vols ** 2 + n_day_rets * self.rf_hist_vols
            self.shifted_levels.loc[i, :] = levels * np.exp(exponent)

        self.log(self.shifted_levels, 'Shifted Levels', True, index=False)
    
    
    def get_full_revaluation_pnl(self):
        """ full revaluation P&L """
        full_revaluation = pricing.FullValuation(
            self.positions_df, self.shifted_levels, self.rng, self.amer_opt_params.copy()
        )
        self.full_revaluation_pnl = full_revaluation.full_valuation

        self.log(self.full_revaluation_pnl, 'Full Revaluation P&L', True, index=False)
    
    
    def get_ptf_pnl(self):
        """ dollar and percentage portfolio P&L """
        # create a dataframe for the portfolio P&L
        self.ptf_pnl = pd.DataFrame(columns=['P&L ($)', 'P&L (%)'], index=self.rng)

        # sum over the individual asset P&L's to get the portfolio P&L's
        self.ptf_pnl.loc[:, 'P&L ($)'] = self.full_revaluation_pnl.sum(axis=1)

        # compute the P&L's as percentages
        self.ptf_pnl.loc[:, 'P&L (%)'] = self.ptf_pnl['P&L ($)'] / self.ptf_market_value

        self.log(self.ptf_pnl, 'Portfolio Revaluation P&L', True, index=False)
    
    
    def calculate_var(self):
        """ correlation and value at risk """
        # create a dataframe for the MC VaR
        self.var = pd.DataFrame(columns=self.position_names, index=['VaR', 'Correlation'])

        # iterate over the positions
        ptf_pnl = self.ptf_pnl['P&L ($)']
        for asset in self.position_names:
            asset_pnl = self.full_revaluation_pnl[asset]
            
            # calculate the individual VaR
            self.var.loc['VaR', asset] = np.percentile(asset_pnl, (1 - self.confidence) * 100)

            # compute the correlation between the asset P&L and the portfolio P&L
            self.var.loc['Correlation', asset] = pearsonr(asset_pnl, ptf_pnl)[0]

        # compute the MC n-day portfolio VaR
        self.var.loc['VaR', 'Portfolio'] = np.percentile(ptf_pnl, (1 - self.confidence) * 100)
        
        self.log(self.var, 'Value at Risk')
    
    
    def calculate_sub_ptf_pnl(self):
        # compute sub-portfolio P&Ls
        self.sub_ptf_pnl_excl = pd.DataFrame(columns=self.position_names, index=self.rng)

        # iterate over each position in the portfolio
        for asset in self.position_names:
            asset_pnl = self.full_revaluation_pnl[asset]
            # compute the sub-portfolio P&Ls of the asset
            self.sub_ptf_pnl_excl[asset] = self.ptf_pnl['P&L ($)'] - asset_pnl

        self.log(self.sub_ptf_pnl_excl, 'Sub-Portfolio P&Ls', True, index=False)
    
    
    def calculate_sub_ptf_var(self):
        # compute sub-portfolio VaR
        self.sub_ptf_var = pd.DataFrame(columns=self.position_names, index=['VaR'])

        # iterate over each position in the portfolio
        for asset in self.position_names:
            sub_ptf_pnl = self.sub_ptf_pnl_excl[asset]
            # compute the sub-portfolio VaR of the asset
            self.sub_ptf_var[asset] = np.percentile(sub_ptf_pnl, (1 - self.confidence) * 100)

        self.log(self.sub_ptf_var, 'Sub-Portfolio VaR')
    
    
    def calculate_component_var(self):
        """ component VaR """
        # create a dataframe for component VaR
        self.component_var = pd.DataFrame(columns=self.position_names)

        # compute the component VaR measures for the assets
        var, corr = self.var.loc['VaR', :], self.var.loc['Correlation', :]
        self.component_var.loc[''] = var * corr

        # compute the component VaR measure for the portoflio
        self.component_var['Sum'] = self.component_var.sum(axis=1)

        self.log(self.component_var, 'Component VaR', index=False)
    
    
    def calculate_incremental_var(self):
        """ incremental VaR """
        # create a dataframe for incremental VaR
        self.incremental_var = pd.DataFrame(columns=self.position_names)

        # compute the incremental VaR measures for the assets
        ptf_var = self.var.loc['VaR', 'Portfolio']
        self.incremental_var.loc['', :] = ptf_var - self.sub_ptf_var.values

        self.log(self.incremental_var, 'Incremental VaR', index=False)
    
    
    def calculate_diversification_metrics(self):
        """ diversification metrics, diversification effect, diversification ratio """
        # create a dataframe for diversification metrics
        cols = ['Undiv. VaR', 'Div. VaR', 'Div. Effect ($)', 'Div. Effect (%)', 'Div. Ratio']
        self.diversification = pd.DataFrame(None, [''], cols)

        # get the diversified VaR
        div_var = self.var.loc['VaR', 'Portfolio']
        self.diversification['Div. VaR'] = div_var

        # compute the undiversified VaR
        undiv_var = self.var.loc['VaR', :].drop('Portfolio').sum()
        self.diversification['Undiv. VaR'] = undiv_var
        
        # compute the diversification effects ($)
        self.diversification['Div. Effect ($)'] = undiv_var - div_var

        # compute the diversification effects (%)
        self.diversification['Div. Effect (%)'] = (undiv_var - div_var) / undiv_var

        # compute the diversification ratio
        self.diversification['Div. Ratio'] = div_var / undiv_var

        self.log(self.diversification, 'Diversification Measures', index=False)
    
    
    def plot_hist(self):
        """ P&L histogram """
        title = 'MC P&L (&) Histogram'
        mc_n_day_ptf_var = self.var.loc['VaR', 'Portfolio']
        plt_lgnd = '{}% {}-Day VaR: {} $'.format(
            self.confidence * 100, self.VaR_horizon, round(mc_n_day_ptf_var, 2)
        )
        plotting.plot_pnl_hist(
            self.ptf_pnl['P&L ($)'], title, mc_n_day_ptf_var, plt_lgnd, self.report
        )
    
    
    def waterfall_plots(self):
        """ waterfall plots """
        # plot diversification effects
        pvar_div_effs = self.diversification.drop(columns=['Div. Ratio'])
        n_day_label = '{}-Day'.format(self.VaR_horizon)
        ptf_market_value = self.positions_df.loc['position', '', 'market_value'].sum()
        asset_metrics = self.var.rename(lambda x: 'Individual VaR' if x == 'VaR' else x)
        asset_metrics = asset_metrics.drop(columns=['Portfolio'])
        plotting.div_effect_plots(
            pvar_div_effs, asset_metrics, ptf_market_value,
            self.position_names, n_day_label, self.report
        )
        
        # plot VaR decomposition
        plotting.var_decomp_plots(
            self.component_var, ptf_market_value, self.position_names,
            n_day_label, self.report
        )


class HistoricalSimulationValueAtRisk():
    """ Historical Simulation Value at Risk """
    def __init__(self, report, VaR_horizon, confidence,
                 position_names, rf, rf_rets, hs_period, n_bootstrap,
                 positions_df, ptf_market_value, amer_opt_params, corr_period):
        self.report = report
        self.VaR_horizon = VaR_horizon
        self.confidence = confidence
        self.position_names = position_names
        self.rf = rf
        self.rf_rets = rf_rets
        self.hs_period = hs_period
        self.n_bootstrap = n_bootstrap
        self.positions_df = positions_df
        self.ptf_market_value = ptf_market_value
        self.amer_opt_params = amer_opt_params
        self.corr_period = corr_period
    
    
    def run(self):
        """ calculate value at risk metrics """
        self.report.chapter('Historical Simulation Value-at-Risk')
        self.calculate_shifted_levels()
        self.get_full_revaluation_pnl()
        self.get_ptf_pnl()
        self.bootstrap_pnls()
        self.calculate_var()
        self.calculate_sub_ptf_pnl()
        self.calculate_sub_ptf_var()
        self.calculate_component_var()
        self.calculate_incremental_var()
        self.calculate_diversification_metrics()
        self.plot_hist()
        self.waterfall_plots()
    
    
    def log(self, df, title, head=False, index=True):
        """ print dataframe and save to report """
        print('\n\n{}:\n'.format(title))
        print(df.head().to_string() if head else df.to_string())
        self.report.write_table(df, title, head=head, index=index)
    
    
    def calculate_shifted_levels(self):
        """ get the shifted level of risk factors (historical scenarios) """
        self.rng = range(self.hs_period)

        # create a dataframe
        self.shifted_levels = pd.DataFrame(columns=self.rf.columns, index=self.rng)

        # get the current risk factor levels
        level = self.rf.iloc[0, :]

        # iterate over each risk factor
        for r_f in self.rf.columns:
            # compute the shifted level
            exponent = self.rf_rets[r_f].values[:self.hs_period]
            self.shifted_levels.loc[:, r_f] = level[r_f] * np.exp(exponent)

        self.log(self.shifted_levels, 'Shifted Levels', True, index=False)
    
    
    def get_full_revaluation_pnl(self):
        """ full revaluation P&L """
        full_revaluation = pricing.FullValuation(
            self.positions_df, self.shifted_levels, self.rng, self.amer_opt_params.copy()
        )
        self.full_revaluation_pnl = full_revaluation.full_valuation

        self.log(self.full_revaluation_pnl, 'Full Revaluation P&L', True, index=False)
    
    
    def get_ptf_pnl(self):
        """ historical simulation P&L's """
        ptf_pnl = self.full_revaluation_pnl.sum(axis=1)
        self.ptf_pnl = pd.DataFrame(ptf_pnl, self.rng, ['P&L'])

        self.log(self.ptf_pnl, 'Portfolio Revaluation P&L', True, index=False)
    
    
    def bootstrap_pnls(self):
        """ bootstrapping pnls to calculate n-day pnls """
        # set seed for reproducibility
        np.random.seed(42)

        # create a dataframe to store simulated PnL's
        cols, idx = self.position_names + ['Portfolio'], range(self.n_bootstrap)
        self.bootstrapped_pnl = pd.DataFrame(None, idx, cols)

        # perform n iterations for bootstrapping
        rng, size = np.arange(self.hs_period - self.VaR_horizon), self.VaR_horizon
        for i in range(self.n_bootstrap):
            # select a random time window of size VaR_horizon
            start = np.random.choice(rng, 1)[0]
            idx = list(range(start, start + self.VaR_horizon))

            # calculate the individual n-day P&L's
            pnls = self.full_revaluation_pnl.loc[idx, :]
            self.bootstrapped_pnl.loc[i, self.position_names] = pnls.sum()

            # calculate the portfolio n-day P&L's
            self.bootstrapped_pnl.loc[i, 'Portfolio'] = self.ptf_pnl.iloc[idx, :].sum().values[0]

        self.log(self.bootstrapped_pnl, 'Bootstrapped P&L', True, index=False)
    
    
    def calculate_var(self):
        """ calculate value at risk and correlation """
        # create a dataframe
        self.var = pd.DataFrame(None, ['VaR', 'Correlation'], self.position_names)
        
        # percentile
        q = (1 - self.confidence) * 100

        # iterate over each asset in the portfolio
        ptf_pnl = self.ptf_pnl.loc[:self.corr_period, 'P&L']
        for asset in self.position_names:
            # compute the asset VaR
            asset_pnl = self.bootstrapped_pnl.loc[:, asset]
            self.var.loc['VaR', asset] = np.percentile(asset_pnl, q)
            
            # compute the correlation between the asset and portfolio P&L's
            asset_pnl = self.full_revaluation_pnl.loc[:self.corr_period, asset]
            self.var.loc['Correlation', asset] = pearsonr(asset_pnl, ptf_pnl)[0]

        # compute the portfolio VaR
        ptf_pnl = self.bootstrapped_pnl.loc[:, 'Portfolio']
        self.var.loc['VaR', 'Portfolio'] = np.percentile(ptf_pnl, q)

        self.log(self.var, 'VaR Measures')
    
    
    def calculate_sub_ptf_pnl(self):
        """ calculate sub-portfolio P&Ls """
        # create a dataframe to store the sub-portfolio PnL's
        self.sub_ptf_pnl = pd.DataFrame(None, range(self.n_bootstrap), self.position_names)

        # iterate over each position in the portfolio
        ptf_pnl = self.bootstrapped_pnl['Portfolio']
        for asset in self.position_names:
            # compute the portfolio P&L excluding the asset
            self.sub_ptf_pnl[asset] = ptf_pnl - self.bootstrapped_pnl[asset]

        self.log(self.sub_ptf_pnl, 'Sub-Portfolio P&L', True, index=False)
    
    
    def calculate_sub_ptf_var(self):
        """ calculate sub-portfolio VaR """
        # create a dataframe
        self.sub_ptf_var = pd.DataFrame(None, ['VaR'], self.position_names)

        # iterate over each position in the portfolio
        for asset in self.position_names:
            # compute the portfolio VaR
            sub_ptf_pnl, q = self.sub_ptf_pnl[asset], (1 - self.confidence) * 100
            self.sub_ptf_var[asset] = np.percentile(sub_ptf_pnl, q)

        self.log(self.sub_ptf_var, 'Sub-Portfolio VaR')
    
    
    def calculate_component_var(self):
        """ component VaR """
        # create a dataframe
        self.component_var = pd.DataFrame(columns=self.position_names)

        # compute the asset component VaR's
        var, corr = self.var.loc['VaR', :], self.var.loc['Correlation', :]
        self.component_var.loc[''] = var * corr

        # compute the sum of asset component VaR's
        self.component_var['Sum'] = self.component_var.sum(axis=1)

        self.log(self.component_var, 'Component VaR', index=False)
    
    
    def calculate_incremental_var(self):
        """ incremental VaR """
        # create a dataframe
        self.incremental_var = pd.DataFrame(columns=self.position_names)

        # compute the incremental VaR for each asset
        ptf_var = self.var.loc['VaR', 'Portfolio']
        self.incremental_var.loc['', :] = ptf_var - self.sub_ptf_var.values

        self.log(self.incremental_var, 'Incremental VaR', index=False)
    
    
    def calculate_diversification_metrics(self):
        """ diversification metrics, diversification effect, diversification ratio """
        # create a dataframe
        cols = ['Undiv. VaR', 'Div. VaR', 'Div. Effect ($)', 'Div. Effect (%)', 'Div. Ratio']
        self.diversification = pd.DataFrame(None, [''], cols)

        # get the diversified VaR
        div_var = self.var.loc['VaR', 'Portfolio']
        self.diversification['Div. VaR'] = div_var

        # compute the undiversified VaR
        undiv_var = self.var.loc['VaR', :].drop('Portfolio').sum()
        self.diversification['Undiv. VaR'] = undiv_var
        
        # compute the diversification effects ($)
        self.diversification['Div. Effect ($)'] = undiv_var - div_var

        # compute the diversification effects (%)
        self.diversification['Div. Effect (%)'] = (undiv_var - div_var) / undiv_var

        # compute the diversification ratio
        self.diversification['Div. Ratio'] = div_var / undiv_var

        self.log(self.diversification, 'Diversification Measures', index=False)
    
    
    def plot_hist(self):
        """ Bootstrapped P&L histogram """
        title = 'HS Bootstrapped P&L (&) Histogram'
        hs_n_day_ptf_var = self.var.loc['VaR', 'Portfolio']
        plt_lgnd = '{}% {}-Day VaR: {} $'.format(
            self.confidence * 100, self.VaR_horizon, round(hs_n_day_ptf_var, 2)
        )
        plotting.plot_pnl_hist(
            self.bootstrapped_pnl.Portfolio, title, hs_n_day_ptf_var, plt_lgnd, self.report
        )
    
    
    def waterfall_plots(self):
        """ waterfall plots """
        # plot diversification effects
        pvar_div_effs = self.diversification.drop(columns=['Div. Ratio'])
        n_day_label = '{}-Day'.format(self.VaR_horizon)
        ptf_market_value = self.positions_df.loc['position', '', 'market_value'].sum()
        asset_metrics = self.var.rename(lambda x: 'Individual VaR' if x == 'VaR' else x)
        asset_metrics = asset_metrics.drop(columns=['Portfolio'])
        plotting.div_effect_plots(
            pvar_div_effs, asset_metrics, ptf_market_value,
            self.position_names, n_day_label, self.report
        )
        
        # plot VaR decomposition
        plotting.var_decomp_plots(
            self.component_var, ptf_market_value, self.position_names,
            n_day_label, self.report
        )


class OptionsCalculations():
    """ additional calculations for all options in the portfolio """
    def __init__(self, positions_df, rf_volatilities, rf_info,
                 rf_corr_mat, asset_covariance_matrix_df, report):
        self.positions_df = positions_df
        self.rf_volatilities = rf_volatilities
        self.rf_info = rf_info
        self.rf_corr_mat = rf_corr_mat
        self.asset_covariance_matrix_df = asset_covariance_matrix_df
        self.report = report
    
    
    def run(self):
        """ options calculations """
        # run additional options calculations only if there are any options in the portfolio
        if 'Option' in self.positions_df.loc['position', '', 'product_type'].tolist():
            self.report.chapter('Options Calculations')
            self.get_option_positions()
            self.get_option_vols_decomp_no_corr()
            self.get_option_vols_rfs_and_corr_mats()
            self.options_volatilities_calculations()
            self.option_vols_decomp_corr()
            self.equality_checks()
        else:
            print('Not running, since there are no options in the portfolio.')
    
    
    def log(self, df, title, head=False):
        """ print dataframe and save to report """
        print('\n\n{}:\n'.format(title))
        print(df.head().to_string() if head else df.to_string())
        self.report.write_table(df, title, head=head)
    
    def get_option_positions(self):
        """ get a list of all option positions """
        self.option_positions = []
        for i, pos in self.positions_df.iteritems():
            if pos.loc['position', '', 'product_type'] == 'Option':
                self.option_positions.append((i, pos.loc['position', '', 'asset']))
    
    
    def get_option_vols_decomp_no_corr(self):
        """ option volatilities decomposition, assuming no correlation of risk factors """
    
        # a dataframe for the volatility decompositions with no correlations
        cols = ['delta', 'gamma', 'delta-gamma', 'vega', 'del-gam-veg', 'rho', 'del-gam-veg-rho']
        idx = [pos[1] for pos in self.option_positions]
        self.option_vol_decomp_no_corr = pd.DataFrame(None, idx, cols)
        
        # iterate over the option positions
        for i, asset in self.option_positions:
            # get the necessary tickers
            asset_price_ticker = self.positions_df[i].loc['position', '', 'price_ticker']
            vol_ticker = self.positions_df[i].loc['position', '', 'vol_ticker']
            rate_ticker = self.positions_df[i].loc['position', '', 'rate_ticker']
            
            # get the unit volatility, delta and gamma
            unit_vol = self.rf_info.loc['Volatility (units)', asset_price_ticker]
            delta = self.positions_df[i].loc['price_sensitivities', 'pos', 'delta']
            gamma = self.positions_df[i].loc['price_sensitivities', 'pos', 'gamma']
        
            # compute greeks
            self.option_vol_decomp_no_corr.loc[asset, 'delta'] = abs(delta * unit_vol)
            self.option_vol_decomp_no_corr.loc[asset, 'gamma'] = abs(0.5 * gamma * unit_vol ** 2)
            unit_delta = self.option_vol_decomp_no_corr.loc[asset, 'delta']
            unit_gamma = self.option_vol_decomp_no_corr.loc[asset, 'gamma']
            del_gam = unit_delta + unit_gamma
            vega = abs(self.rf_volatilities.loc[asset, vol_ticker])
            del_gam_veg = del_gam + vega
            rho = abs(self.rf_volatilities.loc[asset, rate_ticker])
            
            # add greeks to dataframe
            self.option_vol_decomp_no_corr.loc[asset, 'delta-gamma'] = del_gam
            self.option_vol_decomp_no_corr.loc[asset, 'vega'] = vega
            self.option_vol_decomp_no_corr.loc[asset, 'del-gam-veg'] = del_gam_veg
            self.option_vol_decomp_no_corr.loc[asset, 'rho'] = rho
            self.option_vol_decomp_no_corr.loc[asset, 'del-gam-veg-rho'] = del_gam_veg + rho
        
        self.log(self.option_vol_decomp_no_corr, 'Option Greeks (No Correlation)')
    
    
    def get_option_vols_rfs_and_corr_mats(self):
        """ options risk factors: correlation matrices """
    
        # lists for the options risk factors and correlation matrices
        self.options_vols_rfs = []
        self.options_vols_corr_mats = []
        
        # iterate over the option positions
        for i, asset in self.option_positions:
            # get the option risk factor tickers
            option_tickers = [
                self.positions_df[i].loc['position', '', 'price_ticker'],
                self.positions_df[i].loc['position', '', 'rate_ticker'],
                self.positions_df[i].loc['position', '', 'vol_ticker']
            ]
            self.options_vols_rfs.append(option_tickers)
        
            # get the correlation matrix of the option risk factors
            option_vols_corr_mat = self.rf_corr_mat.loc[option_tickers, option_tickers]
            self.options_vols_corr_mats.append(option_vols_corr_mat)
        
            self.log(option_vols_corr_mat, 'RF Correlation Matrix for {}'.format(asset))
    
    
    def get_option_rf_cov_mats(self):
        """ options risk factors: covariance matrices"""
        
        # a list for the option risk factors covariance matrices
        self.option_rf_cov_mats = []
        
        # iterate over the option positions
        for i, (_, asset) in enumerate(self.option_positions):
            rfs, vols = self.options_vols_rfs[i], self.options_vols_calcs[i]
            corr_mat = self.options_vols_corr_mats[i]
            
            # create a dataframe for the option risk factors covariance matrix
            option_rf_cov_mat = pd.DataFrame(None, rfs, rfs)
        
            # calculate the covariance for each risk factor pair
            for rf1 in rfs:
                for rf2 in rfs:
                    vol_prod = vols.loc[rf1, 'Indiv Vol'] * vols.loc[rf2, 'Indiv Vol']
                    option_rf_cov_mat.loc[rf1, rf2] = vol_prod * corr_mat.loc[rf1, rf2]
            
            # append the covariance matrix to the list of covariance matrices
            self.option_rf_cov_mats.append(option_rf_cov_mat)
        
            self.log(option_rf_cov_mat, 'RF Covariance Matrix for {}'.format(asset))
    
    
    def get_options_variances_sds(self):
        """ options calculations: variances and standard deviations """
    
        # initialize lists for options variances and standard deviations
        self.options_calc_variances, self.options_calc_sds = [], []
        
        # iterate over the option positions
        for i, (_, asset) in enumerate(self.option_positions):
            # get the individual volatilities for the risk factors
            vols = self.options_vols_calcs[i].loc[self.options_vols_rfs[i], 'Indiv Vol']
        
            # calculate the variance
            option_calc_variance = np.dot(vols.T, np.dot(self.options_vols_corr_mats[i], vols))
            
            # append the variance and standarad deviation to the lists
            self.options_calc_variances.append(option_calc_variance)
            self.options_calc_sds.append(option_calc_variance ** 0.5)
        
        # put the results in a dataframe
        cols = [pos[1] for pos in self.option_positions]
        self.options_variances_sds = pd.DataFrame(None, ['Variance', 'SD'], cols)
        
        # add the variances and standard deviations to the dataframe
        self.options_variances_sds.loc['Variance', :] = self.options_calc_variances
        self.options_variances_sds.loc['SD', :] = self.options_calc_sds
        
        self.log(self.options_variances_sds, 'Option Variances and Standard Deviations')
    
    
    def options_volatilities_calculations(self):
        """ options volatility calculations """
        self.options_vols_calcs = []
        
        # iterate over the option positions
        for i, (_, asset) in enumerate(self.option_positions):
            idx = self.options_vols_rfs[i] + ['Total']
            cols = ['Price', 'Vol (%)', 'Greek', 'Indiv Vol', 'Corr with opt.', 'Comp. Vol']
            option_vols_calcs = pd.DataFrame(None, idx, cols)
            
            # iterate over the risk factors
            for r_f in self.options_vols_rfs[i]:
                option_vols_calcs.loc[r_f, 'Price'] = self.rf_info.loc['Current Level', r_f]
                option_vols_calcs.loc[r_f, 'Vol (%)'] = self.rf_info.loc['Volatility (%)', r_f]
            option_vols_calcs.loc[:, 'Greek'] = ['Delta+Gamma', 'Rho', 'Vega', 'Del-Gam-Veg-Rho']
            
            # add the individual volatilities
            indiv_vol = self.rf_volatilities.loc[asset, self.options_vols_rfs[i]].abs()
            option_vols_calcs.loc[self.options_vols_rfs[i], 'Indiv Vol'] = indiv_vol
            
            # compute the total of the individual volatilities
            total_indiv_vol = option_vols_calcs.loc[self.options_vols_rfs[i], 'Indiv Vol'].sum()
            option_vols_calcs.loc['Total', 'Indiv Vol'] = total_indiv_vol
                
            self.options_vols_calcs.append(option_vols_calcs)
            
        self.get_option_rf_cov_mats()
        self.get_options_variances_sds()
    
        # iterate over the option positions
        for i, (_, asset) in enumerate(self.option_positions):
            # iterate over the risk factors
            for risk_factor in self.options_vols_rfs[i]:
                # calculate the correlation of the risk factor with the option
                rf_cov_sum = self.option_rf_cov_mats[i].loc[risk_factor, :].sum()
                indiv_vol = self.options_vols_calcs[i].loc[risk_factor, 'Indiv Vol']
                sd = self.options_variances_sds.loc['SD', :][i]
                corr = rf_cov_sum / indiv_vol / sd
                self.options_vols_calcs[i].loc[risk_factor, 'Corr with opt.'] = corr
            
            # set the sum of correlations to 1
            self.options_vols_calcs[i].loc['Total', 'Corr with opt.'] = 1
        
            # calculate the component volatility
            vol = self.options_vols_calcs[i].loc[self.options_vols_rfs[i], 'Indiv Vol']
            corr = self.options_vols_calcs[i].loc[self.options_vols_rfs[i], 'Corr with opt.']
            self.options_vols_calcs[i].loc[self.options_vols_rfs[i], 'Comp. Vol'] = vol * corr
            
            # calculate the total component volatility
            comp_vols = self.options_vols_calcs[i].loc[self.options_vols_rfs[i], 'Comp. Vol']
            self.options_vols_calcs[i].loc['Total', 'Comp. Vol'] = comp_vols.sum()
        
            self.log(self.options_vols_calcs[i], '{} Risk Factors Overview'.format(asset))
    
    
    def option_vols_decomp_corr(self):
        """ options volatilities decompositions - correlated risk factors """
    
        # initialize a list for the options volatilities decompositions
        self.options_vol_decomp_corr = []
        
        # iterate over the option positions
        for i, (ptf_idx, asset) in enumerate(self.option_positions):
            # create a dataframe
            cols = ['delta', 'gamma', 'delta-gamma', 'vega', 'del-gam-veg', 'rho', 'del-gam-veg-rho']
            option_vol_decomp_corr = pd.DataFrame(None, [asset, '%'], cols)
            
            # get the risk factor tickers
            price_ticker = self.positions_df[ptf_idx].loc['position', '', 'price_ticker']
            rate_ticker = self.positions_df[ptf_idx].loc['position', '', 'rate_ticker']
            vol_ticker = self.positions_df[ptf_idx].loc['position', '', 'vol_ticker']
            
            # calculate the greeks
            del_gam = self.options_vols_calcs[i].loc[price_ticker, 'Comp. Vol']
            vega = self.options_vols_calcs[i].loc[vol_ticker, 'Comp. Vol']
            rho = self.options_vols_calcs[i].loc[rate_ticker, 'Comp. Vol']
            var = self.asset_covariance_matrix_df.loc[asset, asset]
            
            # add greeks to dataframe
            option_vol_decomp_corr.loc[asset, 'delta-gamma'] = del_gam
            option_vol_decomp_corr.loc[asset, 'vega'] = vega
            option_vol_decomp_corr.loc[asset, 'del-gam-veg'] = del_gam + vega
            option_vol_decomp_corr.loc[asset, 'rho'] = rho
            option_vol_decomp_corr.loc[asset, 'del-gam-veg-rho'] = var ** 0.5
        
            # calculate the greeks in percentages
            for greek in ['delta-gamma', 'vega', 'del-gam-veg', 'rho']:
                asset_greek = option_vol_decomp_corr.loc[asset, greek]
                del_gam_veg_rho = option_vol_decomp_corr.loc[asset, 'del-gam-veg-rho']
                option_vol_decomp_corr.loc['%', greek] = asset_greek / del_gam_veg_rho
                    
            idx = ['delta-gamma', 'vega', 'rho']
            del_gam_veg_rho_pct = option_vol_decomp_corr.loc['%', idx].sum()
            option_vol_decomp_corr.loc['%', 'del-gam-veg-rho'] = del_gam_veg_rho_pct
            
            # append the resulting dataframe to the list of options volatilitiies decompositions
            self.options_vol_decomp_corr.append(option_vol_decomp_corr)

            label = 'Vol Decomposition - Correlated Risk Factors'.format(asset)
            self.log(option_vol_decomp_corr, label)
    
    
    def equality_checks(self):
        """ perform checks """
        for i, (_, asset) in enumerate(self.option_positions):
            sd = self.options_calc_sds[i]
            del_gam_veg_rho = self.options_vol_decomp_corr[i].loc[asset, 'del-gam-veg-rho']
            np.testing.assert_almost_equal(sd, del_gam_veg_rho)

        
class ModelsSummary():
    """ Comparison of Parametrics, Monte Carlo and Historical Simulation VaR models """
    def __init__(self, pvar_model, mcvar_model, hsvar_model, position_names, report):
        self.position_names = position_names
        self.pvar_model = pvar_model
        self.mcvar_model = mcvar_model
        self.hsvar_model = hsvar_model
        self.report = report
    
    
    def log(self, df, title, head=False):
        """ print dataframe and save to report """
        print('\n\n{}:\n'.format(title))
        print(df.head().to_string() if head else df.to_string())
        self.report.write_table(df, title, head=head)
    
    
    def run(self):
        """ create summary dataframes """
        self.report.chapter('Models Comparison')
        self.component_var_comparison()
        self.incremental_var_comparison()
        self.diversification_measures_comparison()
    
    
    def component_var_comparison(self):
        """ comparison of component VaR's """
        # create a dataframe
        idx = ['PVaR', 'MCVaR', 'HSVaR']
        cols = self.position_names + ['Sum']
        self.component_var = pd.DataFrame(None, idx, cols)

        # add the VaR measures
        self.component_var.loc['PVaR', :] = self.pvar_model.component_var.values
        self.component_var.loc['MCVaR', :] = self.mcvar_model.component_var.values
        self.component_var.loc['HSVaR', :] = self.hsvar_model.component_var.values

        # display the summary
        self.log(self.component_var, 'Component VaR')
    
    
    def incremental_var_comparison(self):
        """ comparison of incremental VaR's """

        # create a dataframe
        idx = ['PVaR', 'MCVaR', 'HSVaR']
        self.incremental_var = pd.DataFrame(None, idx, self.position_names)

        # add the VaR measures
        self.incremental_var.loc['PVaR', :] = self.pvar_model.incremental_var.values
        self.incremental_var.loc['MCVaR', :] = self.mcvar_model.incremental_var.values
        self.incremental_var.loc['HSVaR', :] = self.hsvar_model.incremental_var.values

        # display the summary
        self.log(self.incremental_var, 'Incremental VaR')
    
    
    def diversification_measures_comparison(self):
        """ comparison of diversification effects """

        # create a dataframe
        idx = ['PVaR', 'MCVaR', 'HSVaR']
        cols = ['Undiv. VaR', 'Div. VaR', 'Div. Effect ($)', 'Div. Effect (%)', 'Div. Ratio']
        self.diversification = pd.DataFrame(None, idx, cols)

        # add the VaR measures
        self.diversification.loc['PVaR', :] = self.pvar_model.diversification.values
        self.diversification.loc['MCVaR', :] = self.mcvar_model.diversification.values
        self.diversification.loc['HSVaR', :] = self.hsvar_model.diversification.values

        # display the summary
        self.log(self.diversification, 'Diversification')
