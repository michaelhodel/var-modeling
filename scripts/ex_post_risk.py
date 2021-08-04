import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import jarque_bera, probplot
from sklearn.covariance import ledoit_wolf

from scripts import portfolio, pricing, plotting, helpers


class RiskMetrics():
    """ functions for risk metrics calculations """
    
    @staticmethod
    def annualize_rets(r, periods_per_year, **kwargs):
        """ annualized returns from returns time series """
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        return compounded_growth ** (periods_per_year / n_periods) - 1
    
    
    @staticmethod
    def annualize_vol(r, periods_per_year):
        """ annualized volatility from returns time series """
        return r.std() * (periods_per_year ** 0.5)
    
    
    @staticmethod
    def sharpe_ratio(r, riskfree_rate, periods_per_year):
        """ annualized sharpe ratio from returns time series """
        rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
        excess_ret = r - rf_per_period
        ann_ex_ret = RiskMetrics.annualize_rets(excess_ret, periods_per_year)
        ann_vol = RiskMetrics.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol
    
    
    @staticmethod
    def is_normal(r, level):
        """ Jarque-Beraa normality test, True if normality hypothesis accepted, False otherwise """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(RiskMetrics.is_normal, level=level)
        else:
            statistic, p_value = jarque_bera(r)
            return p_value > level
    
    
    @staticmethod
    def drawdown(r):
        """ wealth index, previous peak and percentage drawdowns from asset returns """
        wealth_index = 1000 * (1 + r).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({
            "Wealth": wealth_index, "Previous Peak": previous_peaks, "Drawdown": drawdowns
        })
    
    
    @staticmethod
    def semideviation(r):
        """ (negative) semideviation of returns time series """
        if isinstance(r, pd.Series):
            is_negative = r < 0
            return r[is_negative].std(ddof=0)
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(RiskMetrics.semideviation)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    
    @staticmethod
    def var_historic(r, level):
        """ historic VaR at specified level (s.t. "level" percent of returns are below) """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(RiskMetrics.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    
    @staticmethod
    def cvar_historic(r, level):
        """ conditional VaR from returns """
        if isinstance(r, pd.Series):
            is_beyond = r <= -RiskMetrics.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(RiskMetrics.cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    
    @staticmethod
    def var_gaussian(r, level, modified=False):
        """ parametric Gaussian VaR; if modified, Cornish-Fisher modification is applied """
        # compute the Z score assuming it was Gaussian
        z = norm.ppf(level / 100)
        # modify the Z score based on observed skewness and kurtosis
        if modified:
            s, k = RiskMetrics.skewness(r), RiskMetrics.kurtosis(r)
            z = (
                z +
                (z ** 2 - 1) * s / 6 +
                (z ** 3 -3 * z) * (k - 3) / 24 -
                (2 * z ** 3 - 5 * z) * (s ** 2) / 36
            )
        return -(r.mean() + z * r.std(ddof=0))
    
    
    @staticmethod
    def skewness(r):
        """ skewness """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r ** 3).mean()
        return exp / sigma_r ** 3
    
    
    @staticmethod
    def kurtosis(r):
        """ kurtosis """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r ** 4).mean()
        return exp / sigma_r ** 4
    
    
    @staticmethod
    def portfolio_return(weights, returns):
        """ computes the return on a portfolio from constituent returns and weights """
        return weights.T @ returns
    
    
    @staticmethod
    def portfolio_vol(weights, covmat):
        """ computes the vol of a portfolio from a covariance matrix and constituent weights """
        vol = (weights.T @ covmat @ weights) ** 0.5
        return vol 
    
    
    @staticmethod
    def sample_cov(r, **kwargs):
        """ sample covariance of the supplied returns """
        return r.cov()
    
    
    @staticmethod
    def cc_cov(r, **kwargs):
        """ covariance matrix via the Elton/Gruber Constant Correlation model """
        rhos = r.corr()
        n = rhos.shape[0]
        rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
        ccor = np.full_like(rhos, rho_bar)
        np.fill_diagonal(ccor, 1.)
        sd = r.std()
        return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)
    
    
    @staticmethod
    def shrinkage_cov(r, delta, **kwargs):
        """ shrinkage covariance estimator, shrinks between sample and CC estimators """
        prior = RiskMetrics.cc_cov(r, **kwargs)
        sample = RiskMetrics.sample_cov(r, **kwargs)
        return delta * prior + (1 - delta) * sample
    
    
    @staticmethod
    def ledoit_wolf_cov(r):
        """ ledoit-wolf covariance estimation """
        shrunk_cov, shrinkage = ledoit_wolf(r)
        return pd.DataFrame(shrunk_cov, r.columns, r.columns)


class WeightingSchemes():
    """ functions for weight calculation for different strategies """
    
    @staticmethod
    def minimize_vol(target_return, er, cov):
        """ optimal weights achieving target return given set of expected returns, covariances """
        n = er.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        return_is_target = {
            'type': 'eq', 'args': (er,),
            'fun': lambda weights, er: target_return - RiskMetrics.portfolio_return(weights,er)
        }
        
        # minimize volatility
        weights = minimize(
            RiskMetrics.portfolio_vol, init_guess, args=(cov,),
            method='SLSQP', options={'disp': False},
            constraints=(weights_sum_to_1, return_is_target), bounds=bounds
        )
        return weights.x
    
    
    @staticmethod
    def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
        """ weights of EW portfolio based on asset returns """
        n = len(r.columns)
        ew = pd.Series(1 / n, index=r.columns)
        if cap_weights is not None:
            # starting cap weight
            cw = cap_weights.loc[r.index[0]]
            # exclude microcaps
            if microcap_threshold is not None and microcap_threshold > 0:
                microcap = cw < microcap_threshold
                ew[microcap] = 0
                ew = ew / ew.sum()
            # limit weight to a multiple of capweight
            if max_cw_mult is not None and max_cw_mult > 0:
                ew = np.minimum(ew, cw * max_cw_mult)
                # reweight
                ew = ew / ew.sum() 
        return pd.Series(ew, index=r.columns)
    
    
    @staticmethod
    def weight_cw(r, cap_weights, window, **kwargs):
        """ weights of the CW portfolio based on the time series of capweights """
        w = cap_weights.loc[r.index[window - 1]]
        return pd.Series(w / w.sum(), index=r.columns)
    
    
    @staticmethod
    def risk_contribution(w, cov):
        """ relative contributions to risk of assets, given weights, covariances """
        total_portfolio_var = RiskMetrics.portfolio_vol(w, cov) ** 2
        # Marginal contribution of each constituent to portfolio variance
        marginal_contrib = cov @ w
        # Relative contribution of each constituent to portfolio variance (risk)
        risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
        return risk_contrib
    
    
    @staticmethod
    def msd_risk(weights, target_risk, cov):
        """ minimise MSE of risk contributions and target risk contributions via asset weights """
        w_contribs = WeightingSchemes.risk_contribution(weights, cov)
        return ((w_contribs - target_risk) ** 2).sum()
    
    
    @staticmethod
    def target_risk_contributions(target_risk, cov):
        """ weights such that risk contributions are equal to target risk contributions """
        n = cov.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n
        
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        # minimize
        weights = minimize(
            WeightingSchemes.msd_risk, init_guess, args=(target_risk, cov),
            method='SLSQP', options={'disp': False},
            constraints=(weights_sum_to_1,), bounds=bounds
        )
        return pd.Series(weights.x, index=cov.columns)

    @staticmethod
    def equal_risk_contributions(cov):
        """ weights of portfolio that equalizes the risk contributions based on covariances """
        n = cov.shape[0]
        return WeightingSchemes.target_risk_contributions(target_risk=np.repeat(1 / n, n), cov=cov)
    
    
    @staticmethod
    def weight_erc(r, cov_estimator=RiskMetrics.sample_cov, **kwargs):
        """ weights of ERC portfolio given returns series and covariance matrix """
        est_cov = cov_estimator(r, **kwargs)
        return WeightingSchemes.equal_risk_contributions(est_cov)
    
    
    @staticmethod
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """ objective function to minimize: negative sharpe ratio """
        r = RiskMetrics.portfolio_return(weights, er)
        vol = RiskMetrics.portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol
    
    
    @staticmethod
    def msr(riskfree_rate, er, cov):
        """ weights of portfolio with maximum sharpe ratio given riskfree rate, E(r), cov.-mat. """
        # inputs and constraints
        n = er.shape[0]
        # equal weighting for init_guess
        init_guess = np.repeat(1 / n, n)
        # minimum and maximum individual allocation
        bounds = ((0.0, 1.0), ) * n
        # equality constraint: sum of portfolio weights variable minus one must equal zero
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        # minimize
        weights = minimize(
            WeightingSchemes.neg_sharpe, init_guess, args=(riskfree_rate, er, cov),
            method='SLSQP', options={'disp': False},
            constraints=(weights_sum_to_1,), bounds=bounds
        )
        return pd.Series(weights.x, index=cov.columns)
    
    
    @staticmethod
    def weight_msr(r, cov_estimator=RiskMetrics.sample_cov, **kwargs):
        """ weights of the MSR portfolio given returns and covariance matrix """
        est_cov = cov_estimator(r, **kwargs)
        exp_ret = RiskMetrics.annualize_rets(r, 12, **kwargs)
        return WeightingSchemes.msr(0, exp_ret, est_cov)
    
    
    @staticmethod
    def optimal_weights(n_points, er, cov):
        """ list fo weights representing grid of points on efficient frontier """
        target_rs = np.linspace(er.min(), er.max(), n_points)
        weights = [WeightingSchemes.minimize_vol(target_r, er, cov) for target_r in target_rs]
        return weights
    
    
    @staticmethod
    def gmv(cov):
        """ weights of the Gloal Minimum Volatility portfolio given covariance matrix """
        n = cov.shape[0]
        return pd.Series(WeightingSchemes.msr(0, np.repeat(1, n), cov), index=cov.columns)

    @staticmethod
    def weight_gmv(r, cov_estimator=RiskMetrics.sample_cov, **kwargs):
        """ weights of GMV portfolio given a covariance matrix """
        est_cov = cov_estimator(r, **kwargs)
        return WeightingSchemes.gmv(est_cov)


class PerformanceAnalysis():
    """ evaluation of ex-post performance and rules-based strategies """
    def __init__(self, rf, position_configurations, horizon,
                 estimation_window, confidence, riskfree_rate, jarque_bera_alpha,
                 cov_estim_method, report, shrinkage_coefficient=None):
        self.rf = rf
        self.position_configurations = position_configurations
        self.horizon = horizon
        self.estimation_window = estimation_window
        self.var_level = round((1 - confidence) * 100, 4)
        self.ann_factor = self.get_ann_factor(horizon)
        self.riskfree_rate = riskfree_rate
        self.jarque_bera_alpha = jarque_bera_alpha
        self.cov_estim_method = cov_estim_method
        self.report = report
        self.shrinkage_coefficient = shrinkage_coefficient
        
        
    def log(self, obj, title, head=False, index=True):
        """ print pandas object to console and save it to report """
        print('\n\n{}:\n'.format(title))
        # convert series to dataframe
        df = pd.DataFrame(obj).T if isinstance(obj, pd.Series) else obj
        index = False if isinstance(obj, pd.Series) or not index else True
        # print to stdout and report
        print(df.head().to_string() if head else df.to_string())
        self.report.write_table(df, title, head=head, index=index)
        
        
    def run(self):
        """ perform analysis """
        # run backward-looking analysis only if there are at least two equities
        if self.position_configurations['type'].tolist().count('Equity') >= 2:
            self.report.chapter('Ex-Post Portfolio Performance')
            self.create_portfolio()
            self.valuate()
            self.calculate_returns()
            self.horizon_returns()
            self.metrics()
            self.construct_cap_weighted_benchmark()
            self.construct_equal_weighted_benchmark()
            self.compare_ew_and_cw()
            self.risk_stats_plots()
            self.compute_efficient_frontier()
            self.construct_risk_parity_portfolio()
            self.backtest_returns()
            self.stats_summary()
        else:
            print('Not running, since there are less than two equities in the portfolio.')
    
    
    def get_ann_factor(self, horizon):
        """ specify annualization factor based on chosen granularity (horizon) """
        if isinstance(horizon, int):
            return 252 / horizon
        elif isinstance(horizon, str):
            return {'B': 252, 'W': 52, 'M': 12}[horizon]
    
    
    def create_portfolio(self):
        """ re-create portfolio at beginning of backtesting period """
        # extract equities positions (analysis only for equities)
        equitied_idx = self.position_configurations['type'] == 'Equity'
        equities_configs = self.position_configurations[equitied_idx]
        
        # create a list of tickers for equities
        self.equities = equities_configs.price_ticker.tolist()
        
        # re-create portfolio using levels at first available date
        ptf = portfolio.PortfolioDefinition(
            position_configurations=equities_configs,
            rf_levels=self.rf.iloc[-1:],
            report=self.report,
            amer_opt_params=None
        )
        
        self.positions_df = ptf.construct_portfolio()
    
    
    def valuate(self):
        """ valuate assets and portfolio based on historical levels of risk factors """
        levels, rng = self.rf.copy().reset_index(), range(self.rf.shape[0])
        full_valuation_pnls = pricing.FullValuation(self.positions_df, levels, rng)
        asset_valuations = full_valuation_pnls.full_valuation
        self.asset_valuations = asset_valuations[::-1]
        self.asset_valuations.index = pd.DatetimeIndex(self.rf[::-1].index)
        self.asset_valuations.columns = self.equities
        
        # calculate asset values
        for i, pos in self.positions_df.iteritems():
            asset = pos.loc['position', '', 'price_ticker']
            market_value = self.positions_df[i].loc['position', '', 'market_value']
            self.asset_valuations[asset] += market_value

        self.log(self.asset_valuations, 'Asset Valuations', head=True)
        plotting.plot_time_series(self.asset_valuations, 'Value', 'Value', self.report)
        
        # calculate portoflio value
        ptf_val = self.asset_valuations.sum(axis=1)
        self.ptf_valuation = pd.DataFrame(ptf_val, pd.DatetimeIndex(self.rf.index[::-1]), ['Value'])

        self.log(self.ptf_valuation, 'Portfolio Valuation', head=True)
        plotting.plot_time_series(self.ptf_valuation, 'Portfolio Value', 'Value', self.report)
    
    
    def calculate_returns(self):
        """ calculate assets and portfolio returns from valuations """
        # calculate asset returns
        self.asset_rets = self.asset_valuations.pct_change().dropna()
        self.log(self.asset_rets, 'Asset Returns', head=True)
        plotting.plot_time_series(self.asset_rets, 'Return', 'Return', self.report)
        
        # calculate portoflio returns
        self.ptf_rets = self.ptf_valuation.pct_change().dropna()
        self.ptf_rets.columns = ['Return']
        self.log(self.ptf_rets, 'Portfolio Returns', head=True)
        plotting.plot_time_series(self.ptf_rets, 'Portfolio Returns', 'Return', self.report)
    
    
    def horizon_returns(self):
        """ calculate asset and portfolio returns over horizon-length periods from valuations """
        # n-day asset returns
        if isinstance(self.horizon, int):
            n_th_day_asset_valuations = self.asset_valuations[::self.horizon]
            lab = '{}-day Returns'.format(self.horizon)
        elif isinstance(self.horizon, str):
            n_th_day_asset_valuations = self.asset_valuations.resample(self.horizon).first()
            lab = '{} Returns'.format(
                {'B': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[self.horizon]
            )
        self.n_day_asset_rets = n_th_day_asset_valuations.pct_change().dropna()
        df_to_plot = self.n_day_asset_rets.set_index(
            self.n_day_asset_rets.index.strftime('%Y-%m-%d')
        )
        plotting.bar_time_series(df_to_plot, lab, 'Return', self.report)
        
        # n-day portfolio returns
        if isinstance(self.horizon, int):
            n_th_day_ptf_valuations = self.ptf_valuation[::self.horizon]
            lab = '{}-day Portfolio Returns'.format(self.horizon)
        elif isinstance(self.horizon, str):
            n_th_day_ptf_valuations = self.ptf_valuation.resample(self.horizon).first()
            lab = '{} Portfolio Returns'.format(
                {'B': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[self.horizon]
            )
        self.n_day_ptf_rets = n_th_day_ptf_valuations.pct_change().dropna()
        df_to_plot = self.n_day_ptf_rets.set_index(self.n_day_ptf_rets.index.strftime('%Y-%m-%d'))
        plotting.bar_time_series(df_to_plot, lab, 'Return', self.report)
    
    
    def metrics(self):
        """ expected returns, volatilities, correlation and covariance matrices """
        self.er = RiskMetrics.annualize_rets(self.n_day_asset_rets, self.ann_factor)
        self.ev = RiskMetrics.annualize_vol(self.n_day_asset_rets, self.ann_factor)
        self.corr = self.n_day_asset_rets.corr()
        if self.cov_estim_method == 'sample':
            self.cov = RiskMetrics.sample_cov(self.n_day_asset_rets)
        elif self.cov_estim_method == 'shrinkage':
            self.cov = RiskMetrics.shrinkage_cov(
                self.n_day_asset_rets, delta=self.shrinkage_coefficient
            )
        elif self.cov_estim_method == 'ledoit_wolf':
            self.cov = RiskMetrics.ledoit_wolf_cov(self.n_day_asset_rets)
        self.covmat_ann = self.cov * self.ann_factor

        # display metrics
        metrics = [
            self.er, self.ev, self.corr, self.cov, self.covmat_ann
        ]
        labels = [
            'Expected Annualized Returns', 'Expected Annualized Volatilities',
            'Correlation Matrix', 'Covariance Matrix', 'Annualized Covariance Matrix'
        ]

        for metric, label in zip(metrics, labels):
            self.log(metric, label)
    
    
    def construct_cap_weighted_benchmark(self):
        """ construction of cap-weighted benchmark """
        # get a list of tickers
        tickers = self.asset_valuations.columns.tolist()
        
        # get total shares outstanding
        total_shares = helpers.get_total_shares_outstanding(tickers)
        
        # get number of shares per position
        pos_shares = pd.Series(
            self.positions_df.loc['position', '', 'securities_contracts'].values,
            index=self.positions_df.loc['position', '', 'price_ticker'].values
        )
        
        # calculate daily position market caps
        daily_marketcaps = self.asset_valuations / pos_shares * total_shares
        
        # use n-day data
        if isinstance(self.horizon, int):
            nth_day_marketcaps = daily_marketcaps[::self.horizon][1:]
        elif isinstance(self.horizon, str):
            nth_day_marketcaps = daily_marketcaps.resample(self.horizon).first()[1:]

        self.log(nth_day_marketcaps, 'Market Capitalizations', head=True)

        # Compute and inspect price evolution of benchmark expressed in terms of capitalization
        total_nth_day_marketcap = nth_day_marketcaps.sum(axis=1)
        title = 'Evolution of Market Capitalization of Index'
        total_nth_day_marketcap.plot(figsize=(12, 6));
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # Compute benchmark capweights by dividing mkt caps by total mkt cap
        self.capitalization_weights = nth_day_marketcaps.divide(total_nth_day_marketcap, axis=0)
        self.log(self.capitalization_weights, 'Capitalization Weights', head=True)

        # Compute n-day market return
        total_market_return = (self.capitalization_weights * self.n_day_asset_rets).sum(axis=1)

        # Plot n-day market return
        title = 'Returns of Cap-Weighted Index'
        to_plot = pd.Series(
            total_market_return.values, index=total_market_return.index.strftime('%Y-%m-%d')
        )
        to_plot.plot.bar(figsize=(12, 6));
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # Plot cumulative market return
        self.total_market_index = (1 + total_market_return).cumprod()
        title = 'Cumulative Return Cap-Weighted Index'
        to_plot = pd.Series(
            self.total_market_index.values, index=self.total_market_index.index.strftime('%Y-%m-%d')
        )
        to_plot.plot(figsize=(12, 6));
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # Capitalization Weighted returns, volatility
        self.w_cw = WeightingSchemes.weight_cw(
            self.n_day_asset_rets, cap_weights=self.capitalization_weights,
            window=self.estimation_window
        )
        self.r_cw = RiskMetrics.portfolio_return(self.w_cw, self.er)
        self.vol_cw = RiskMetrics.portfolio_vol(self.w_cw, self.covmat_ann)

        self.log(self.w_cw, 'Weights of Capweight Scheme')
        to_print = ['Portfolio Return for Capweight Scheme: {}'.format(round(self.r_cw, 4)),
                    'Portfolio Volatility for Capweight Scheme: {}'.format(round(self.vol_cw, 4))]
        for line in to_print:
            print(line)
            self.report.write(line)
    
    
    def construct_equal_weighted_benchmark(self):
        """ construction of equal-weighted benchmark"""        
        # equally weighted returns, volatility
        self.w_ew = WeightingSchemes.weight_ew(self.n_day_asset_rets)
        self.r_ew = RiskMetrics.portfolio_return(self.w_ew, self.er)
        self.vol_ew = RiskMetrics.portfolio_vol(self.w_ew, self.covmat_ann)

        scheme = 'Equal Weighted Scheme'
        self.log(self.w_ew, 'Weights of {}'.format(scheme))
        to_print = ['Portfolio Return for {}: {}'.format(scheme, round(self.r_ew, 4)),
                    'Portfolio Volatility for {}: {}'.format(scheme, round(self.vol_ew, 4))]
        for line in to_print:
            print(line)
            self.report.write(line)
        
        # calculate equal weights
        n_ew = self.n_day_asset_rets.shape[1]
        w_ew = np.repeat(1 / n_ew, n_ew)
        ind_equalweight = self.capitalization_weights.multiply(
            1 / self.capitalization_weights / n_ew, axis=0
        )
        self.log(ind_equalweight, 'Weights of Equal-Weighted Benchmark', head=True)

        # calculate n-day return of EW index
        total_eqweighted_return = (ind_equalweight * self.n_day_asset_rets).sum(axis=1)
        title = 'Returns of Equal-Weighted Index'
        to_plot = pd.Series(
            total_eqweighted_return.values, index=total_eqweighted_return.index.strftime('%Y-%m-%d')
        )
        to_plot.plot.bar(figsize=(12, 6));
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # calculate cumulative return of equal-weighted index
        self.total_eqweighted_index = (1 + total_eqweighted_return).cumprod()
        title = 'Cumulative Return Equal-Weighted Index'
        self.total_eqweighted_index.plot(figsize=(12, 6));
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();
    
    
    def compare_ew_and_cw(self):
        """ comparison cumulative returns of equal-weighted and cap-weighted """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        title = 'Equal Cap Weighted Vs. Market Cap Weighted Indices'
        self.total_market_index.plot(label='Mkt-weighted', legend=True);
        self.total_eqweighted_index.plot(label='Eq-weighted', legend=True);
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();
    
    
    def risk_stats_plots(self):
        """ plotting of risk statistics for n-day asset returns """
        rm = RiskMetrics
        rf, af = self.riskfree_rate, self.ann_factor
        lvl, sfx = self.var_level, ' VaR @ {}%'.format(str(100 - self.var_level))
        
        # tuples are of the form (function, arguments, title)
        risk_statistics_plotting = [
            (rm.sharpe_ratio, {'riskfree_rate': rf, 'periods_per_year': af}, 'Sharpe Ratio'),
            (rm.var_gaussian, {'level': lvl, 'modified': False}, 'Parametric (Gaussian)' + sfx),
            (rm.var_gaussian, {'level': lvl, 'modified': True}, 'Modified (Cornish-Fisher)' + sfx),
            (rm.var_historic, {'level': lvl}, 'Historic' + sfx),
            (rm.cvar_historic, {'level': lvl}, 'Conditional' + sfx),
            (rm.semideviation, {}, 'Negative Semi-Deviation'),
            (rm.skewness, {}, 'Skewness'),
            (rm.kurtosis, {}, 'Kurtosis')
        ]

        # plot idiosyncratic risk statistics for n-day asset returns
        for configuration in risk_statistics_plotting:
            metric_series = configuration[0](r=self.n_day_asset_rets, **configuration[1])
            metric_series.sort_values().plot.bar();
            self.report.write_plot(configuration[2])
            plt.title(configuration[2]);
            plt.show();
            plt.close();

        # empirical distribution of n-day returns
        for column in self.n_day_asset_rets:
            self.n_day_asset_rets[column].hist();
            title = 'Empirical {} Returns Distribution'.format(column)
            plt.xlabel('Return')
            plt.ylabel('Count')
            self.report.write_plot(title)
            plt.title(title);
            plt.show();
            plt.close();

        # check for normality of returns
        test_res = RiskMetrics.is_normal(self.n_day_asset_rets, level=self.jarque_bera_alpha)
        self.log(test_res, 'Normality Test Results')

        # Inspect normality of returns
        for column in self.n_day_asset_rets:
            probplot(self.n_day_asset_rets[column], dist='norm', plot=plt);
            self.report.write_plot(column)
            plt.title(column);
            plt.show();
            plt.close();
    
    
    def plot_ef(self, n_points, er, cov, style='.-', legend=False, show_cml=False,
                riskfree_rate=0, show_ew=False, show_gmv=False):
        """ multi-asset efficient frontier using the optimal weights """
        weights = WeightingSchemes.optimal_weights(n_points, er, cov)
        rets = [RiskMetrics.portfolio_return(w, er) for w in weights]
        vols = [RiskMetrics.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
        ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
        plt.xlabel('Volatility')
        plt.ylabel('Returns')
        if show_cml:
            ax.set_xlim(left = 0)
            w_msr = WeightingSchemes.msr(riskfree_rate, er, cov)
            r_msr = RiskMetrics.portfolio_return(w_msr, er)
            vol_msr = RiskMetrics.portfolio_vol(w_msr, cov)
            ax.plot(
                [vol_msr], [r_msr], color='red', marker="*", linestyle='dashed',
                linewidth=2, markersize=18, label='msr'
            )
            plt.annotate("MSR", xy=(vol_msr, r_msr), ha='right', va='bottom', rotation=45)
        if show_ew:
            w_ew = np.repeat(1 / er.shape[0], er.shape[0])
            r_ew = RiskMetrics.portfolio_return(w_ew, er)
            vol_ew = RiskMetrics.portfolio_vol(w_ew, cov)
            ax.plot([vol_ew], [r_ew], color='green', marker='o', markersize=10, label='ew')
            plt.annotate(
                "EW", xy=(vol_ew, r_ew), horizontalalignment='right',
                verticalalignment='bottom', rotation=45
            )
        if show_gmv:
            w_gmv = WeightingSchemes.gmv(cov)
            r_gmv = RiskMetrics.portfolio_return(w_gmv, er)
            vol_gmv = RiskMetrics.portfolio_vol(w_gmv, cov)
            ax.plot([vol_gmv], [r_gmv], color='goldenrod', marker="D", markersize=12, label='gmv')
            plt.annotate(
                "GMV", xy=(vol_gmv, r_gmv), horizontalalignment='right',
                verticalalignment='bottom', rotation=45
            )
        title = 'Ex-Ante Efficient Frontier'
        self.report.write_plot(title)
        ax.set_title(title);
        plt.show();
        plt.close();
    
    
    def compute_efficient_frontier(self):
        """ construct efficient frontier based on classical markowitz model """
        # display efficient frontier
        self.plot_ef(
            100, self.er, self.covmat_ann, style='.-', legend=False, show_cml=True,
            riskfree_rate=self.riskfree_rate, show_ew=True, show_gmv=True
        );

        # Maximum Sharpe Ratio returns, volatility
        scheme = 'Maximum Sharpe Ratio Scheme'
        w_msr = WeightingSchemes.msr(self.riskfree_rate, self.er, self.covmat_ann)
        r_msr = RiskMetrics.portfolio_return(w_msr, self.er)
        vol_msr = RiskMetrics.portfolio_vol(w_msr, self.covmat_ann)

        self.log(w_msr, 'Weights of {}'.format(scheme))
        to_print = ['Portfolio Return for {}: {}'.format(scheme, round(r_msr, 4)),
                    'Portfolio Volatility for {}: {}'.format(scheme, round(vol_msr, 4))]
        for line in to_print:
            print(line)
            self.report.write(line)

        # Global Minimum Variance returns, volatility
        scheme = 'Global Minimum Variance Scheme'
        w_gmv = WeightingSchemes.gmv(self.covmat_ann)
        r_gmv = RiskMetrics.portfolio_return(w_gmv, self.er)
        vol_gmv = RiskMetrics.portfolio_vol(w_gmv, self.covmat_ann)

        self.log(w_gmv, 'Weights of {}'.format(scheme))        
        to_print = ['Portfolio Return for {}: {}'.format(scheme, round(r_gmv, 4)),
                    'Portfolio Volatility for {}: {}'.format(scheme, round(vol_gmv, 4))]
        for line in to_print:
            print(line)
            self.report.write(line)
    
    
    def risk_contributions(self, w, cov, strategy):
        """ relative risk contributions, portfolio weights, volatility, component risk """
        # relative risk contributions of portfolio
        RRC = WeightingSchemes.risk_contribution(w, cov)

        # ensure weights sum to 1
        np.testing.assert_almost_equal(sum(RRC), 1)

        # visualize Relative Risk Contributions
        title = 'Relative (%) Risk Contributions of the {} Portfolio'.format(strategy)
        RRC.plot.bar();
        plt.xlabel('Asset')
        plt.ylabel('Risk Contribution')
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # visualize weights of portfolio
        title = 'Asset Weights - {} Portfolio'.format(strategy)
        w.plot.bar();
        plt.xlabel('Asset')
        plt.ylabel('Weight')
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # portfolio volatility given sample covariance matrix
        ptf_vol = RiskMetrics.portfolio_vol(w, cov)
        print('{} Portfolio volatility: {}'.format(strategy, round(ptf_vol, 4)))
        self.report.write('{} Portfolio volatility: {}'.format(strategy, round(ptf_vol, 4)))

        # risk contributions (Component Risk) of constituents of ew portfolio
        RC = RRC * ptf_vol
        title = 'Component Risk of the {} Portfolio'.format(strategy)
        RC.plot.bar();
        plt.xlabel('Asset')
        plt.ylabel('Component Risk')
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();

        # ensure sum of risk contributions equals portfolio volatility
        np.testing.assert_almost_equal(ptf_vol, RC.sum())
    
    
    def construct_risk_parity_portfolio(self):
        """ construction of risk parity portfolio """
        # Equal Risk Contribution returns, volatility
        scheme = 'Equal Risk Contribution Scheme'
        w_erc = WeightingSchemes.weight_erc(self.n_day_asset_rets)
        r_erc = RiskMetrics.portfolio_return(w_erc, self.er)
        vol_erc = RiskMetrics.portfolio_vol(w_erc, self.covmat_ann)

        self.log(w_erc, 'Weights of {}'.format(scheme))        
        to_print = ['Portfolio Return for {}: {}'.format(scheme, round(r_erc, 4)),
                    'Portfolio Volatility for {}: {}'.format(scheme, round(vol_erc, 4))]
        for line in to_print:
            print(line)
            self.report.write(line)

        # risk contributions, asset weights, portfolio volatility and component risk in EW Portfolio
        self.risk_contributions(self.w_ew, self.cov, 'Equal Weights')

        # risk contributions, asset weights, portfolio volatility and component risk in ERC Portfolio
        self.risk_contributions(w_erc, self.cov, 'Equal Risk Contribution')
    
    
    def backtest_ws(self, r, estimation_window, ws, verbose=False, **kwargs):
        """ backtest weighting scheme ws on asset returns r in estimation_window """
        n_periods = r.shape[0]
        windows = [(s, s + estimation_window) for s in range(n_periods - estimation_window)]
        weights = [ws(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
        weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
        returns = (weights * r).sum(axis=1, min_count=1)
        return returns
    
    
    def backtest_returns(self):
        """ backtest returns applying security weighting schemes"""
        # backtesting inputs
        rm, ws, backtest = RiskMetrics, WeightingSchemes, self.backtest_ws
        bt_config = {'r': self.n_day_asset_rets, 'estimation_window': self.estimation_window}
        cov_bt_config = bt_config.copy()
        if self.cov_estim_method == 'shrinkage':
            cov_bt_config['cov_estimator'] = rm.shrinkage_cov
            cov_bt_config['delta'] = self.shrinkage_coefficient
        elif self.cov_estim_method == 'sample':
            cov_bt_config['cov_estimator'] = rm.sample_cov
        elif self.cov_estim_method == 'ledoit_wolf':
            cov_bt_config['cov_estimator'] = rm.ledoit_wolf_cov
        cw_config = {'cap_weights': self.capitalization_weights, 'window': self.estimation_window}

        # historical weights
        market_values = self.positions_df.loc['position', '', 'market_value'].values
        market_values = pd.Series(market_values, index=self.n_day_asset_rets.columns)
        weight_hs = lambda w: market_values / market_values.sum()
        
        # dataframe for backtesting returns
        backtesting_results = pd.DataFrame({
            'Historical': backtest(**bt_config, ws=weight_hs),
            'EW': backtest(**bt_config, ws=ws.weight_ew),
            'CW': backtest(**bt_config, ws=ws.weight_cw, **cw_config),
            'MSR': backtest(**cov_bt_config, ws=ws.weight_msr),
            'GMV': backtest(**cov_bt_config, ws=ws.weight_gmv),
            'ERC': backtest(**cov_bt_config, ws=ws.weight_erc),
        })

        # keep only returns for estimation window
        self.backtesting_results = backtesting_results[self.estimation_window:]
        self.log(self.backtesting_results, 'Backtesting Returns', head=True)

        # cumulative returns
        cumulative_returns = (1 + self.backtesting_results).cumprod()
        self.log(cumulative_returns, 'Cumulative Returns', head=True)
        
        # plot cumulative returns
        title = 'Strategies Cumulative Return'
        cumulative_returns.plot(figsize=(16, 12))
        self.report.write_plot(title)
        plt.title(title);
        plt.show();
        plt.close();
    
    
    def stats_summary(self):
        """ aggregated summary stats for the returns """
        r = self.backtesting_results
        rf, af, lvl = self.riskfree_rate, self.ann_factor, self.var_level
        rm = RiskMetrics
        lab = ' ({}%)'.format(str(100 - lvl))
        
        # compute annualized statistics
        returns = r.aggregate(rm.annualize_rets, periods_per_year=af)
        volatility = r.aggregate(rm.annualize_vol, periods_per_year=af)
        semideviation = r.aggregate(rm.semideviation) * np.sqrt(af)
        skewness = r.aggregate(rm.skewness)
        kurtosis = r.aggregate(rm.kurtosis)
        gaussian_var = r.aggregate(rm.var_gaussian, level=lvl, modified=False)
        cornish_fisher_var = r.aggregate(rm.var_gaussian, level=lvl, modified=True)
        historic_var = r.aggregate(rm.var_historic, level=lvl)
        historic_cvar = r.aggregate(rm.cvar_historic, level=lvl)
        max_drawdown = r.aggregate(lambda r: rm.drawdown(r).Drawdown.min())
        sharpe_ratio = r.aggregate(rm.sharpe_ratio, riskfree_rate=rf, periods_per_year=af)
        sortino_ratio = returns / semideviation
        
        # create a dataframe
        self.performance_summary = pd.DataFrame({
            'Return': returns,
            'Volatility': volatility,
            'Semi-Deviation': semideviation,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Gaussian VaR' + lab: gaussian_var,
            'Modified VaR' + lab: cornish_fisher_var,
            'Historic VaR' + lab: historic_var,
            'Historic CVaR' + lab: historic_cvar,
            'Max. Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
         })
        
        # display results
        self.log(self.performance_summary.T, 'Performance Summary (Annualized)')
