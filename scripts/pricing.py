from scipy.stats import norm, pearsonr
import scipy.optimize as optimize
import QuantLib as ql
import pandas as pd
import numpy as np
from datetime import timedelta


def get_T(settlement, maturity):
    """ getting the duration of an instrument in years """
    day_delta = pd.to_datetime(maturity) - pd.to_datetime(settlement)
    return day_delta / timedelta(days=365)


class OptionPricing():
    """ implementation of option pricing and greek methods """
    def __init__(self, callput, S, K, settlement, maturity, r, d, v):
        self.callput = callput
        self.S = S
        self.K = K
        self.settlement = settlement
        self.maturity = maturity
        self.r = r
        self.d = d
        self.v = v
    
    
    def calculate_price(self, compute_greeks):
        """ compute option price and greeks """
        self.price = None
        if compute_greeks:
            self.calculate_greeks()
    
    
    def calculate_greeks(self):
        """ class methods as default method to calculate greeks """
        self.greeks = {
            'delta': self.delta(), 'gamma': self.gamma(),
            'theta': self.theta(), 'vega': self.vega(), 'rho': self.rho()
        }
    
    
    def delta(self):
        """ calculate delta """
        price, S = self.price, self.S
        delta_diff = S * 0.0001
        self.S = S + delta_diff
        self.calculate_price(compute_greeks=False)
        delta_up = self.price
        self.S = S - delta_diff
        self.calculate_price(compute_greeks=False)
        delta_down = self.price
        self.price, self.S = price, S
        return (delta_up - delta_down) / float(2.0 * delta_diff)
    
    
    def gamma(self):
        """ calculate gamma """
        price, S = self.price, self.S
        gamma_diff = S * 0.0001
        self.S = S + gamma_diff
        gamma_up = self.delta()
        self.S = S - gamma_diff
        gamma_down = self.delta()
        self.price, self.S = price, S
        return (gamma_up - gamma_down) / float(2.0 * gamma_diff)
    
    
    def vega(self):
        """ calculate vega """
        price, v = self.price, self.v
        vega_diff = v * 0.0001
        self.v = v + vega_diff
        self.calculate_price(compute_greeks=False)
        vega_up = self.price
        self.v = v - vega_diff
        self.calculate_price(compute_greeks=False)
        vega_down = self.price
        self.price, self.v = price, v
        return (vega_up - vega_down) / float(2.0 * vega_diff)
    
    
    def rho(self):
        """ calculate rho """
        price, r = self.price, self.r
        rho_diff = r * 0.0001
        self.r = r + rho_diff
        self.calculate_price(compute_greeks=False)
        rho_up = self.price
        self.r = r - (0 if (r - rho_diff) < 0 else rho_diff)
        self.calculate_price(compute_greeks=False)
        rho_down = self.price
        self.price, self.r = price, r
        return (rho_up - rho_down) / float((1.0 if r - rho_diff < 0 else 2.0) * rho_diff)
    
    
    def theta(self):
        """ calculate theta """
        price, maturity = self.price, self.maturity
        theta_diff = 1 / 252.0
        self.maturity = (pd.to_datetime(maturity) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.calculate_price(compute_greeks=False)
        theta_up = self.price
        self.maturity = (pd.to_datetime(maturity) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        self.calculate_price(compute_greeks=False)
        theta_down = self.price
        self.price, self.maturity = price, maturity
        return (theta_down - theta_up) / float(2.0 * theta_diff)
    

class BlackScholes(OptionPricing):
    """ Black-Scholes option pricing method """
    def __init__(self, callput, S, K, settlement, maturity, r, d, v):
        super().__init__(callput, S, K, settlement, maturity, r, d, v)
    
    
    def calculate_price(self, compute_greeks):
        """ calculate the Black-Scholes price of an option """
        self.T = get_T(self.settlement, self.maturity)
        d1 = (np.log(float(self.S) / self.K) + \
              ((self.r - self.d) + self.v * self.v / 2.0) * self.T) / (self.v * np.sqrt(self.T))
        d2 = d1 - self.v * np.sqrt(self.T)
        if self.callput == 'Call':
            self.price = self.S * np.exp(-self.d * self.T) * norm.cdf(d1) - \
            self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.callput == 'Put':
            self.price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - \
            self.S * np.exp(-self.d * self.T) * norm.cdf(-d1)
            
        if compute_greeks:
            self.calculate_greeks()
    
    
    def calculate_greeks(self):
        """ closed-form computations for Black-Scholes greeks """
        sgn = 1 if self.callput == 'Call' else -1
        T_sqrt = np.sqrt(self.T)
        if self.callput == 'Call':
            d1 = (np.log(float(self.S) / self.K) + \
                  ((self.r - self.d) + self.v * self.v / 2.0) * self.T) / (self.v * T_sqrt)
        elif self.callput == 'Put':
            d1 = (np.log(float(self.S) / self.K) + self.r * self.T) / \
                  (self.v * T_sqrt) + 0.5 * self.v * T_sqrt
        else:
            raise ValueError('callput must be either "Call" or "Put".')
        d2 = d1 - self.v * T_sqrt
        delta = sgn * norm.cdf(sgn * d1)
        gamma = norm.pdf(d1) / (self.S * self.v * T_sqrt)
        theta = -(self.S * self.v * norm.pdf(d1)) / (2 * T_sqrt) - \
                  sgn * self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        vega = self.S * T_sqrt * norm.pdf(d1)
        rho = sgn * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(sgn * d2)
        self.greeks = {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
        
        
class LongstaffSchwartzMonteCarlo(OptionPricing):
    """ Longstaff-Schwartz Monte Carlo option pricing method """
    def __init__(self, callput, S, K, settlement, maturity, r, d, v, timeSteps, requiredSamples):
        super().__init__(callput, S, K, settlement, maturity, r, d, v)
        self.M = timeSteps
        self.simulations = requiredSamples
    
    
    def calculate_price(self, compute_greeks):
        """ calculate option price """
        self.T = get_T(self.settlement, self.maturity)
        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)
        self.MCprice_matrix()
        self.MCpayoff()
        self.MCvalue_matrix()
        self.price = np.sum(self.value_matrix[1, :] * self.discount) / float(self.simulations)
        if compute_greeks:
            self.calculate_greeks()
    
    
    def MCprice_matrix(self, seed=42):
        """ Monte Carlo price matrix simulation """
        np.random.seed(seed)
        self.price_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        self.price_matrix[0, :] = self.S
        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal(self.simulations // 2)
            brownian = np.concatenate((brownian, -brownian))
            exp_a = (self.r - self.v ** 2 / 2.) * self.time_unit
            exp_b = self.v * brownian * np.sqrt(self.time_unit)
            self.price_matrix[t, :] = self.price_matrix[t - 1, :] * np.exp(exp_a + exp_b)
    
    
    def MCpayoff(self):
        """ inner option value """
        zeros_mat = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        if self.callput == 'Call':
            self.payoff = np.maximum(self.price_matrix - self.K, zeros_mat)
        elif self.callput == 'Put':
            self.payoff = np.maximum(self.K - self.price_matrix, zeros_mat)
    
    
    def MCvalue_matrix(self):
        """ value matrix """
        self.value_matrix = np.zeros_like(self.payoff)
        self.value_matrix[-1, :] = self.payoff[-1, :]
        for t in range(self.M - 1, 0 , -1):
            y_fit = self.value_matrix[t + 1, :] * self.discount
            regression = np.polyfit(self.price_matrix[t, :], y_fit, 5)
            continuation_value = np.polyval(regression, self.price_matrix[t, :])
            cond = self.payoff[t, :] > continuation_value
            x, y = self.payoff[t, :], self.value_matrix[t + 1, :] * self.discount
            self.value_matrix[t, :] = np.where(cond, x, y)


class CoxRossRubinstein(OptionPricing):
    """ Cox-Ross-Rubinstein option pricing method """
    def __init__(self, callput, S, K, settlement, maturity, r, d, v, steps):
        super().__init__(callput, S, K, settlement, maturity, r, d, v)
        self.n = steps
    
    
    def calculate_price(self, compute_greeks):
        """ calculate option price """
        self.T = get_T(self.settlement, self.maturity)
        self.dt = float(self.T) / float(self.n)
        
        # up and down moves
        self.up = np.exp(self.v * (np.sqrt(self.dt)))
        self.down = 1 / self.up
        
        # probabilities of up and down moves
        self.pu = (np.exp((self.r - self.d) * self.dt) - self.down) / (self.up - self.down)
        self.pd = 1 - self.pu

        self.build_trees()
        self.backwards_discount()
        self.price = self.payoff_tree[0][0]
        
        if compute_greeks:
            self.calculate_greeks()
    
    
    def build_trees(self):
        """ build stock and payoff tree """
        self.stock_tree = np.zeros((self.n + 1, self.n + 1))
        self.payoff_tree = np.zeros((self.n + 1, self.n + 1))
        iopt = 1 if self.callput == 'Call' else -1
        for j in range(self.n + 1):
            for i in range(j + 1):
                self.stock_tree[i][j] = self.S * (self.up ** i) * (self.down ** (j - i))
                self.payoff_tree[i][j] = np.maximum(0, iopt * (self.stock_tree[i][j] - self.K))
    
    
    def backwards_discount(self):
        """ backwards discounting """
        iopt = 1 if self.callput == 'Call' else -1
        for j in range(self.n - 1, -1, -1): 
            for i in range(j + 1):
                base_a = self.pu * self.payoff_tree[i + 1][j + 1] + self.pd
                base_b = self.payoff_tree[i][j + 1]
                exponent = -1 * (self.r - self.d) * self.dt
                self.payoff_tree[i][j] = base_a * base_b * np.exp(exponent)
                x1, x2 = iopt * (self.stock_tree[i][j] - self.K), self.payoff_tree[i][j]
                self.payoff_tree[i][j] = np.maximum(x1, x2)


class QuantLibPricer(OptionPricing):
    """ implementation of CRR, LSMC and BS option pricing and greek methods """
    def __init__(self, method, callput, S, K, settlement, maturity, r, d, v, opts=None):
        super().__init__(callput, S, K, settlement, maturity, r, d, v)
        self.method = method
        if callput == 'Call':
            self.option_type = ql.Option.Call
        elif callput == 'Put':
            self.option_type = ql.Option.Put
        self.opts = opts
    
    
    def handle_date_conventions(self):
        """ handle date conventions """
        self.settlement_date = ql.Date(*list(map(int, self.settlement.split('-')))[::-1])
        self.maturity_date = ql.Date(*list(map(int, self.maturity.split('-')))[::-1])
        ql.Settings.instance().evaluationDate = self.settlement_date
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.UnitedStates()
    
    
    def specify_option(self):
        """ specify option via payoff and exercise """
        payoff = ql.PlainVanillaPayoff(self.option_type, self.K)
        if self.method in ['crr', 'lsmc']:
            exercise = ql.AmericanExercise(self.settlement_date, self.maturity_date)
        elif self.method == 'bs':
            exercise = ql.EuropeanExercise(self.maturity_date)
        self.option = ql.VanillaOption(payoff, exercise)
    
    
    def specify_handles(self):
        """ specify handles for price inputs"""
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.S))
        rate_structure = ql.FlatForward(self.settlement_date, self.r, self.day_count)
        self.rate_handle = ql.YieldTermStructureHandle(rate_structure)
        dividend_structure = ql.FlatForward(self.settlement_date, self.d, self.day_count)
        self.dividend_handle = ql.YieldTermStructureHandle(dividend_structure)
        vol_structure = ql.BlackConstantVol(
            self.settlement_date, self.calendar, self.v, self.day_count)
        self.vol_handle = ql.BlackVolTermStructureHandle(vol_structure)
    
    
    def specify_engine(self):
        """ specify process and pricing engine """
        self.bsm_process = ql.BlackScholesMertonProcess(
            self.spot_handle, self.dividend_handle, self.rate_handle, self.vol_handle)
        if self.method == 'crr':
            self.engine = ql.BinomialVanillaEngine(self.bsm_process, 'crr', **self.opts)
        elif self.method == 'lsmc':
            self.engine = ql.MCAmericanEngine(
                self.bsm_process, 'PseudoRandom', **self.opts, polynomOrder=5)
        elif self.method == 'bs':
            self.engine = ql.AnalyticEuropeanEngine(self.bsm_process)
        self.option.setPricingEngine(self.engine)
    
    
    def calculate_price(self, compute_greeks):
        """ compute option price and greeks """
        self.handle_date_conventions()
        self.specify_option()
        self.specify_handles()
        self.specify_engine()
        self.price = self.option.NPV()
        if compute_greeks:
            self.greeks = dict()
            greeks_names = ['delta', 'gamma', 'rho', 'theta', 'vega']
            for greek in greeks_names:
                try:
                    self.greeks[greek] = getattr(self.option, greek)()
                except:
                    self.greeks[greek] = getattr(self, greek)()


def bond_price(par, settlement, maturity, ytm, coup, freq=2):
    """ pricing a bond """
    T = get_T(settlement, maturity)
    freq = float(freq)
    periods = int(T * freq)
    coupon = coup * par / freq
    dt = [(i + 1) / freq for i in range(int(periods))]
    price = sum([coupon / (1 + ytm / freq) ** (freq * t) for t in dt])
    price += par / (1 + ytm / freq) ** (freq * T)
    return price


def bond_ytm(price, par, settlement, maturity, coup, freq=2, guess=0.05):
    """ yield to maturity of a bond """
    T = get_T(settlement, maturity)
    freq = float(freq)
    periods = int(T * freq)
    coupon = coup * par
    dt = [(i+1) / freq for i in range(int(periods))]
    def ytm_func(y):
        ytm = sum([coupon / freq / (1 + y / freq) ** (freq * t) for t in dt])
        return ytm + par / (1 + y / freq) ** (freq * T) - price
    return optimize.newton(ytm_func, guess)


def bond_mod_duration(price, par, settlement, maturity, coup, freq, dy=0.0001):
    """ modified duration of a bond """
    T = get_T(settlement, maturity)
    ytm = bond_ytm(price, par, settlement, maturity, coup, freq)
    ytm_minus = ytm - dy    
    price_minus = bond_price(par, settlement, maturity, ytm_minus, coup, freq)
    ytm_plus = ytm + dy
    price_plus = bond_price(par, settlement, maturity, ytm_plus, coup, freq)
    mduration = (price_minus - price_plus) / (2 * price * dy)
    PV_01 = mduration * price * dy
    return {'mduration': mduration, 'PV_01': PV_01}


def bond_convexity(price, par, settlement, maturity, coup, freq, dy=0.0001):
    """ convexity of a bond """
    T = get_T(settlement, maturity)
    ytm = bond_ytm(price, par, settlement, maturity, coup, freq)
    ytm_minus = ytm - dy    
    price_minus = bond_price(par, settlement, maturity, ytm_minus, coup, freq)
    ytm_plus = ytm + dy
    price_plus = bond_price(par, settlement, maturity, ytm_plus, coup, freq)
    convexity = (price_minus + price_plus - 2 * price) / (price * dy ** 2)
    CVEX_01 = 0.5 * convexity * dy ** 2 * price
    return {'convexity': convexity, 'CVEX_01': CVEX_01}
    

def future_price(S, delivery_price, r, d, settlement, maturity):
    """ price of a future """
    T = get_T(settlement, maturity)
    return S * np.exp(-d * T) - delivery_price * np.exp(-r * T)


class FullValuation():
    """ value assets from risk factor levels, calcualte pnl """
    def __init__(self, positions_df, levels, rng, amer_opt_params=None):
        self.positions_df = positions_df.copy()
        self.levels = levels
        self.rng = rng
        if amer_opt_params is not None:
            self.option_pricing = amer_opt_params.pop('option_pricing')
            self.amer_opt_p_method = amer_opt_params.pop('amer_opt_p_method')
            self.amer_opt_params = amer_opt_params
    
        # get the option, bond and future price inputs
        self.options_price_inputs_mapper = self.get_price_inputs_mapper('Option')
        self.bonds_price_inputs_mapper = self.get_price_inputs_mapper('Bond')
        self.futures_price_inputs_mapper = self.get_price_inputs_mapper('Future')

        # map the asset type to the corresponding repricing function
        self.pricing_mapper_mapper = {
            'Equity':   self.equity_pricing_mapper,
            'Option':   self.option_pricing_mapper,
            'Bond':     self.bond_pricing_mapper,
            'Future':   self.future_pricing_mapper
        }
        
        # map the asset type to the corresponding valuation function
        self.valuation_mapper_mapper = {
            'Equity':   self.equity_valuation_mapper,
            'Option':   self.option_valuation_mapper,
            'Bond':     self.bond_valuation_mapper,
            'Future':   self.future_valuation_mapper
        }

        self.get_full_valuation()
    
    
    def price_inputs(self, i, asset_type):
        """ asset price inputs as dictionary """
        price_inputs = self.positions_df[i].loc['price_inputs', asset_type, :]
        return { k[-1]: v for k, v in dict(price_inputs).items() }
    
    
    def get_price_inputs_mapper(self, asset_type):
        """ pricing inputs for each asset of the provided type in the portfolio """
        price_inputs_mapper = dict()
        for i in self.positions_df.columns:
            if self.positions_df[i].loc['position', '', 'product_type'] == asset_type:
                price_inputs_mapper[i] = self.price_inputs(i, asset_type.lower())
        return price_inputs_mapper
    
    
    def equity_pricing_mapper(self, levels, i):
        """ price equities from levels """

        # get the needed price ticker
        price_ticker = self.positions_df[i].loc['position', '', 'price_ticker']

        # price the equity
        return levels[1][price_ticker]
    
    
    def option_pricing_mapper(self, levels, i):
        """ price options from levels """

        # get the option price inputs
        adj_option_price_inputs = self.options_price_inputs_mapper[i].copy()

        # get the tickers for the risk factors
        price_ticker = self.positions_df[i].loc['position', '', 'price_ticker']
        rate_ticker = self.positions_df[i].loc['position', '', 'rate_ticker']
        vol_ticker = self.positions_df[i].loc['position', '', 'vol_ticker']

        # adjust the option price inputs
        adj_option_price_inputs['S'] = levels[1][price_ticker]
        adj_option_price_inputs['r'] = levels[1][rate_ticker] / 100
        adj_option_price_inputs['v'] = levels[1][vol_ticker] / 100

        # price the option
        option_type = adj_option_price_inputs.pop('option_type')

        if self.option_pricing == 'custom':
            # use Black-Scholes for European options
            if option_type == 'European':
                pricer = BlackScholes(**adj_option_price_inputs)
                             
            # use numerical approach for pricing American options
            elif option_type == 'American':
                if self.amer_opt_p_method == 'lsmc':
                    pricer = LongstaffSchwartzMonteCarlo(
                        **adj_option_price_inputs, **self.amer_opt_params
                    )
                elif self.amer_opt_p_method == 'crr':
                    pricer = CoxRossRubinstein(
                        **adj_option_price_inputs, **self.amer_opt_params
                    )
            
        elif self.option_pricing == 'quantlib':
            method = 'bs' if option_type == 'European' else self.amer_opt_p_method
            pricer = QuantLibPricer(
                **adj_option_price_inputs, method=method, opts=self.amer_opt_params
            )
        
        pricer.calculate_price(compute_greeks=False)
        return pricer.price
    
    
    def bond_pricing_mapper(self, levels, i):
        """ price bonds from levels """

        # get the bond price inputs
        adj_bond_price_inputs = self.bonds_price_inputs_mapper[i].copy()

        # get the risk factor ticker
        yield_ticker = self.positions_df[i].loc['position', '', 'yield_ticker']

        # adjust the yield
        adj_bond_price_inputs['ytm'] = levels[1][yield_ticker] / 100

        # price the bond
        return bond_price(**adj_bond_price_inputs)
    
    
    def future_pricing_mapper(self, levels, i):
        """ price futures from levels """

        # get the future price inputs
        adj_future_price_inputs = self.futures_price_inputs_mapper[i].copy()

        # get the risk factor ticker
        price_ticker = self.positions_df[i].loc['position', '', 'price_ticker']

        # adjust the spot price
        adj_future_price_inputs['S'] = levels[1][price_ticker]

        # price the future
        return future_price(**adj_future_price_inputs)
    
    
    def equity_valuation_mapper(self, prices, i):
        """ value equity positions from prices """
        market_value = self.positions_df[i].loc['position', '', 'market_value']
        price = self.positions_df[i].loc['position', '', 'price']
        return prices * market_value / price
    
    
    def option_valuation_mapper(self, prices, i):
        """ value option positions from prices """
        market_value = self.positions_df[i].loc['position', '', 'market_value']
        notional_value = self.positions_df[i].loc['position', '', 'notional_value']
        S = self.positions_df[i].loc['price_inputs', 'option', 'S']
        return prices * notional_value / S
    
    
    def bond_valuation_mapper(self, prices, i):
        """ value bond positions from prices """
        securities_contracts = self.positions_df[i].loc['position', '', 'securities_contracts']
        market_value = self.positions_df[i].loc['position', '', 'market_value']
        return prices * securities_contracts / 100
    
    
    def future_valuation_mapper(self, prices, i):
        """ value future positions from prices """
        securities_contracts = self.positions_df[i].loc['position', '', 'securities_contracts']
        market_value = self.positions_df[i].loc['position', '', 'market_value']
        return prices * securities_contracts
    
    
    def get_full_valuation(self):
        """ value all positions and calculate pnls """
        # create a dataframe for valuations
        position_names = self.positions_df.loc['position', '', 'asset'].tolist()
        self.full_valuation = pd.DataFrame(columns=position_names, index=self.rng)

        # iterate over the positions in the portfolio
        for i, pos in self.positions_df.iteritems():
            # get the necessary identifiers
            asset = pos.loc['position', '', 'asset']
            product_type = pos.loc['position', '', 'product_type']
            market_value = pos.loc['position', '', 'market_value']
            
            # iterate over the levels time series and price asset
            mapper = lambda t: self.pricing_mapper_mapper[product_type](t, i)
            self.full_valuation[asset] = list(map(mapper, self.levels.iterrows()))
            
            # calculate position valuations
            self.full_valuation[asset] = self.valuation_mapper_mapper[product_type](
                self.full_valuation[asset], i
            )
            
            # calculate pnls
            self.full_valuation[asset] -= market_value
