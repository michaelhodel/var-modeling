import pandas as pd
import numpy as np

from scripts import pricing


class PortfolioDefinition():
    """ compute measures for each asset in the portfolio """

    def __init__(self, position_configurations, rf_levels, report, amer_opt_params):
        self.position_configurations = position_configurations
        self.rf_levels = rf_levels
        self.report = report
        if amer_opt_params is not None:
            self.option_pricing = amer_opt_params.pop('option_pricing')
            self.amer_opt_p_method = amer_opt_params.pop('amer_opt_p_method')
            self.amer_opt_params = amer_opt_params
    
    
    def initialize_positions_df(self):
        """ create a dataframe for the positions information """

        # specify the multiindex for the positions dataframe
        multiindex = {
            'position': {
                '': ['product_type', 'asset', 'price_ticker', 'rate_ticker',
                    'yield_ticker', 'vol_ticker', 'price', 'securities_contracts',
                    'notional_value', 'direction', 'market_value']
            },
            'price_inputs': {
                'option': ['option_type', 'callput', 'settlement',
                           'maturity', 'r', 'd', 'K', 'v', 'S'],
                'bond': ['settlement', 'maturity', 'coup', 'ytm', 'par', 'freq'],
                'future': ['S', 'delivery_price', 'r', 'd', 'settlement', 'maturity']
            },
            'price_sensitivities': {
                '': ['risk_characteristic'],
                'sec': ['delta', 'gamma', 'vega', 'pv01', 'cvex01'],
                'pos': ['delta', 'gamma', 'vega', 'pv01', 'cvex01']
            }
        }

        outer_index, mid_index, inner_index = [], [], []
        for outer_k, outer_v in multiindex.items():
            for inner_k, inner_v in outer_v.items():
                outer_index.extend([outer_k] * len(inner_v))
                mid_index.extend([inner_k] * len(inner_v))
                inner_index.extend(inner_v)

        # initialize an empty dataframe with as many columns as specified positions
        idx = pd.MultiIndex.from_arrays([outer_index, mid_index, inner_index])
        cols = list(range(len(self.position_configurations)))
        self.positions_df = pd.DataFrame(None, idx, cols)
    
    
    def specify_equity_position(self, idx, config):
        """ add the information of an equity position to the positions dataframe """
        # fixed values product type and risk characteristic
        self.positions_df[idx].loc['position', '', 'product_type'] = 'Equity'
        self.positions_df[idx].loc['price_sensitivities', '', 'risk_characteristic'] = 'Linear'

        # specify the position name based on the ticker and direction
        self.positions_df[idx].loc['position', '', 'asset'] = '{} {}'.format(
            config['price_ticker'], config['direction']
        )
        
        # add inputs from the equity position configuration
        price_ticker = config['price_ticker']
        level = self.rf_levels[price_ticker][0]
        contracts = float(config['securities_contracts'])
        self.positions_df[idx].loc['position', '', 'price_ticker'] = price_ticker
        self.positions_df[idx].loc['position', '', 'price'] = level
        self.positions_df[idx].loc['position', '', 'securities_contracts'] = contracts
        self.positions_df[idx].loc['position', '', 'direction'] = config['direction']
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'delta'] = 1.0 
        
        # compute the market value
        self.positions_df[idx].loc['position', '', 'market_value'] = level * contracts
        
        # add the delta
        self.positions_df[idx].loc['price_sensitivities', 'pos', 'delta'] = contracts
    
    
    def specify_option_position(self, idx, config):
        """ add the information of an option position to the positions dataframe """
        # fixed values product type and risk characteristic
        self.positions_df[idx].loc['position', '', 'product_type'] = 'Option'
        self.positions_df[idx].loc['price_sensitivities', '', 'risk_characteristic'] = 'Nonlinear'

        # specify the position name based on the ticker and the option type
        self.positions_df[idx].loc['position', '', 'asset'] = '{} {} {} {}'.format(
            config['option_type'], config['price_ticker'], config['callput'], config['direction']
        )
        
        # add inputs from the option position configuration
        for underlying in ['price_ticker', 'rate_ticker', 'vol_ticker']:
            self.positions_df[idx].loc['position', '', underlying] = config[underlying]
            
        contracts = float(config['securities_contracts'])
        self.positions_df[idx].loc['price_inputs', 'option', 'option_type'] = config['option_type']
        self.positions_df[idx].loc['price_inputs', 'option', 'callput'] = config['callput']
        self.positions_df[idx].loc['price_inputs', 'option', 'settlement'] = config['settlement']
        self.positions_df[idx].loc['price_inputs', 'option', 'maturity'] = config['maturity']
        self.positions_df[idx].loc['price_inputs', 'option', 'd'] = float(config['d'])
        self.positions_df[idx].loc['position', '', 'securities_contracts'] = contracts
        self.positions_df[idx].loc['position', '', 'direction'] = config['direction']

        # add the option price inputs (in basis points where appropriate)
        r = self.rf_levels[config['rate_ticker']][0] / 100
        S = self.rf_levels[config['price_ticker']][0]
        v = self.rf_levels[config['vol_ticker']][0] / 100
        K = self.rf_levels[config['price_ticker']][0]
        
        self.positions_df[idx].loc['price_inputs', 'option', 'r'] = r
        self.positions_df[idx].loc['price_inputs', 'option', 'S'] = S
        self.positions_df[idx].loc['price_inputs', 'option', 'v'] = v
        self.positions_df[idx].loc['price_inputs', 'option', 'K'] = K
        
        # compute the option price based on the option price inputs
        inputs_series = self.positions_df[idx].loc['price_inputs', 'option', :]
        inputs = { k[-1]: v for k, v in dict(inputs_series).items()}
        option_type = inputs.pop('option_type')

        if self.option_pricing == 'custom':
            # use Black and Scholes for pricing European options
            if option_type == 'European':
                pricer = pricing.BlackScholes(**inputs)
            
            # use numerical approach for pricing American options
            elif option_type == 'American':
                if self.amer_opt_p_method == 'lsmc':
                    pricer = pricing.LongstaffSchwartzMonteCarlo(**inputs, **self.amer_opt_params)
                elif self.amer_opt_p_method == 'crr':
                    pricer = pricing.CoxRossRubinstein(**inputs, **self.amer_opt_params)
        
        elif self.option_pricing == 'quantlib':
            method = 'bs' if option_type == 'European' else self.amer_opt_p_method
            pricer = pricing.QuantLibPricer(**inputs, method=method, opts=self.amer_opt_params)
        
        pricer.calculate_price(compute_greeks=True)
        price, greeks = pricer.price, pricer.greeks
        
        # add the price and greeks
        self.positions_df[idx].loc['position', '', 'price'] = price
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'delta'] = greeks['delta']
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'gamma'] = greeks['gamma']
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'vega'] = greeks['vega'] / 100
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'pv01'] = greeks['rho'] / 10000

        # compute the notional value
        notional = S * contracts
        self.positions_df[idx].loc['position', '', 'notional_value'] = notional

        # compute the market value
        self.positions_df[idx].loc['position', '', 'market_value'] = price * notional / S
        
        # adjust the greeks
        for greek in ['delta', 'vega', 'gamma', 'pv01']:
            sec_grk = self.positions_df[idx].loc['price_sensitivities', 'sec', greek]
            grk = sec_grk * contracts
            self.positions_df[idx].loc['price_sensitivities', 'pos', greek] = grk
    
    
    def specify_bond_position(self, idx, config):
        """ add the information of a bond position to the positions dataframe """
        # fixed values product type and risk characteristic
        self.positions_df[idx].loc['position', '', 'product_type'] = 'Bond'
        self.positions_df[idx].loc['price_sensitivities', '', 'risk_characteristic'] = 'Nonlinear'

        # specify the position name from the ticker, product type (bond) and direction
        self.positions_df[idx].loc['position', '', 'asset'] = '{} Bond {}'.format(
            config['yield_ticker'], config['direction']
        )
        
        # add inputs from the bond position configuration
        contracts = float(config['securities_contracts'])
        self.positions_df[idx].loc['position', '', 'yield_ticker'] = config['yield_ticker']
        self.positions_df[idx].loc['price_inputs', 'bond', 'coup'] = float(config['coup'])
        self.positions_df[idx].loc['price_inputs', 'bond', 'freq'] = int(config['freq'])
        self.positions_df[idx].loc['price_inputs', 'bond', 'par'] = float(config['par'])
        self.positions_df[idx].loc['price_inputs', 'bond', 'settlement'] = config['settlement']
        self.positions_df[idx].loc['price_inputs', 'bond', 'maturity'] = config['maturity']
        self.positions_df[idx].loc['position', '', 'securities_contracts'] = contracts
        self.positions_df[idx].loc['position', '', 'direction'] = config['direction']
        
        # compute the yield to maturity
        ytm = self.rf_levels.loc[:, config['yield_ticker']][0] / 100
        self.positions_df[idx].loc['price_inputs', 'bond', 'ytm'] = ytm
        
        # compute the bond price based on the bond price inputs
        price_inputs_series = self.positions_df[idx].loc['price_inputs', 'bond', :]
        price_inputs = { k[-1]: v for k, v in dict(price_inputs_series).items()}
        price = pricing.bond_price(**price_inputs)
        self.positions_df[idx].loc['position', '', 'price'] = price

        # compute the market value
        self.positions_df[idx].loc['position', '', 'market_value'] = price * contracts / 100
        
        # specify the inputs required for calculating the greeks
        keys = ['par', 'settlement', 'maturity', 'coup', 'freq']
        bond_greeks_input = {
            'price': price, **{
                k: self.positions_df[idx].loc['price_inputs', 'bond', k] for k in keys
            }
        }

        # compute the modified duration
        pv01 = pricing.bond_mod_duration(**bond_greeks_input)['PV_01']
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'pv01'] = pv01
        
        # compute the convexity
        cvex01 = pricing.bond_convexity(**bond_greeks_input)['CVEX_01']
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'cvex01'] = cvex01

        # compute position sensitivities
        self.positions_df[idx].loc['price_sensitivities', 'pos', 'pv01'] = pv01 * -contracts / 100
        self.positions_df[idx].loc['price_sensitivities', 'pos', 'cvex01'] = cvex01 * contracts / 100
    
    
    def specify_future_position(self, idx, config):
        """ add the information of a future position to the positions dataframe """
        # fixed values product type, risk characteristic and delta
        self.positions_df[idx].loc['position', '', 'product_type'] = 'Future'
        self.positions_df[idx].loc['price_sensitivities', '', 'risk_characteristic'] = 'Linear'
        self.positions_df[idx].loc['price_sensitivities', 'sec', 'delta'] = 1.0

        # specify the position name from the ticker, product type (Future) and direction
        self.positions_df[idx].loc['position', '', 'asset'] = '{} Future {}'.format(
            config['price_ticker'], config['direction']
        )
        
        # add inputs from the future position configuration
        contracts = float(config['securities_contracts'])
        self.positions_df[idx].loc['position', '', 'price_ticker'] = config['price_ticker']
        self.positions_df[idx].loc['position', '', 'direction'] = config['direction']
        self.positions_df[idx].loc['position', '', 'securities_contracts'] = contracts
        
        # add float price inputs
        for numerical in ['r', 'd', 'delivery_price']:
            val = float(config[numerical])
            self.positions_df[idx].loc['price_inputs', 'future', numerical] = val
        
        self.positions_df[idx].loc['price_inputs', 'future', 'settlement'] = config['settlement']
        self.positions_df[idx].loc['price_inputs', 'future', 'maturity'] = config['maturity']
        
        # add the spot price
        S = self.rf_levels[config['price_ticker']][0]
        self.positions_df[idx].loc['price_inputs', 'future', 'S'] = S

        # compute the the future price based on future price inputs
        price_inputs_series = self.positions_df[idx].loc['price_inputs', 'future', :]
        price_inputs = { k[-1]: v for k, v in dict(price_inputs_series).items()}
        price = pricing.future_price(**price_inputs)
        self.positions_df[idx].loc['position', '', 'price'] = price

        # compute the notional value
        delivery = self.positions_df[idx].loc['price_inputs', 'future', 'delivery_price']
        self.positions_df[idx].loc['position', '', 'notional_value'] = contracts * delivery
        
        # compute the market value
        market_value = price * contracts
        self.positions_df[idx].loc['position', '', 'market_value'] = market_value
        
        # compute the delta
        self.positions_df[idx].loc['price_sensitivities', 'pos', 'delta'] = market_value / price
    
    
    def construct_portfolio(self):
        """ construct the portfolio """
        self.initialize_positions_df()

        # a mapper from position specification to position based on product type
        position_config_mapper = {
            'Equity': self.specify_equity_position,
            'Option': self.specify_option_position,
            'Bond': self.specify_bond_position,
            'Future': self.specify_future_position
        }

        # iterate over the position configurations dataframe
        for i, pos_config in self.position_configurations.iterrows():
            # assign values to corresponding column in positions dataframe
            position_config_mapper[pos_config['type']](i, pos_config)

        return self.positions_df
    
    
    def get_asset_overview(self, asset_type):
        """ create asset overview for each asset of asset_type """
        # keep only assets of specified asset type in df
        mask = self.positions_df.loc['position', '', 'product_type'] == asset_type
        df = self.positions_df.copy()[mask[mask].index]
        # drop redundant rows (nan values)
        df = df.dropna()
        # convert multiindex to regular index
        idx = df.index.get_level_values(2).tolist()
        prefixes = [i if i in ['sec', 'pos'] else '' for i in df.index.get_level_values(1)]
        df.index = [p + ' ' + i if p != '' else i for p, i in list(zip(prefixes, idx))]
        # drop unneccessary rows
        df = df.drop(['product_type', 'asset'])
        return df
    
    
    def log_asset_overviews(self):
        """ create an asset overview for each present asset type """
        # get a list of all asset types for which there are positions
        present_asset_types = self.positions_df.loc['position', '', 'product_type'].tolist()
        for asset_type in ['Equity', 'Option', 'Bond', 'Future']:
            if asset_type in present_asset_types:
                # create an overview of the assets in that asset class
                df = self.get_asset_overview(asset_type)
                self.report.write_table(df, '{} Positions Overview'.format(asset_type))
    
    
    def run(self):
        """ constructs the portfolio, portfolio market value, logs information to report """
        # construct the portfolio
        self.construct_portfolio()
        # get a list of position names
        position_names = self.positions_df.loc['position', '', 'asset'].tolist()
        # add the position names to the reporting class
        setattr(self.report, 'position_names', list(position_names))

        # log separate tables for each asset class to the report
        idx = ['Asset {}'.format(i) for i in range(len(position_names))]
        assets_table = pd.DataFrame(position_names, index=idx, columns=['Position Name'])
        self.report.write_table(assets_table, 'Portfolio Assets')
        self.log_asset_overviews()
        
        # compute the portfolio market value, display
        ptf_market_value = self.positions_df.loc['position', '', 'market_value'].sum()
        ptf_market_value_report = 'Portfolio Market Value ($):\t{}'.format(round(ptf_market_value))
        print(ptf_market_value_report)
        self.report.write(ptf_market_value_report)
    
        return self.positions_df, ptf_market_value, position_names
