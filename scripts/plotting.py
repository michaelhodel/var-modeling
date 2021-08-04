import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.dates import YearLocator

from scripts import helpers


def plot_time_series(df, title, ylabel, report):
    """ function for plotting multiple time series """
    fmt_year = YearLocator()
    for col in df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plt.gcf().subplots_adjust(bottom=0.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.plot(pd.to_datetime(df.index), df[col])
        ax.xaxis.set_minor_locator(fmt_year)
        ax.xaxis.set_tick_params(rotation=45)
        full_title = '{} {}'.format(col, title) if df.shape[1] != 1 else title
        report.write_plot(full_title)
        ax.set_title(full_title)
        plt.show()
        plt.close()


def plot_div_eff(labels, values, dollar, day_label, report):
    """ plotting VaR diversification effects """
    n_assets = len(labels) - 3
    symb = '$' if dollar else '%'
    f, ax = plt.subplots(figsize=(16, 8))
    plt.xticks(rotation=15)
    title = '{} Stand Alone VaR & Diversification Effect ({})'.format(day_label, symb)
    plt.xlabel('VaR / Effect', size=16);
    plt.ylabel(symb, size=16);

    # color the bars
    bars = plt.bar(labels, values);
    for asset in labels[:n_assets]:
        bars[labels.index(asset)].set_color('pink')
    for var_label in ['Undiv. VaR', 'Div. VaR']:
        bars[labels.index(var_label)].set_color('r')
    bars[labels.index('Div. Effect')].set_color('g')

    # adjust starting heights of bars
    for i in range(1, n_assets):
        pre_h = bars[labels.index(labels[i - 1])].get_height()
        pre_y = bars[labels.index(labels[i - 1])].get_y()
        bars[labels.index(labels[i])].set_y(pre_h + pre_y)

    div_eff_bounds = list(bars[labels.index('Div. Effect')].get_bbox().bounds)
    div_eff_bounds[1] = values[labels.index('Undiv. VaR')] - values[labels.index('Div. Effect')]
    bars[labels.index('Div. Effect')].set_bounds(*div_eff_bounds)

    # adjust plot height
    ax.set_ylim(0, max([bar.get_height() + bar.get_y() for bar in bars]) * 1.1)

    # add values as labels
    label_offset = max(values) * 0.005
    bot_lab_idx = labels.index('Div. Effect')
    tups = list(zip(ax.patches, values))
    rotation = -12.5 if dollar else 0
    for i, (rect, value) in enumerate(tups[:bot_lab_idx] + tups[bot_lab_idx + 1:]):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_y() + rect.get_height() + label_offset
        label_fmt = '${}' if dollar else '{} %'
        label = label_fmt.format(helpers.float_formatter(value, True))
        bar_rotation = rotation if i < len(tups) - 3 else 0
        ax.text(x, y, label, ha='center', va='bottom', fontsize=16, rotation=bar_rotation)
    div_eff_rect = bars[bot_lab_idx]
    label_fmt = '-${}' if dollar else '-{} %'
    ax.text(
        x=div_eff_rect.get_x() + div_eff_rect.get_width() / 2,
        y=div_eff_rect.get_y() - label_offset * 10,
        s=label_fmt.format(helpers.float_formatter(values[bot_lab_idx], True)),
        ha='center', va='bottom', fontsize=16)

    report.write_plot(title)
    plt.title(title, size=32);
    plt.show();
    plt.close();


def plot_var_decomp(labels, values, dollar, day_label, report):
    """ plotting VaR decomposition """
    symb = '$' if dollar else '%'
    f, ax = plt.subplots(figsize=(16, 8))
    plt.xticks(rotation=15)
    title = '{} Diversified VaR - Decomposition ({})'.format(symb, day_label)
    plt.xlabel('VaR / Effect', size=16);
    plt.ylabel(symb, size=16);

    # color the bars
    bars = plt.bar(labels, list(map(lambda v: abs(v), values)));
    for asset, value in zip(labels[:-1], values[:-1]):
        bars[labels.index(asset)].set_color('pink' if value > 0 else 'g')
    bars[-1].set_color('r' if values[-1] > 0 else 'g')

    # adjust starting heights of bars
    for i in range(1, len(labels[:-1])):
        pre_h, pre_y = bars[i - 1].get_height(), bars[i - 1].get_y()
        if values[i] > 0:
            bars[i].set_y(pre_h + pre_y)
        else:
            bars[i].set_y(pre_h + pre_y + values[i])

    ax.set_ylim(0, max([bar.get_height() + bar.get_y() for bar in bars]) * 1.1)

    # add values as labels
    label_offset = max(values) * 0.005
    rect_lab_tups = list(zip(ax.patches, values))
    rotation = -12.5 if dollar else 0
    for i, (rect, value) in enumerate(rect_lab_tups):
        x, y = rect.get_x() + rect.get_width() / 2, rect.get_y()
        y += rect.get_height() + label_offset if value > 0 else -label_offset * 20
        label_fmt = '${}' if dollar else '{} %'
        label = label_fmt.format(helpers.float_formatter(value, True))
        bar_rotation = - rotation if value < 0 else rotation
        bar_rotation = bar_rotation if i < len(rect_lab_tups) - 1 else 0
        ax.text(x, y, label, ha='center', va='bottom', fontsize=16, rotation=bar_rotation)

    report.write_plot(title)
    plt.title(title, size=32);
    plt.show();
    plt.close();


def div_effect_plots(diversification_effects, asset_metrics, ptf_market_value,
                     position_names, n_day_label, report):
    """ VaR diversification effect """
    # get a list of tuples for position labels and values
    lab_val_tups = list(zip(position_names, [*-asset_metrics.loc['Individual VaR', :]]))

    # sort the tuples by descending absolute value
    sorted_lab_val_tups = sorted(lab_val_tups, key=lambda t:(-t[1], t[0]))
    pos_bar_labels, pos_bar_values_USD = zip(*sorted_lab_val_tups)

    # add the portfolio-level measures
    bar_labels = list(pos_bar_labels) + ['Undiv. VaR', 'Div. Effect', 'Div. VaR']
    divs = diversification_effects.loc['', ['Undiv. VaR', 'Div. Effect ($)', 'Div. VaR']]
    bar_values_USD = list(pos_bar_values_USD) + (-divs.values).tolist()

    # calculate percentage values
    bar_values_pct = list(np.array(bar_values_USD) / ptf_market_value * 100)
    
    # plot the dollar and percentage measures
    for boolean, values in [(True, bar_values_USD), (False, bar_values_pct)]:
        plot_div_eff(
            labels=bar_labels, values=values, dollar=boolean,
            day_label=n_day_label, report=report
        )

    
def var_decomp_plots(component_VaR, ptf_market_value, position_names, n_day_label, report):
    """ diversified VaR decomposition """
    # get a list of tuples with labels and values for the positions
    lab_val_tups = list(zip(position_names, [*-component_VaR.loc['', :][:-1]]))

    # sort the list of position tuples by descending value
    sorted_lab_val_tups = sorted(lab_val_tups, key=lambda t:(-t[1], t[0]))
    pos_bar_labels, pos_bar_values_USD = zip(*sorted_lab_val_tups)

    # extend the list with the portfolio  VaR
    bar_labels = list(pos_bar_labels) + ['Portfolio']
    bar_values_USD = list(pos_bar_values_USD) + [-component_VaR.loc['', :][-1]]
    
    # calculate percentage values
    bar_values_pct = list(np.array(bar_values_USD) / ptf_market_value * 100)
    
    # plot the dollar and percentage measures
    for boolean, values in [(True, bar_values_USD), (False, bar_values_pct)]:
        plot_var_decomp(
            labels=bar_labels, values=values, dollar=boolean,
            day_label=n_day_label, report=report
        )

        
def plot_pnl_pdf(title, vol, var, lgnd, report):
    """ create a P&L probability density function """
    fig, ax = plt.subplots(1, figsize=(16, 8))
    mu, sigma = 0, vol
    x = np.linspace(mu - 5 * sigma, 5 * sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma));

    # add a vertical line for the VaR
    _ = plt.axvline(x=var, color='r', linestyle='--')

    # add title and labels
    _ = plt.legend(labels=[lgnd])
    _ = plt.xlabel('P&L', size=16)
    _ = plt.ylabel('density', size=16)

    # display the histogram
    report.write_plot(title)
    _ = plt.title(title, size=32)
    plt.show();
    plt.close();


def plot_pnl_hist(pnls, title, var, lgnd, report):
    """ create a histogram of the simulated P&L measures """
    fig, ax = plt.subplots(1, figsize=(16, 8))
    _ = plt.hist(pnls, bins=64)

    # add a vertical line for the VaR
    _ = plt.axvline(x=var, color='r', linestyle='--')

    # add title and labels
    _ = plt.legend(labels=[lgnd])
    _ = plt.xlabel('P&L', size=16)
    _ = plt.ylabel('nr. obs.', size=16)

    # display the histogram
    report.write_plot(title)
    _ = plt.title(title, size=32)
    plt.show();
    plt.close();

    
def bar_time_series(df, title, ylabel, report):
    """ time series bar chart for n-day returns """
    for col in df:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plt.gcf().subplots_adjust(bottom=0.25)
        df[col].plot.bar();
        ax.set_xticklabels([v if i % 4 == 0 else '' for i, v in enumerate(df.index)])
        ax.xaxis.set_tick_params(rotation=45, length=0);
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        full_title = title if df.shape[1] == 1 else '{} {}'.format(col, title)
        report.write_plot(full_title)
        plt.title(full_title)
        plt.show();
        plt.close();


def plot_corr_mat(matrix, title, report):
    """ plot correlation matrix """
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(matrix, annot=True, cmap='RdYlGn_r', ax=ax, vmin=-1, vmax=1);
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right');
    report.write_plot(title)
    plt.title(title);
    plt.show();
    plt.close();
