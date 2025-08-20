from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import pandas as pd
import streamlit as st 
import os 
import numpy as np 
from streamlit_toggle import st_toggle_switch
import re
import io 
from scipy.stats import norm
from scipy.optimize import minimize
import time
import numpy_financial as npf
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker 
from PIL import Image, ImageDraw, ImageFont 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

# Function - Headers P90 and P10 
def render_header(label, column):
    html = f"""
    <div style="text-align: center; font-weight: 600;">
        {label}
    </div>
    """
    column.markdown(html, unsafe_allow_html=True)

# Function - Customized Font in Bold 
def font_size(text, font_size):
    html = f'<p style="font-size:{font_size}px; font-weight:bold;">{text}</p>'
    st.markdown(html, unsafe_allow_html=True)

# Function - Descriptor Text
def description(descriptor):
    html = f"""<div style="display: flex; align-items: center; height: 38px;">
            <span style="margin: auto 0;">{descriptor}</span>
            </div>
            """
    st.markdown(html, unsafe_allow_html=True)

# Function - Descriptor Checkbox
def description_checkbox(descriptor):
    html = f"""<div style="display: flex; align-items: center; height: 45px;">
            <span style="margin: auto 0;">{descriptor}</span>
            </div>
            """
    st.markdown(html, unsafe_allow_html=True)

# Function - Write Data to Excel
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Function - Find First Year
def find_first_production(production_array):
    for i, val in enumerate(production_array):
        if val > 0:
            return i
    return None # if no production found

# Function - Straight Line Depreciation
def calc_straight_line_depreciation(capex_array, production_array, depreciation_years):
    first_prod_year = find_first_production(production_array)
    if first_prod_year is None:
        return np.zeros_like(production_array)
    
    total_investment = capex_array.sum()
    annual_depr = total_investment/depreciation_years

    n_years = len(production_array)
    depreciation_schedule = np.zeros(n_years)

    for year in range(first_prod_year, min(first_prod_year + depreciation_years, n_years)):
        depreciation_schedule[year] = annual_depr 

    return depreciation_schedule

# Function - Declining Balance
def calc_declining_balance_depreciation(capex_array, production_array, depreciation_rate):
    first_prod_year = find_first_production(production_array)
    max_years = len(production_array)
    depn_open = np.zeros(max_years)
    depn_add = np.zeros(max_years)
    depn = np.zeros(max_years)
    depn_close = np.zeros(max_years)
    depn_open[first_prod_year] = np.sum(capex_array[:first_prod_year+1])
    
    total_investment = capex_array.sum()
    for t in range(first_prod_year, max_years):
        if t > first_prod_year:
            depn_open[t] = depn_close[t-1]
        depn[t] = (depn_open[t] + capex_array[t]) * depreciation_rate/100
        depn_close[t] = depn_open[t] + capex_array[t] - depn[t]
    
    return depn

# Function - Univariate Price Volatility
def price_vol_process(price_series, volatility, mean_reversion, long_term_mean=0, drift=0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    years = len(price_series)
    dt = 1
    prices = np.zeros(years)
    prices[0] = price_series[0]
    returns = np.zeros(years)
    returns[0] = np.log( price_series[1] / price_series[0] )
    price_min = price_series[0] * 0.5
    price_max = price_series[0] * 1.4

    for t in range(1, years):
        dW = np.random.normal(0, np.sqrt(dt))
        dr = mean_reversion * (long_term_mean - returns[t-1]) * dt + volatility * np.sqrt(dt) * np.random.normal()

        returns[t] = dr 
        prices[t] = prices[t-1] * np.exp(dr)

        # Clip prices
        prices[t] = np.clip( prices[t], price_min, price_max)

    return prices

# Function - Fit Lognormal Distributions from P10, P50, P90
def fit_lognormal_from_p10_p50_p90(p10, p50, p90):
    z_10 = norm.ppf(0.10)
    z_90 = norm.ppf(0.90)

    mu = np.log(p50)
    sigma = (np.log(p90) - np.log(p10)) / (z_90 - z_10)

    return mu, sigma

# Function - Define Triangular PPF
def triangular_ppf(u, a, c, b):
    F_c = (c - a) / (b - a)
    x = np.empty_like(u, dtype=float)
    left_mask = u <= F_c 
    # for left part: x = a + sqrt(u*(b-a)*(c-a))
    x[left_mask] = a + np.sqrt(u[left_mask] * (b-a) * (c-a))
    x[~left_mask] = b - np.sqrt((1 - u[~left_mask]) * (b-a) * (b-c))
    return x 

def fit_triangular_from_p10_p50_p90(p10, p50, p90, x0 = None):
    probs = np.array([0.1, 0.5, 0.9])
    y = np.array([p10, p50, p90])

    if x0 is None:
        a0 = min(p10, p50, p90)
        b0 = max(p10, p50, p90)
        c0 = p50

        if not (a0 < c0 < b0):
            c0 = a0 + 0.5 * (b0 - a0)
        x0 = np.array([a0, c0, b0], dtype=float)

    def obj(x):
        a, c, b = x
        if not (a < c < b):
            return 1e12 + ( a - c )**2 + (c - b)**2
        model = triangular_ppf(probs, a, c, b)
        return np.sum((model - y)**2)
    
    rng = (p90 - p10) if (p90 > p10) else (abs(p50) + 1.0)
    bounds = [(p10 - 5*rng, p10 + 1*rng), (p50 - 2*rng, p50 + 2*rng), (p90 - 1*rng, p90 + 5*rng)]
    res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B')
    if not res.success:
        a, c, b = p10, p50, p90
    else:
        a, c, b = res.x
    return float(a), float(c), float(b)

# Function - Calculate PSC Cost Recovery
def calc_psc_cost_recovery(recoverables, revenue, cost_recovery_ceiling):
    
    recoverables = np.array(recoverables)
    revenue = np.array(revenue)

    max_years = len(revenue)
    cr_open = np.zeros(max_years)
    cr_add = np.zeros(max_years)
    cr_use = np.zeros(max_years)
    cr_close = np.zeros(max_years)
    cost_recovery_ceiling_calc = cost_recovery_ceiling / 100
    max_cost_recovery = revenue * cost_recovery_ceiling_calc
    cr_use[0] = np.minimum(max_cost_recovery[0], recoverables[0])
    cr_close[0] = revenue[0] - cr_use[0]

    for t in range(1, max_years):
        cr_open[t] = cr_close[t-1]
        cr_add[t] = recoverables[t]
        cr_use[t] = np.minimum(cr_open[t] + cr_add[t], max_cost_recovery[t])
        cr_close[t] = cr_open[t] + cr_add[t] - cr_use[t]

    return cr_use

def calc_PSC(capex_array, production_array, price_array, opex_array, cost_recovery_ceiling):
    if psc_capex_depreciation_method == "Straight Line":
        depreciation = calc_straight_line_depreciation(capex_array, production_array, psc_dep_sl_years)
    elif psc_capex_depreciation_method == "Declining Balance":
        depreciation = calc_declining_balance_depreciation(capex_array, production_array, psc_dep_db_rate)
    else:
        depreciation = capex_array

    incl_opex = 1 if psc_include_opex == "Yes" else 0
    recoverables = opex_array * incl_opex + depreciation 
    
    revenue = price_array * production_array
    cost_recovery = calc_psc_cost_recovery(recoverables, revenue, cost_recovery_ceiling)
    profit_oil = revenue - cost_recovery 
    project_costs = capex_array + opex_array 

    if psc_split_method == "Profit Oil":
        govt_take = profit_oil * psc_profit_oil_gov/100 
    else: 
        r_factor_inflows = revenue if r_factor_method == "Cuml Revenues / Cuml Costs" else profit_oil
        r_factor = calc_R_factor(rfactors, rfactor_gov_shares, project_costs, r_factor_inflows)
        govt_take = profit_oil * r_factor 

    ncf = revenue - capex_array - opex_array - govt_take 
    return ncf 

def calc_concession(capex_array, production_array, price_array, opex_array, royalty_input):
    revenue = price_array * production_array
    if royalty_basis == "Earnings":

        if royalty_capex_depreciation_method == "Straight Line":
            depreciation = calc_straight_line_depreciation(capex_array, production_array, royalty_dep_sl_years)
        elif royalty_capex_depreciation_method == "Declining Balance":
            depreciation = calc_declining_balance_depreciation(capex_array, production_array, royalty_dep_sl_years)
        else:
            depreciation = capex_array
        
        royalties = ( revenue - depreciation - opex_array ) * royalty_input/100 
        
    else: 
        royalties = revenue * royalty_input/100

    ncf = revenue - capex_array - opex_array - royalties 
    return ncf


# Function - Decline Curve Functions: Exponential, Harmonic, Hyperbolic
def constrained_decline(initial_prod, decline_rate, total_resource, capacity, model='Exponential', b=0.5, max_years=15):

    t_days = []
    prod_daily = []
    cum_prod_daily = 0
    day = 0
    days_in_yr = 365
    max_days = int(max_years * days_in_yr)
    decline = decline_rate/100
    total_resource_bbl = total_resource * 10**6

    while cum_prod_daily < total_resource_bbl and day < max_days:
        if model == "Exponential":
            q = initial_prod * np.exp(-decline * day)
        elif model == "Harmonic":
            q = initial_prod / (1 + decline * day )
        elif model == "Hyperbolic":
            q = initial_prod / ((1 + b * decline * day) ** (1 / b))
        else:
            raise ValueError("Invalid model type")

        q = min(q, capacity)

        if cum_prod_daily + q > total_resource_bbl:
            q = total_resource_bbl - cum_prod_daily 
        
        prod_daily.append(q)
        t_days.append(day)
        cum_prod_daily += q
        day += 1

    prod_daily = np.array(prod_daily)
    t_days = np.array(t_days)
    cum_prod_daily = np.cumsum(prod_daily)

    total_days = len(prod_daily)
    prod_yearly = np.zeros(max_years)

    for i in range(max_years):
        start_day = int(i * days_in_yr)
        end_day = int(min((i+1) * days_in_yr, total_days))
        prod_yearly[i] = prod_daily[start_day:end_day].sum() / 10**6

    cum_prod_yearly = np.cumsum(prod_yearly)

    return prod_yearly, cum_prod_yearly, prod_daily, cum_prod_daily

# Function - Fit Production Profile to Start Year 
def gen_production_profile(initial_prod, decline_rate, resource_size, capacity, p50_prod_profile, model="Exponential", b=0.5):
    first_prod_year = find_first_production(p50_prod_profile)
    if first_prod_year is None:
        return np.zeros(len(p50_prod_profile))
    
    max_prod_years = len(p50_prod_profile) - first_prod_year
    if max_prod_years <= 0:
        return np.zeros(len(p50_prod_profile))
    
    annual_prod, _, _, _ = constrained_decline(initial_prod, decline_rate, resource_size, capacity, model, b, max_prod_years)
    fitted_prod_profile = np.concatenate((np.zeros(first_prod_year), np.array(annual_prod)))
    return fitted_prod_profile

# Function - Create R-Factor Time_Series
def calc_R_factor(rfactors_series, gov_take_series, project_costs_array, project_inflows_array):
    cum_costs = project_costs_array.cumsum()
    cum_inflows = project_inflows_array.cumsum()
    R_factor = np.where(cum_inflows > 0, cum_inflows / cum_costs, 0)

    bins = rfactors_series
    tranche_indices = np.digitize(R_factor, bins, right=True)
    gov_shares_array = np.array(gov_take_series)/100
    tranche_indices_clipped = np.clip(tranche_indices, 0, len(gov_take_series) - 1)
    R_factor_series = gov_shares_array[tranche_indices_clipped]
    return R_factor_series

# Function - Calculate Payback
def calc_payback(cash_flow_array):
    cf_series = pd.Series(cash_flow_array)
    cum_cash_flow = cf_series.cumsum()
   
    leading_zeros = (cum_cash_flow == 0).sum()

    neg_indices = np.where(cum_cash_flow < 0)[0]
    if len(neg_indices) == 0:
        # Payback occurs in year 0 or cash flow is positive from the start
        return 0.0

    last_neg_year = neg_indices.max()
    last_neg_cum_cf = cum_cash_flow.iloc[last_neg_year]

    # Check if last_neg_year is the last element
    if last_neg_year + 1 >= len(cash_flow_array):
        # Cannot divide by next year; payback not reached
        return np.nan  # or some large value to indicate no payback

    next_cf = (
        cash_flow_array.iloc[last_neg_year + 1]
        if isinstance(cash_flow_array, pd.Series)
        else cash_flow_array[last_neg_year + 1]
    )
    fraction = -last_neg_cum_cf / next_cf

    Payback = np.round(last_neg_year + fraction - leading_zeros, 2)
    return Payback

# Function - Add Watermark
def add_watermark(ax, logo_path, num_logos=1, zoom=0.30, alpha=0.3):
    # Load and resize logo
    logo_img = Image.open(logo_path).convert("RGBA")
    
    # Create combined watermark with multiple logos
    padding = 30
    logos = []
    for i in range(num_logos):
        logos.append(logo_img)
    total_width = logo_img.width * num_logos + padding * (num_logos - 1)
    total_height = logo_img.height
    watermark_img = Image.new("RGBA", (total_width, total_height), (255,255,255,0))
    
    for i, logo in enumerate(logos):
        x = i * (logo.width + padding)
        watermark_img.paste(logo, (x,0), logo)
    
    # Convert to array for OffsetImage
    watermark_np = np.array(watermark_img)
    
    # Create OffsetImage
    im = OffsetImage(watermark_np, zoom=zoom, alpha=alpha)
    
    # Place at axes center
    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    ab = AnnotationBbox(im, (x_center, y_center), frameon=False, box_alignment=(0.5,0.5))
    
    ax.add_artist(ab)

# Function - Set Chart Style     
def set_chart_style(fig_height, base_font_size=9):
    scale_factor = fig_height / 7  # 7 is cash flow chart height
    plt.rcParams.update({
        'font.size': base_font_size * scale_factor,
        'axes.labelsize': base_font_size * scale_factor * 1.1,
        'axes.titlesize': base_font_size * scale_factor * 1.1,
        'xtick.labelsize': base_font_size * scale_factor,
        'ytick.labelsize': base_font_size * scale_factor,
        'legend.fontsize': base_font_size * scale_factor
    })

# Function - Set Style Axis 
def style_axes(ax, grid_axis='y', grid_color='#d3d3d3', grid_linestyle='--', grid_alpha=0.3):
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#d3d3d3')
        spine.set_linewidth(0.8)
    ax.grid(True, axis=grid_axis, linestyle=grid_linestyle, alpha=grid_alpha, color=grid_color)
    
    # Color negative ticks
    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if tick < 0:
            label.set_color('red')
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick < 0:
            label.set_color('red')



# Function - Plot P50 Cash Flows
def plot_P50_cashflow(years_array, revenue_array, capex_array, opex_array, govt_take_array, truncate_len=None):
    
    if truncate_len is not None:
        years_array = years_array[:truncate_len]
        revenue_array = revenue_array[:truncate_len]
        capex_array = capex_array[:truncate_len]
        opex_array = opex_array[:truncate_len]
        govt_take_array = govt_take_array[:truncate_len]
    
    start_year = int(years_array[0])
    project_years = int(len(years_array))
    years = np.arange(start_year, start_year + project_years)
    ncf = revenue_array - capex_array - opex_array - govt_take_array
    cumulative_ncf = np.cumsum(ncf)

    fig, ax = plt.subplots(figsize=(14,4), facecolor='none')
    ax.set_facecolor('white')

    # Revenues (positive bars)
    ax.bar(years, revenue_array, label='Revenue', color='#a8ddb5')

    # Negative cash flows: stack from bottom 
    bar1 = ax.bar(years, -capex_array, label='Capex', color='#fbb4b9')
    bar2 = ax.bar(years, -opex_array, bottom=-capex_array, label='Opex', color='#fdae6b')
    bar3 = ax.bar(years, -govt_take_array, bottom=-(capex_array+opex_array), label='Govt Take', color='#bcbddc')
    
    ax.plot(years, ncf, label='P50 NCF', color='#6baed6', linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(years, cumulative_ncf, label='Cumulative NCF', color='#de2d26', linewidth=2)
    ax2.set_ylabel('Cumulative Cash Flow (USD mln)', labelpad=15)
    ax2.set_facecolor('none')

    for spine in ax.spines.values():
        spine.set_color('#d3d3d3')
    for spine in ax2.spines.values():
        spine.set_color('#d3d3d3')

    lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    ax.axhline(0, color='black', linewidth=1)

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Annual Cash Flow (USD mln)')

    # Fix X-axis: integer year ticks only 
    max_ticks = 20
    n_years = len(years)
    if n_years > max_ticks:
        step = n_years // max_ticks 
    else:
        step = 1

    ax.set_xticks(years[::step])
    ax.set_xticklabels(years[::step])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d')) # format as integer 

    # Fix Y-axis: adding padding so bars don't touch borders
    all_cashflows = np.concatenate([
        revenue_array, 
        -(capex_array + opex_array + govt_take_array),
        ncf
    ])
    y_min, y_max = all_cashflows.min(), all_cashflows.max()
    y_range = y_max - y_min 
    padding = y_range * 0.05 # Add 5% padding to the top and bottom 
    ax.set_ylim(y_min - padding, y_max + padding)

    y1_lo, y1_hi = ax.get_ylim()
    y2_lo, y2_hi = ax2.get_ylim()

    frac1 = (0 - y1_lo) / (y1_hi - y1_lo) if (y1_hi - y1_lo) != 0 else 0.5
    
    cum_range = y2_hi - y2_lo
    padding = cum_range * 0.05  # 5% padding

    # Set new limits keeping 0 anchored
    new_y2_lo = y2_lo - padding - frac1 * (cum_range - (y2_hi - y2_lo))
    new_y2_hi = y2_hi + padding + (1 - frac1) * (cum_range - (y2_hi - y2_lo))
    ax2.set_ylim(new_y2_lo, new_y2_hi)


    ax.legend(handles, labels, loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#d3d3d3')
    plt.tight_layout()

    # After setting xticks and yticks
    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if tick < 0:
            label.set_color('red')

    # Color secondary y-axis negative ticks
    for tick, label in zip(ax2.get_yticks(), ax2.get_yticklabels()):
        if tick < 0:
            label.set_color('red')

    #add_watermark(fig, ax, "mc_app_base/Enerquill_Logo.webp")
    add_watermark(ax, "mc_app_base/Enerquill_Logo.webp")
    st.pyplot(fig)

# Function - Plot P50 Cashflows with NCF Ranges
def plot_P50_cashflow_with_ranges(years_array, revenue_array, capex_array, opex_array, govt_take_array, low_percentile_ncf, high_percentile_ncf, label_low_percentile_ncf, label_high_percentile_ncf, truncate_len=None):

    if truncate_len is not None:
        years_array = years_array[:truncate_len]
        revenue_array = revenue_array[:truncate_len]
        capex_array = capex_array[:truncate_len]
        opex_array = opex_array[:truncate_len]
        govt_take_array = govt_take_array[:truncate_len]
        low_percentile_ncf = low_percentile_ncf[:truncate_len]
        high_percentile_ncf = high_percentile_ncf[:truncate_len]

    start_year = int(years_array[0])
    project_years = int(len(years_array))
    years = np.arange(start_year, start_year + project_years)
    ncf = revenue_array - capex_array - opex_array - govt_take_array
    cumulative_ncf = np.cumsum(ncf)

    fig, ax = plt.subplots(figsize=(14,4), facecolor='none')
    ax.set_facecolor('white')

    # Revenues (positive bars)
    ax.bar(years, revenue_array, label='Revenue', color='#a8ddb5')

    # Negative cash flows: stack from bottom 
    bar1 = ax.bar(years, -capex_array, label='Capex', color='#fbb4b9')
    bar2 = ax.bar(years, -opex_array, bottom=-capex_array, label='Opex', color='#fdae6b')
    bar3 = ax.bar(years, -govt_take_array, bottom=-(capex_array+opex_array), label='Govt Take', color='#bcbddc')
    
    ax.plot(years, ncf, label='P50 NCF', color='#6baed6', linewidth=2)
    ax.plot(years, low_percentile_ncf, label=label_low_percentile_ncf, color='#fb8072', linewidth=2, linestyle='--')
    ax.plot(years, high_percentile_ncf, label=label_high_percentile_ncf, color='#8dd3c7', linewidth=2, linestyle='--')

    ax2 = ax.twinx()
    ax2.plot(years, cumulative_ncf, label='Cumulative NCF', color='#de2d26', linewidth=2)
    ax2.set_ylabel('Cumulative Cash Flow (USD mln)', labelpad=15)
    ax2.set_facecolor('none')

    for spine in ax.spines.values():
        spine.set_color('#d3d3d3')
    for spine in ax2.spines.values():
        spine.set_color('#d3d3d3')

    lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    handles, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    ax.axhline(0, color='black', linewidth=1)

    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Annual Cash Flow (USD mln)')

    # Fix X-axis: integer year ticks only 
   # Fix X-axis: integer year ticks only 
    max_ticks = 20
    n_years = len(years)
    if n_years > max_ticks:
        step = n_years // max_ticks 
    else:
        step = 1

    ax.set_xticks(years[::step])
    ax.set_xticklabels(years[::step])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d')) # format as integer 

    # Fix Y-axis: adding padding so bars don't touch borders
    all_cashflows = np.concatenate([
        revenue_array, 
        -(capex_array + opex_array + govt_take_array),
        ncf, 
        low_percentile_ncf, 
        high_percentile_ncf
    ])
    y_min, y_max = all_cashflows.min(), all_cashflows.max()
    y_range = y_max - y_min 
    padding = y_range * 0.05 # Add 5% padding to the top and bottom 
    ax.set_ylim(y_min - padding, y_max + padding)

    y1_lo, y1_hi = ax.get_ylim()
    y2_lo, y2_hi = ax2.get_ylim()

    frac1 = (0 - y1_lo) / (y1_hi - y1_lo) if (y1_hi - y1_lo) != 0 else 0.5

    cum_range = y2_hi - y2_lo
    padding = cum_range * 0.05  # 5% padding

    # Set new limits keeping 0 anchored
    new_y2_lo = y2_lo - padding - frac1 * (cum_range - (y2_hi - y2_lo))
    new_y2_hi = y2_hi + padding + (1 - frac1) * (cum_range - (y2_hi - y2_lo))
    ax2.set_ylim(new_y2_lo, new_y2_hi)


    ax.legend(handles, labels, loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#d3d3d3')
    plt.tight_layout()

    # After setting xticks and yticks
    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if tick < 0:
            label.set_color('red')

    # Color secondary y-axis negative ticks
    for tick, label in zip(ax2.get_yticks(), ax2.get_yticklabels()):
        if tick < 0:
            label.set_color('red')

    #add_watermark(fig, ax, "Enerquill_Logo.webp")
    add_watermark(ax, "mc_app_base/Enerquill_Logo.webp")
    st.pyplot(fig)

# Function - Plot Tornado Chart
def plot_tornado(df, base_case_NPV):
    df["LowDiff"] = df["Low"] - base_case_NPV
    df["HighDiff"] = df["High"] - base_case_NPV 

    df["Range"] = (df["HighDiff"] - df["LowDiff"]).abs()
    df = df.sort_values("Range", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 2), facecolor='none')
    ax.set_facecolor('white')

    y_pos = np.arange(len(df))

    max_abs = max(abs(df["LowDiff"].min()), abs(df["HighDiff"].max())) * 1.3
    ax.set_xlim(-max_abs, max_abs)

    tick_values = ax.get_xticks()
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{t + base_case_NPV:.0f}" for t in tick_values])

    for i, row in df.iterrows():
        ax.barh(i, row["LowDiff"], color="#fbb4b9" if row["LowDiff"] < 0 else "#a8ddb5", align="center", left=0)
        ax.barh(i, row["HighDiff"], color="#fbb4b9" if row["HighDiff"] < 0 else "#a8ddb5", align="center", left=0)

        ax.text(row["LowDiff"], i, f'{row["Low"]:.1f}', va='center',
                ha='right' if row["LowDiff"] < 0 else 'left', fontsize=plt.rcParams['ytick.labelsize'],
                color='red' if row["Low"] < 0 else 'black')
        ax.text(row["HighDiff"], i, f'{row["High"]:.1f}', va='center',
                ha='left' if row["HighDiff"] > 0 else 'right', fontsize=plt.rcParams['ytick.labelsize'],
                color='red' if row["High"] < 0 else 'black')

    ax.axvline(0, color="black", linewidth=0.5)

    ax.annotate(
        f'P50 NPV (USD mln)\n{base_case_NPV:.1f}',
        xy=(0, 1.02), xycoords=("data", "axes fraction"),  # x at 0 (data coords), y above plot
        ha='center', va='bottom',
        fontsize=plt.rcParams['axes.labelsize'],
        fontweight='bold',
        color='black'
    )


    if base_case_NPV < 0:
        # Get the y position for the second line
        ax.annotate(
            f'{base_case_NPV:.1f}',
            xy=(0, 1.02), xycoords=("data", "axes fraction"),
            ha='center', va='bottom',
            fontsize=plt.rcParams['axes.labelsize'],
            fontweight='bold',
            color='red'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Variable"])
    ax.invert_yaxis()  # largest range at top

    ax.set_xlabel("NPV (USD mln)")

    for spine in ax.spines.values():
        spine.set_color('#d3d3d3')
        spine.set_linewidth(0.5)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='#d3d3d3')

    for tick_val, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label_val = tick_val + base_case_NPV  # this is what’s shown
        tick_label.set_color("red" if label_val < 0 else "black")

    fig.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.15)
    plt.tight_layout()
    add_watermark(ax, "mc_app_base/Enerquill_Logo.webp", 1, 0.1)

    st.pyplot(fig)

# --------------------------------------------------------------
# Configure page for wide layout and title
# --------------------------------------------------------------
st.set_page_config("Monte Carlo simulation", layout="wide", initial_sidebar_state="expanded")
st.title("Monte Carlo Simulator")
st.caption("*This simulator has been designed to demonstrate functionality and simplified assumptions have been used. Outputs from this simulator should not be used for decision making purposes.*")
st.write("")

st.markdown(
    """
    <style>
    .ag-header-cell-label {
        justify-content: center;
    }
    /* Remove white background and border from the grid container */
    .ag-theme-streamlit {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0d6efd, #0a58ca); /* Primary blue */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none; /* Remove border */
        height: 3em;
        width: 100%;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #0a58ca, #0d6efd);
        transform: translateY(-2px);
        box-shadow: 0px 6px 12px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

mc_cell_style_jscode = JsCode("""
        function(params) {
        return {
        backgroundColor: '#f0f0f0',
        color: (params.value !== null && params.value < 0) ? 'red' : 'black',
        verticalAlign: 'middle'
    };
}
""")

plt.rcParams.update({
    'font.size': 10,        # Base font size for everything
    'axes.labelsize': 10,  # Font size for X/Y axis labels
    'axes.titlesize': 10,  # Font size for titles (even though you remove in tornado)
    'xtick.labelsize': 9,  # Font size for X-axis tick labels
    'ytick.labelsize': 9,  # Font size for Y-axis tick labels
    'legend.fontsize': 9   # Font size for legend
})

st.markdown(
    """
    <style>
    /* Keep your preferred font globally */
    html, body, [class*="css"]  {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    /* Ensure Streamlit sidebar collapse button displays correctly */
    [data-testid="collapsedControl"] svg {
        display: inline !important;  /* show the arrow icon */
    }
    [data-testid="collapsedControl"]::before {
        content: '>>';  /* fallback if icon fails */
        font-family: 'Inter', system-ui, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)



#------------------------------
# File upload and instructions
#------------------------------
default_file_path = "mc_app_base/Default.xlsx"

instructions_col, _, excel_input_col = st.columns([3, 0.2, 1.5])
with instructions_col:
    font_size("Instructions", 18)
    st.markdown("""
    - This free demo has limited functionality. Additional functions can be added upon request for paid versions. 
    - If uploading your own CSV or Excel file, the free demo: 
        - Requires row 1 includes headers for time-series (e.g. Years)
        - Requires Column A includes **Price**, **Capex**, **Opex** and **Production** as descriptors. 
        - Allows multiple rows of Capex and Opex can be included, but they will be aggregated and treated as a single Capex / Opex line item
        - Allows only one Price line
    - Cells are editable after upload but the ability to paste from a clipboard is not activated
    - **"Download Table as Excel"** provides the default data in Excel format, which can be repopulated with your data            
                """)

is_default_file = False

with excel_input_col:
    
    font_size("Optional: Upload Your File", 18)
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xls", "xlsx"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Load uploaded file depending on type
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, header=None)
            else:
                df = pd.read_excel(uploaded_file, header=None)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = None

        if df is not None:
            col_a = df.iloc[:,0].astype(str).str.lower() # set first column values as lower-case strings 
            required_words = ["production", "capex", "opex", "price"]
            missing_words = [word for word in required_words if not col_a.str.contains(word).any()]

            if missing_words:
                st.error(f"Missing required data: {','.join(missing_words)}")
            else:
                row_indices = [col_a[col_a.str.contains(word)].index[0] for word in required_words]
                
                for idx in row_indices:
                    row_data = df.iloc[idx, 1:]
                    df.iloc[idx, 1:] = pd.to_numeric(row_data, errors="coerce").fillna(0).astype(float)
                st.success("✅ Uploaded File with Sufficient Data")
                
                
    else: 
        # Load default file
        try:
            df = pd.read_excel(default_file_path, header=None)
            is_default_file = True
            st.info("ℹ️ Using default data")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = None            

  
st.markdown("---")

st.write(df)

#------------------------------
# Sidebar for Fiscal Inputs
#------------------------------

with st.sidebar: 
    st.header("Fiscal Selection")
    st.caption("This demo includes a small selection of fiscal terms but corporate tax calculations have been deactivated.")

    st.write("**Select Fiscal Regime**")
    fiscal_regime = st.radio(
        label="Fiscal Regime",
        options=["Production Sharing Contract", "Concession"],
        index=0, # Default to PSC
        horizontal=False,
        label_visibility="collapsed"
    )

    # This section if PSC is selected
    if fiscal_regime == "Production Sharing Contract":
        st.write("**PSC Parameters**")

        psc_cr_label_col, psc_cr_input_col = st.columns([1.5, 1.5])
        with psc_cr_label_col:
            description("Cost Recovery Ceiling (%)")
        
        with psc_cr_input_col:
            psc_cost_recovery_ceiling = st.number_input("Cost Recovery Ceiling(%)", min_value = 0, max_value=100, value=70, step=1, label_visibility="collapsed")

        psc_dep_label_col, psc_dep_input_col = st.columns([1.5, 1.5])
        with psc_dep_label_col:
            description("Depreciation Method")

        with psc_dep_input_col:
            psc_capex_depreciation_method = st.selectbox(
                "Depreciation Method", 
                options=["Straight Line", "Immediate", "Declining Balance"],
                index=1, label_visibility="collapsed"
            )

        # This section if Straight Line depreciation is selected
        if psc_capex_depreciation_method == "Straight Line":
            psc_dep_sl_label_col, psc_dep_sl_input_col = st.columns([1.5, 1.5])
            with psc_dep_sl_label_col:
                    description("Depreciation Years")

            with psc_dep_sl_input_col:
                    psc_dep_sl_years = st.number_input("Depreciation Years", min_value = 2, max_value=50, value=5, step=1, label_visibility="collapsed")

        # This section if Declining Balance depreciation is selected         
        elif psc_capex_depreciation_method == "Declining Balance":
            psc_dep_db_label_col, psc_dep_db_input_col = st.columns([1.5, 1.5])
            with psc_dep_db_label_col:
                description("Declining Balance Rate (%)")
                
            with psc_dep_db_input_col:
                psc_dep_db_rate = st.number_input("Depreciation Rate", min_value = 1, max_value=99, value=20, step=1, label_visibility="collapsed")

        psc_opex_incl_label_col, psc_opex_incl_input_col = st.columns([1.5, 1.5])
        with psc_opex_incl_label_col:
            description("Cost Recover Opex")

        with psc_opex_incl_input_col:
            psc_include_opex = st.radio("Cost Recovery Opex", options=["Yes", "No"], horizontal=True, label_visibility="collapsed")

        st.write("**Profit Split Method**")
        psc_split_method_label_col, psc_split_method_select_col = st.columns([1.5, 1.5])
        with psc_split_method_label_col:
            psc_split_method = st.radio("Profit Oil or R-Factor", options=["Profit Oil", "R-Factor"], horizontal=True, label_visibility="collapsed")

        # This section if Profit Oil split selected: 
        if psc_split_method == "Profit Oil":
            psc_govt_share_label_col, psc_govt_share_input_col = st.columns([1.5, 1.5])
            with psc_govt_share_label_col:
                description("Govt Share: Profit Oil (%)")

            with psc_govt_share_input_col:
                psc_profit_oil_gov = st.slider("Profit Oil Share to Government (%)", min_value = 0, max_value=100, value=60, step=1, label_visibility="collapsed")

        # This section if R-Factor is selected: 
        elif psc_split_method == "R-Factor":

            rfactor_basis_label_col, rfactor_basis_input_col = st.columns([1, 2])
            with rfactor_basis_label_col:
                description("R-Factor Basis")

            with rfactor_basis_input_col:
                rfactor_basis = r_factor_method = st.selectbox("Select R-Factor Basis", options=["Cuml Revenues / Cuml Costs", "Cuml Net Earnings / Cuml Costs"], index=0, label_visibility="collapsed")

            rfactor_tranche_label_col, rfactor_tranche_input_col, _ = st.columns([1, 1, 1])
            with rfactor_tranche_label_col:
                description("No. of Tranches")

            with rfactor_tranche_input_col:
                rfactor_num_tranches = st.number_input("Number of R-Factor Tranches", min_value=1, max_value=5, value=3, step=1, label_visibility="collapsed")  

            _, rfactor_header_col, rfactor_gov_share_col, = st.columns([1, 1, 1])
            with rfactor_header_col:
                render_header("R-Factor", rfactor_header_col)

            with rfactor_gov_share_col:
                render_header("Govt Share (%)", rfactor_gov_share_col)

            rfactors = []
            rfactor_gov_shares = []

            for i in range(rfactor_num_tranches):
                tranche_label = f"Tranche {i + 1}"
                rfactor_tranche_label_col, rfactor_tranche_factor_col, rfactor_tranche_gov_col = st.columns([1, 1, 1])

                with rfactor_tranche_label_col:
                    description(tranche_label)

                with rfactor_tranche_factor_col:
                    rfactor_factor = st.number_input(f"{tranche_label} R-Factor", min_value = 0.1, value=1.0 + i * 0.5, step=0.1, label_visibility="collapsed")
                    rfactors.append(rfactor_factor)

                with rfactor_tranche_gov_col:
                    rfactor_gov = st.number_input(f"{tranche_label} Govt Share (%)", min_value=0.0, max_value=100.0, value=50.0 + i*10, step=1.0, label_visibility="collapsed")
                    rfactor_gov_shares.append(rfactor_gov)

    # This section if concession is selected 
    if fiscal_regime == "Concession":
        st.write("**Royalty Parameters**")

        royalty_rate_label_col, royalty_rate_input_col = st.columns([1.5, 1.5])
        with royalty_rate_label_col:
            description("Royalty Rate (%)")

        royalty_label_col, royalty_input_col = st.columns([1.5, 1.5])
        with royalty_label_col:
            description("Royalty Basis")

        with royalty_input_col:
            royalty_basis = st.selectbox("Royalty Basis", options=["Revenues", "Earnings"], index=0, label_visibility="collapsed")

        with royalty_rate_input_col:
            royalty_rate = st.number_input("Royalty Rate", min_value=0, max_value=95, value=20, step=1, label_visibility="collapsed")

        if royalty_basis == "Earnings":
            royalty_capex_dep_label_col, royalty_capex_dep_input_col = st.columns([1.5, 1.5])
            with royalty_capex_dep_label_col:
                description("Depreciation Method")      

            with royalty_capex_dep_input_col:
                royalty_capex_depreciation_method = st.selectbox(
                    "Capex Depreciation Method", 
                    options=["Straight Line", "Immediate", "Declining Balance"],
                    index=0, label_visibility="collapsed"
            )

            # This section if Straight Line depreciation is selected
            if royalty_capex_depreciation_method == "Straight Line":
                royalty_dep_sl_label_col, royalty_dep_sl_input_col = st.columns([1.5, 1.5])
                with royalty_dep_sl_label_col:
                    description("Depreciation Years")

                with royalty_dep_sl_input_col:
                    royalty_dep_sl_years = st.number_input("Depreciation Years", min_value = 2, max_value=50, value=5, step=1, label_visibility="collapsed")

            # This section if Declining Balance depreciation is selected         
            elif royalty_capex_depreciation_method == "Declining Balance":
                royalty_dep_db_label_col, royalty_dep_db_input_col = st.columns([1.5, 1.5])
                with royalty_dep_db_label_col:
                    description("Declining Balance Rate (%)")
                
                with royalty_dep_db_input_col:
                    royalty_dep_db_rate = st.number_input("Depreciation Rate", min_value = 1, max_value=99, value=20, step=1, label_visibility="collapsed")

    st.write("---")
    st.header("Production Options")
    st.caption("**Enable Production Sensitivities** overwrites the production profile in the Excel file with a dynamic production function and enables sensitivities.")

    if uploaded_file is not None and df is not None:
        if not missing_words:
            prod_enabled = st.checkbox("Enable Dynamic Production & Sensitivities", value=False)
        
    else:
        prod_enabled = st.checkbox("Enable Dynamic Production & Sensitivities", value=True)

    st.write("")

    if prod_enabled:

        _, prodsens_header_p10, prodsens_header_p50, prodsens_header_p90 = st.columns([2, 1, 1, 1])
        render_header("P10 (Low)", prodsens_header_p10)
        render_header("P50 (Base)", prodsens_header_p50)
        render_header("P90 (High)", prodsens_header_p90)

        st.write("")

        resource_label_col, resource_p10_col, resource_p50_col, resource_p90_col = st.columns([2, 1, 1, 1])
        with resource_label_col:
            description("Resource (MMbbl)")

        with resource_p10_col:
            resource_p10 = st.number_input("P10 Resource", min_value=0, step=1, value=70, label_visibility="collapsed")

        with resource_p50_col:
            resource_p50 = st.number_input("P50 Resource", min_value=0, step=1, value=100, label_visibility="collapsed")    

        with resource_p90_col:
            resource_p90 = st.number_input("P90 Resource", min_value=0, step=1, value=200, label_visibility="collapsed")
        
        resource_initial_prod_label_col, resource_initial_prod_col_p50 = st.columns([2, 3.1])
        with resource_initial_prod_label_col:
            description("Initial Prod (bbl / day)")
    
        with resource_initial_prod_col_p50:
            initial_prod_p50 = st.number_input("Initial Prod (bbl/d)", min_value = 0, value=40000, label_visibility="collapsed")

        resource_decline_label_col, resource_decline_input_col_p50 = st.columns([2, 3.1])
        with resource_decline_label_col:
            description("Decline Rate (% / day)")

        with resource_decline_input_col_p50:
            resource_decline_rate_p50 = st.number_input("Decline Rate P50", min_value=0.00, max_value=100.00, value=0.02, label_visibility="collapsed")    

        capacity_label_col,capacity_input_col = st.columns([2, 3.1])
        with capacity_label_col:
            description("Capacity (bbl / day)")
        
        with capacity_input_col:
            facility_capacity = st.number_input("Capacity", min_value=0, value=35000, label_visibility="collapsed")

        prod_model_label_col, prod_model_input_col = st.columns([2, 3.1])
        with prod_model_label_col:
            description("Decline Model")

        with prod_model_input_col:
            decline_model = st.selectbox( "Decline Model", options=["Exponential", "Harmonic", "Hyperbolic"], label_visibility="collapsed")

        # Show 'b' only if model is Hyperbolic 
        if decline_model == "Hyperbolic":
            decline_b_label_col, decline_b_input_col= st.columns([2, 3.1])

            with decline_b_label_col:
                description("b (Exponent)")
            
            with decline_b_input_col:
                b_value =  st.number_input("Hyerbolic b", min_value = 0.0, max_value=2.0, value=0.5, step=0.01, label_visibility="collapsed")

    else:
        st.write("")

    st.write("---")
    st.header("Cost Sensitivities")
    st.caption("By default, this demo runs cost sensitivities. Base scalar = 1.")

    _, cost_sens_header_p10, cost_sens_header_p90 = st.columns([2, 1.5, 1.5])
    with cost_sens_header_p10:
        render_header("P10 (Low)", cost_sens_header_p10)

    with cost_sens_header_p90:
        render_header("P90 (High)", cost_sens_header_p90)

    st.write("")
    capex_scalar_label_col, capex_p10_scalar_col, capex_p90_scalar_col = st.columns([2, 1.5, 1.5])
    with capex_scalar_label_col:
        description("Capex Scalar")

    with capex_p10_scalar_col:
        capex_p10_scalar = st.number_input("P10_Capex", min_value = 0.0, value=0.8, label_visibility="collapsed")

    with capex_p90_scalar_col:
        capex_p90_scalar = st.number_input("P90_Capex", min_value = 0.0, value=1.25, label_visibility="collapsed")

    opex_scalar_label_col, opex_p10_scalar_col, opex_p90_scalar_col = st.columns([2, 1.5, 1.5])
    with opex_scalar_label_col:
        description("Opex Scalar")

    with opex_p10_scalar_col:
        opex_p10_scalar = st.number_input("P10_Opex", min_value = 0.0, value=0.9, label_visibility="collapsed")

    with opex_p90_scalar_col:
        opex_p90_scalar = st.number_input("P90_Opex", min_value = 0.0, value=1.15, label_visibility="collapsed")

    st.write("---")

    st.header("Output Selections")
    st.caption("When enabled, **Run Monte Carlo** generates probabilistic economic metrics. Otherwise, the outputs only include P50 economic metrics.")
    mc_label_space = 2
    mc_label_toggle = 1.5

    output_p50cf_label_col, output_p50cf_select_col= st.columns([mc_label_space, mc_label_toggle])
    with output_p50cf_label_col:
        description("Show P50 Cashflow Chart")

    with output_p50cf_select_col:
        show_p50cf_chart = st_toggle_switch(label="", key="p50cf_toggle", default_value=True, label_after=True, inactive_color = '#55555', active_color="#21ba45", track_color="#29a745")

    output_tornado_label_col, output_tornado_select_col= st.columns([mc_label_space, mc_label_toggle])
    with output_tornado_label_col:
        description("Show Tornado Chart")

    with output_tornado_select_col:
        show_tornado_chart = st_toggle_switch(label="", key="tornado_toggle", default_value=True, label_after=True, inactive_color = '#55555', active_color="#21ba45", track_color="#29a745")

    output_mc_label_col, output_mc_select_col= st.columns([mc_label_space, mc_label_toggle])
    with output_mc_label_col:
        description("Run Monte Carlo")

    with output_mc_select_col:
        run_monte_carlo = st_toggle_switch(label="", key="monte_carlo_toggle", default_value=False, label_after=True, inactive_color = '#55555', active_color="#21ba45", track_color="#29a745")

    if run_monte_carlo:
        st.write("")
        st.write("**Monte Carlo Settings**")
        output_volatility_label_col, output_volatility_toggle_col =  st.columns([mc_label_space, mc_label_toggle])
        with output_volatility_label_col:
            description("Include Price Volatility")

        with output_volatility_toggle_col:
            run_price_volatility = st_toggle_switch(label="", key="price_volatility_toggle", default_value=False, label_after=True, inactive_color = '#55555', active_color="#21ba45", track_color="#29a745")

        
        output_sims_label_col, output_sims_input_col = st.columns([mc_label_space, mc_label_toggle])
        with output_sims_label_col:
            description("Number of Simulations")

        with output_sims_input_col:
            n_simulations = st.number_input(label="Number of Simulations", min_value=1, max_value=10000, value=5000, step=1, label_visibility="collapsed")    
        
        st.write("")
        st.write("**Monte Carlo Output Options**")

        
        s_curve_type = st.radio(
            label="Dist_Curve",
            options=["S-Curve", "Probability of Exceedance"],
            index=0, # Default to PSC
            horizontal=True,
            label_visibility="collapsed")

        output_NPVdist_label_col, output_NPVdist_select_col = st.columns([mc_label_space, mc_label_toggle])
        with output_NPVdist_label_col:
            description("NPV Distribution")

        with output_NPVdist_select_col:
            show_NPV_dist = st_toggle_switch(label="", key="NPV_dist_toggle", default_value=True, label_after=True, inactive_color = '#55555', active_color="#21ba45", track_color="#29a745")

        output_IRRdist_label_col, output_IRRdist_toggle_col = st.columns([mc_label_space, mc_label_toggle])
        with output_IRRdist_label_col:
            description("IRR Distribution")

        with output_IRRdist_toggle_col:
            show_IRR_dist = st_toggle_switch(label="", key="IRR_dist_toggle", default_value=False, label_after=True, inactive_color = '#55555', active_color="#21ba45", track_color="#29a745")
        
        percentiles_selected = []
        if show_NPV_dist or show_IRR_dist:
            Percentile_label_col, Percentile_select_col = st.columns([1, 2])
            with Percentile_label_col:
                description("Show Percentiles")
            
            with Percentile_select_col:
                percentiles_selected = st.multiselect("Choose Percentiles", options=["P10", "P25", "P50", "P75", "P90"], default=["P10","P50","P90"], max_selections=3, label_visibility="collapsed")

        
        st.write("")
        _, run_mc_col,_ = st.columns([0.5, 2, 0.5])
        with run_mc_col:
            run_mc_button = st.button("▶ Run Monte Carlo Simulation", type="primary")
            progress_placeholder = st.empty()
            time_placeholder = st.empty()
      
# -----------------------------------------
# Generate Dynamic P50 Profile for Default
# -----------------------------------------

# Generate first the P90 resource profile 
if df is not None and prod_enabled:
    df = df.dropna(axis=1, how='all')
    df.iloc[:,0] = df.iloc[:, 0].astype(str)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64']).columns

    # Convert only these numeric columns to float
    df[numeric_cols] = df[numeric_cols].astype(float)

    high_resource_prod, _ , _, _= constrained_decline(initial_prod_p50, resource_decline_rate_p50, resource_p90, facility_capacity, decline_model, b=b_value if decline_model == "Hyperbolic" else 0.5, max_years=50)

    # Calculate the length of production years for the P90 resource profile 
    high_resource_prod_years = sum(1 for x in high_resource_prod if x > 0)

    # Identify the production row from the uploaded data 
    prod_rows_mask = df.iloc[:, 0].str.lower().str.contains("production")
    prod_rows_idx = df.index[prod_rows_mask].tolist()

    if len(prod_rows_idx) == 0:
        raise ValueError("No production rows found in the uploaded file")

    prod_data = df.iloc[prod_rows_idx, 2:].fillna(0)
    summed_production = prod_data.sum(axis=0)

    leading_zeros = 0
    for val in summed_production:
        if val == 0:
            leading_zeros += 1
        else:
            break 

    high_resource_list = high_resource_prod.tolist()
    while high_resource_list and high_resource_list[-1] == 0:
        high_resource_list.pop()

    Profile_Production_P90 = [0.0] * leading_zeros + high_resource_list + [0.0]
    time_series_len = len(Profile_Production_P90)

    Profile_Production_P50 = gen_production_profile(initial_prod_p50, resource_decline_rate_p50, resource_p50, facility_capacity, Profile_Production_P90, decline_model, b=b_value if decline_model == "Hyperbolic" else 0.5)
    Profile_Production_P50 = np.array(Profile_Production_P50, dtype=float)

    required_cols = 2 + len(Profile_Production_P50)
    current_cols = df.shape[1]
    extra_cols = required_cols - current_cols

    if extra_cols > 0: 
        for i in range(extra_cols):
            new_col_name = f"{current_cols + i}"
            df[new_col_name] = 0

    for idx in prod_rows_idx:
        df.iloc[idx, 2:required_cols] = Profile_Production_P50
    
    # -----------------------------------------
    # Adjust length of other variables
    # -----------------------------------------

    target_len = len(Profile_Production_P50)
    data_cols = df.columns[2:]

    # Update years row based on the length of the P50 profile 
    years_row = df.iloc[0, 2:]

    # Find last valid year from Excel upload
    last_upload_year = int(years_row[years_row !=0].iloc[-1])

    # Replace trailing zeros with consecutive years
    for i in range(len(years_row)):
        if years_row.iloc[i] == 0:
            years_row.iloc[i] = last_upload_year + 1
            last_upload_year +=1

    df.iloc[0, 2:] = years_row    
    years_row = [int(year) for year in years_row]

    # Identify the capex row(s) from the uploaded data 
    capex_rows_mask = df.iloc[:, 0].str.lower().str.contains("capex")
    capex_rows_idx = df.index[capex_rows_mask].tolist()
    capex_rows = df.iloc[capex_rows_idx, 2:]

    for i, idx in enumerate(capex_rows_idx):
        row = capex_rows.iloc[i].tolist()

        if len(row) > target_len:
            row = row[:target_len]

        df.iloc[idx, 2:2+target_len] = row
        
        if df.shape[1] - 2 > target_len:
            df.iloc[idx, 2+target_len:] = 0

    # Identify the opex row(s) from the uploaded data 
    opex_rows_mask = df.iloc[:, 0].str.lower().str.contains("opex")
    opex_rows_idx = df.index[opex_rows_mask].tolist()
    opex_rows = df.iloc[opex_rows_idx, 2:]

    for i, idx in enumerate(opex_rows_idx):
        row = opex_rows.iloc[i].tolist()
        if len(row) > target_len:
            row = row[:target_len]

        df.iloc[idx, 2:2+target_len] = row

        last_opex_val = next((val for val in reversed(row) if val != 0), 0)

        if df.shape[1] > current_cols:
            df.iloc[idx, current_cols:] = last_opex_val

    # Identify the price row from the uploaded data 
    price_row_mask = df.iloc[:, 0].str.lower().str.contains("price")
    price_row_idx = df.index[price_row_mask].tolist()[0]
    price_row = df.iloc[price_row_idx, 2:].tolist()
    
    if len(price_row) > target_len:
        price_row = price_row[:target_len]
    
    last_price_val = next((val for val in reversed(price_row) if val != 0), 0)

    df.iloc[price_row_idx, 2:2+target_len] = price_row 

    if df.shape[1] > current_cols:
        df.iloc[price_row_idx, current_cols:] = last_price_val

    df = df.iloc[:, :2 + target_len]  

numeric_cols = df.select_dtypes(include=['int64', 'Int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].astype(float)

df.columns = df.iloc[0]
df.columns = [int(col) if str(col).replace('.0','').isdigit() else col for col in df.columns]
df = df[1:].reset_index(drop=True)


# ----------------------------------------------------
# Display DataFrame using AgGrid
# ----------------------------------------------------

first_col = df.columns[0]
second_col = df.columns[1] if len(df.columns) > 1 else None 


if df is not None:
    df.columns = [str(col) for col in df.columns]

    # Build AgGrid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True, resizable=True)

    gb.configure_column(first_col, cellStyle={'textAlign':'left'}, width=350, headerCheckboxSelection=False, suppressMenu=True, sortable=False, filter=False, headerComponentParams={'menuIcon': 'none'})

    if second_col:
        gb.configure_column(second_col, width=280, headerCheckboxSelection=False, suppressMenu=True, sortable=False, filter=False, headerComponentParams={'menuIcon': 'none'})

    for col in df.columns[2:]:
        gb.configure_columns(col, type=["numericColumn"], valueFormatter="x.toFixed(2)", width=88, suppressSizeToFit=True, headerCheckboxSelection=False, suppressMenu=True, sortable=False, filter=False, headerComponentParams={'menuIcon': 'none'})

    # Calculate grid height dynamically
    row_height = 33  # Adjust if your rows are taller
    header_height = 40
    num_rows = len(df)
    grid_height = header_height + (row_height * num_rows)
    grid_options = gb.build()

    # Display the AgGrid
    font_size("📊 P50 Cash Flow Inputs", 18)
    st.caption("Production data is replaced by dynamic production generator if **Enable Dynamic Production & Sensitivities** is checked.")
    st.caption("Demo Limitations: Multiple rows of uploaded data within the same categories (e.g. Capex, Opex, Production) are aggregated for sensitivity analysis, maximum P90 production years are limited to 50 and cash flow lengths are automatically adjusted to fit P90 production profiles when production sensitivities are enabled.")
    AgGrid(
        df,
        gridOptions=grid_options,
        height=grid_height,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=True,
        theme="streamlit"
    )
else:
    st.info("Upload a file or provide a valid default file to view the data.")

excel_data = to_excel(df)
st.download_button(
    label="📥 Download Table as Excel",
    data=excel_data,
    file_name="Cash_Flow_Inputs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.markdown("---")

# --------------------------------------------------
# Begin Calculations in Python 
# --------------------------------------------------

# -------------------------------------
# Extract the time series inputs
# -------------------------------------

# Extract the price row, excluding the description and unit 
price_row_idx = df.iloc[:,0].str.lower().str.contains("price").idxmax()
Profile_price_base = df.iloc[price_row_idx,2:]

# Extract the production row(s), excluding the description and unit 
prod_rows_mask = df.iloc[:, 0].str.lower().str.contains("production")
prod_rows = df.loc[prod_rows_mask, df.columns[2:]]
Profile_production_base = prod_rows.sum(axis=0)

# Extract the capex row(s), excluding the description and unit 
capex_rows_mask = df.iloc[:, 0].str.lower().str.contains("capex")
capex_rows = df.loc[capex_rows_mask, df.columns[2:]]
Profile_capex_base = capex_rows.sum(axis=0)

# Extract the opex row(s), excluding the description and unit
opex_rows_mask = df.iloc[:, 0].str.lower().str.contains("opex")
opex_rows = df.loc[opex_rows_mask, df.columns[2:]]
Profile_opex_base = opex_rows.sum(axis=0)

# Extract the header, excluding the description and unit
Profile_years = [col for col in df.columns if col.isdigit()]

# -----------------------------------------------------
# Apply Economic Cut-off to Calculations
# -----------------------------------------------------
Profile_production_base = np.array(Profile_production_base, dtype=float)
Profile_price_base = np.array(Profile_price_base, dtype=float)
Profile_revenue_base = Profile_production_base * Profile_price_base 

# Identify Economic Cut-off post Revenue-based Royalty
if fiscal_regime == "Concession" and royalty_basis == "Revenues":
    Rate_royalty_revenue = royalty_rate/100
else:
    Rate_royalty_revenue = 0

Profile_capex_base = np.array(Profile_capex_base, dtype=float)
Profile_opex_base = np.array(Profile_opex_base, dtype=float)

Profile_pre_tax_cf_base = Profile_revenue_base * (1 - Rate_royalty_revenue) - Profile_capex_base - Profile_opex_base
Profile_pre_tax_cf_cum_base = Profile_pre_tax_cf_base.cumsum()

# Find the position of the max cumulative cash flow
max_pos = Profile_pre_tax_cf_cum_base.argmax()
max_val = Profile_pre_tax_cf_cum_base[max_pos]



Profile_years_numeric = pd.Series(pd.to_numeric(df.columns[2:], errors='coerce'))
valid_mask = Profile_years_numeric.notna()
Profile_years_numeric = Profile_years_numeric[valid_mask].astype(int).to_numpy()

max_year = Profile_years_numeric[max_pos]  # now it's an int
econ_cutoff_flag = (Profile_years_numeric <= max_year).astype(float)
chart_length = int(sum(econ_cutoff_flag))



# -----------------------------------------------------
# Display Pre-Tax Calculations
# -----------------------------------------------------

font_size("📈 Economic Cut-off", 18)
st.caption("Economic cut-off determines the project life for subsequent calculations. This snippet illustrates how interim calculations can be displayed in spreadsheet format to address issues around calculation transparency.")

Profile_royalty = Rate_royalty_revenue * Profile_revenue_base

Profile_pre_tax_cf_display = Profile_revenue_base - Profile_royalty - Profile_capex_base - Profile_opex_base
Profile_pre_tax_cf_cum_display = Profile_pre_tax_cf_display.cumsum()

pre_tax_cf_rows = [
    ("Production", "MMbbl / yr", Profile_production_base),
    ("Price", "USD / bbl", Profile_price_base), 
    ("Revenue", "USD mln", Profile_revenue_base),
    ("", "", np.array([""] * len(Profile_years))),
    ("Revenue Royalty", "USD mln", Profile_royalty),
    ("Capex", "USD mln", Profile_capex_base), 
    ("Opex", "USD mln", Profile_opex_base),
    ("Pre-Tax Cash Flow", "USD mln", Profile_pre_tax_cf_display),
    ("Cuml Pre-Tax Cash Flow", "USD mln", Profile_pre_tax_cf_cum_display),
    ("Project Flag", " ", econ_cutoff_flag)
]

pre_tax_cf_data_dict = {
    "Description": [desc for desc, _, _ in pre_tax_cf_rows], 
    "Units": [unit for _, unit, _ in pre_tax_cf_rows]
}

for i, year in enumerate(Profile_years):
    pre_tax_cf_data_dict[str(year)] = [
        # 🔧 check type first, then index
        (data.iloc[i] if isinstance(data, pd.Series) else data[i]) if 
        ((data.iloc[i] if isinstance(data, pd.Series) else data[i]) != "") else ""
        for _, _, data in pre_tax_cf_rows
    ]
    

df_pre_tax_cf_display = pd.DataFrame(pre_tax_cf_data_dict)

for col in df_pre_tax_cf_display.columns[2:]:  # skip Description and Units
    df_pre_tax_cf_display[col] = df_pre_tax_cf_display[col].apply(
        lambda x: None if (x is None or pd.isna(x) or x == "") else f"{x:.2f}"
    )

pre_tax_cf_first_col = df_pre_tax_cf_display.columns[0]
pre_tax_cf_gb = GridOptionsBuilder.from_dataframe(df_pre_tax_cf_display)
pre_tax_cf_gb.configure_default_column(editable=False, resizable=True)


pre_tax_cf_gb.configure_column(pre_tax_cf_first_col, cellStyle={'textAlign':'left'}, width=400, headerCheckboxSelection=False, suppressMenu=True, sortable=False, filter=False, headerComponentParams={'menuIcon':'none'})

negative_red_style = """
function(params) {
    if (params.value < 0) {
        return {'color': 'red'};
    }
    return null;
}
"""

for col in df_pre_tax_cf_display.columns[1:]:
    pre_tax_cf_gb.configure_column(col, type=["numericColumn"],
        headerCheckboxSelection=False,
        suppressMenu=True,
        sortable=False,
        filter=False,
        headerComponentParams={'menuIcon': 'none'},
        cellStyle=JsCode(negative_red_style), 
        width=88
        )
    
    
pre_tax_cf_gb.configure_grid_options(domLayout='autoHeight')
pre_tax_cf_grid_options = pre_tax_cf_gb.build()

AgGrid(df_pre_tax_cf_display, gridOptions=pre_tax_cf_grid_options, fit_columns_on_grid_load=False, allow_unsafe_jscode=True, theme="streamlit", height=350)

st.markdown("---")

# -----------------------------------------------------
# Set discount rate parameters
# -----------------------------------------------------

discount_rate = 0.1
project_years = len(Profile_years)
discount_years = np.arange(project_years)
discount_factor = (1 + discount_rate) ** (-discount_years)

# -----------------------------------------------------
# Set the time series length based on economic cut-off
# -----------------------------------------------------

min_len = min(len(Profile_production_base), len(econ_cutoff_flag))
econ_cutoff_flag = econ_cutoff_flag[:min_len]
Profile_production_base = Profile_production_base[:min_len]
Profile_opex_base = Profile_opex_base[:min_len]
Profile_capex_base = Profile_capex_base[:min_len]
Profile_price_base = Profile_price_base[:min_len]

Production_P50 = Profile_production_base * econ_cutoff_flag
Opex_P50 = Profile_opex_base * econ_cutoff_flag
Capex_P50 = Profile_capex_base * econ_cutoff_flag
Revenue_P50 = Production_P50 * Profile_price_base


# -----------------------------------------------------
# Calculate P50 Metrics
# -----------------------------------------------------

if fiscal_regime == "Production Sharing Contract":
    P50_NCF_Post_fiscal = calc_PSC(Capex_P50, Production_P50, Profile_price_base, Opex_P50, psc_cost_recovery_ceiling)
else:
    P50_NCF_Post_fiscal = calc_concession(Capex_P50, Production_P50, Profile_price_base, Opex_P50, royalty_rate)

P50_Govt_Take = Revenue_P50 - Capex_P50 - Opex_P50 - P50_NCF_Post_fiscal
DCF_P50 = P50_NCF_Post_fiscal * discount_factor
NPV_P50 = np.sum(DCF_P50)

try: 
    irr_value = npf.irr(P50_NCF_Post_fiscal)
    IRR_P50 = irr_value * 100 if not np.isnan(irr_value) else None 
except Exception:
    IRR_P50 = None

P50_NCF_Post_fiscal_cum = P50_NCF_Post_fiscal.cumsum()
Max_Exposure_P50 = np.min(P50_NCF_Post_fiscal_cum)
Payback_P50 = calc_payback(P50_NCF_Post_fiscal)

base_case_results_data = {
    "Base Case Metrics": [f"NPV @ {discount_rate * 100:.0f}% (USD mln)", 
               "IRR (%)",
               "Max Exposure (USD mln)",
               "Payback (Years)"], 
    "Value": [
        NPV_P50, 
        IRR_P50, 
        Max_Exposure_P50, 
        Payback_P50
    ]
}

df_base_case_results = pd.DataFrame(base_case_results_data)

gb = GridOptionsBuilder.from_dataframe(df_base_case_results)
gb.configure_default_column(editable=False, resizable=True, sortable=False, filter=False)

# Use JsCode wrapper for JavaScript
cell_style_jscode = JsCode("""
function(params) {
    return {
        backgroundColor: '#f0f0f0',
        color: params.value < 0 ? 'red' : 'black',
        textAlign: 'center',       // horizontal alignment
        verticalAlign: 'middle'    // vertical alignment
    };
}
""")

value_formatter = JsCode("""
function(params) {
if (params.value === null || params.value === undefined) {
    return "";  // or "0.00" if you prefer
}
return params.value.toFixed(2);
}
""")

# Apply cellStyle to column
font_size("💰 P50 Project Economics", 18)
st.caption("Project economics are truncated by economic cut-off. Where cumulative cash flow is always < 0, data is set to 0.")
gb.configure_column(
    "Base Case Metrics",
    editable=False,
    resizable=True,
    sortable=False,
    filter=False,
    suppressMenu=True,
    headerCheckboxSelection=False,
    headerComponentParams={'menuIcon':'none'}
)

gb.configure_column(
    "Value",
    headerName="",
    editable=False,
    resizable=True,
    sortable=False,
    filter=False,
    cellStyle=cell_style_jscode,
    valueFormatter=value_formatter,
    suppressMenu=True,
    headerCheckboxSelection=False,
    type=["numericColumn"],
    headerComponentParams={'menuIcon':'none'}
)

grid_options = gb.build()

# Display grid
AgGrid(
    df_base_case_results,
    gridOptions=grid_options,
    height=160,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    theme="streamlit"
)

st.markdown("---")

# -----------------------------------------------------
# Calculate Monte Carlo 
# -----------------------------------------------------

if run_monte_carlo and run_mc_button:
    start_time = time.time()
    progress_bar = progress_placeholder.progress(0)
    mc_results = []
    
    capex_a, capex_c, capex_b = fit_triangular_from_p10_p50_p90(capex_p10_scalar, 1, capex_p90_scalar)
    opex_a, opex_c, opex_b = fit_triangular_from_p10_p50_p90(opex_p10_scalar, 1, opex_p90_scalar)
    if prod_enabled:
        resource_mu, resource_sigma = fit_lognormal_from_p10_p50_p90(resource_p10, resource_p50, resource_p90)

    MC_capex_scalars = np.random.triangular(capex_a, capex_c, capex_b, size=n_simulations)
    MC_opex_scalars = np.random.triangular(opex_a, opex_c, opex_b, size=n_simulations)
    
    for i in range(n_simulations):
        progress_bar.progress((i + 1) / n_simulations)

        # Fit Capex to distribution
        MC_capex_scalar = MC_capex_scalars[i]
        MC_opex_scalar = MC_opex_scalars[i]
        
        MC_capex_profile = Profile_capex_base * MC_capex_scalar

        # Fit Opex to distribution
        
        MC_opex_profile = Profile_opex_base * MC_opex_scalar

        if prod_enabled:
            # Fit production parameters to distribution  
            MC_resource = np.random.lognormal(resource_mu, resource_sigma)
            MC_production_profile = gen_production_profile(initial_prod_p50, resource_decline_rate_p50, MC_resource, facility_capacity, Profile_production_base, decline_model, b=b_value if decline_model == "Hyperbolic" else 0.5)
        else:
            MC_production_profile = Profile_production_base

        # Price
        if run_price_volatility:
            MC_price_profile = price_vol_process(Profile_price_base, 0.2, 0.6)
        else:
            MC_price_profile = Profile_price_base

        MC_Revenue = Profile_price_base * MC_production_profile

        if fiscal_regime == "Production Sharing Contract":
            
            MC_Pre_Cutoff_CF = MC_Revenue - MC_capex_profile - MC_opex_profile
            MC_Pre_Cutoff_CF_cum = MC_Pre_Cutoff_CF.cumsum()
            MC_Max_Cum_idx = np.argmax(MC_Pre_Cutoff_CF_cum)
            MC_flag = np.zeros_like(MC_Pre_Cutoff_CF_cum)
            MC_flag[:MC_Max_Cum_idx+1] = 1
                        
            MC_Capex_econ_cutoff = MC_capex_profile * MC_flag
            MC_Prod_econ_cutoff = MC_production_profile * MC_flag
            MC_Opex_econ_cutoff = MC_opex_profile * MC_flag

            MC_NCF_post_fiscal = calc_PSC(MC_Capex_econ_cutoff, MC_Prod_econ_cutoff, MC_price_profile, MC_Opex_econ_cutoff, psc_cost_recovery_ceiling)
        else:
            if royalty_basis == "Revenues":
                MC_Royalties = MC_Revenue * royalty_rate / 100
            else:
                MC_Royalties = np.zeros_like(MC_Revenue)

            MC_Pre_Cutoff_CF = MC_Revenue - MC_capex_profile - MC_opex_profile - MC_Royalties
            MC_Pre_Cutoff_CF_cum = MC_Pre_Cutoff_CF.cumsum()
            MC_Max_Cum_idx = np.argmax(MC_Pre_Cutoff_CF_cum)
            MC_flag = np.zeros_like(MC_Pre_Cutoff_CF_cum)
            MC_flag[:MC_Max_Cum_idx+1] = 1
                        
            MC_Capex_econ_cutoff = MC_capex_profile * MC_flag
            MC_Prod_econ_cutoff = MC_production_profile * MC_flag
            MC_Opex_econ_cutoff = MC_opex_profile * MC_flag

            MC_NCF_post_fiscal = calc_concession(MC_Capex_econ_cutoff, MC_Prod_econ_cutoff, MC_price_profile, MC_Opex_econ_cutoff, royalty_rate)

        MC_DCF = MC_NCF_post_fiscal * discount_factor
        MC_NPV = np.sum(MC_DCF)
        
        try:
            MC_IRR = npf.irr(MC_NCF_post_fiscal) * 100
            if np.isnan(MC_IRR):
                MC_IRR = None
        except Exception:
            MC_IRR = None
        
        mc_results.append({
            'NPV': MC_NPV,
            'IRR': MC_IRR, 
            "NCF": MC_NCF_post_fiscal
        })

    df_results = pd.DataFrame(mc_results)

    percentile_map = {"P10":10, "P25":25, "P50":50, "P75":75, "P90":90}
    percentile_values = [percentile_map[p] for p in percentiles_selected]

    low_percentile = min(percentiles_selected, key=lambda p: percentile_map[p])
    high_percentile = max(percentiles_selected, key=lambda p: percentile_map[p])

    ncf_matrix = np.stack(df_results['NCF'].values)

    low_percentile_ncf = np.percentile(ncf_matrix, percentile_map[low_percentile], axis=0)
    high_percentile_ncf = np.percentile(ncf_matrix, percentile_map[high_percentile], axis=0)

    label_low_percentile_ncf = f"{low_percentile} NCF"
    label_high_percentile_ncf = f"{high_percentile} NCF"

    npv_count = df_results['NPV'].notna().sum()
    irr_count = df_results['IRR'].notna().sum()

    npv_percentiles = np.percentile(df_results['NPV'].dropna(), percentile_values)
    irr_percentiles = np.percentile(df_results['IRR'].dropna(), percentile_values)

    df_percentiles = pd.DataFrame({
    "Probabilistic Metrics": [f"NPV @ {discount_rate * 100:.0f}% (USD mln)", "IRR (%)"],
    "Valid Sim Count": [npv_count, irr_count],
    **{p: [npv, irr] for p, npv, irr in zip(percentiles_selected, npv_percentiles, irr_percentiles)}
})

    elapsed_time = time.time() - start_time
    time_placeholder.write(f"⏱️ Simulation completed in {elapsed_time:.2f} seconds")

if run_monte_carlo and run_mc_button:
        
    font_size("🎲 Probabilistic Outputs", 18)
    st.caption("Demo has limited probabilistic outputs to NPV and IRR. Valid Sim Count shows the number of non N/A results in simulations, which is relevant to IRR simulations")

    gb_mc = GridOptionsBuilder.from_dataframe(df_percentiles)
    gb_mc.configure_default_column(editable=False, resizable=True, sortable=False, filter=False)


    for col in percentiles_selected:
        gb_mc.configure_column(
            col,     
            width=95,
            editable=False,
            resizable=True,
            sortable=False,
            filter=False,
            cellStyle=mc_cell_style_jscode,
            valueFormatter=value_formatter,
            suppressMenu=True,
            headerCheckboxSelection=False,
            type=["numericColumn"],
            headerComponentParams={'menuIcon':'none'}
            )
        

    gb_mc.configure_column(
        "Valid Sim Count",
        width=100,
        editable=False,
        resizable=True,
        sortable=False,
        filter=False,
        suppressMenu=True,  # THIS disables the menu
        headerCheckboxSelection=False,
        type=["numericColumn"],
        headerComponentParams={'menuIcon':'none'}
    )

    grid_options_mc = gb_mc.build()

    AgGrid(
    df_percentiles,
    gridOptions=grid_options_mc,
    height=93,
    allow_unsafe_jscode=True,
    fit_columns_on_grid_load=True,
    theme="streamlit"
    )
    st.markdown("---")

# -----------------------------------------------------
# Show Cash Flow Chart
# -----------------------------------------------------



if show_p50cf_chart:
    if run_monte_carlo and run_mc_button:

        reversed_idx = next((i for i, val in enumerate(reversed(high_percentile_ncf)) if val != 0), None)

        if reversed_idx is None:
            trailing_zeros_count = len(high_percentile_ncf)  # all zeros
        else:
            trailing_zeros_count = reversed_idx

        chart_length_mc = len(high_percentile_ncf) - trailing_zeros_count

        font_size("P50 Cash Flow Chart with Net Cash Flow Ranges", 18)
        st.caption("Length of cash flow chart extends to the length of the high NPV scenario's NCF profile ")
        set_chart_style(fig_height=7)
        plot_P50_cashflow_with_ranges(Profile_years, Revenue_P50, Capex_P50, Opex_P50, P50_Govt_Take, low_percentile_ncf, high_percentile_ncf, label_low_percentile_ncf, label_high_percentile_ncf, chart_length_mc)

    else:

        font_size("P50 Cash Flow Chart", 18)
        set_chart_style(fig_height=7)
        plot_P50_cashflow(Profile_years, Revenue_P50, Capex_P50, Opex_P50, P50_Govt_Take, chart_length)

    st.markdown("---")

# -----------------------------------------------------
# Calculate Inputs Required for Tornado
# -----------------------------------------------------

def calc_econ_cutoff_flag(capex_array, price_array, production_array, opex_array, royalty_rate=royalty_rate if fiscal_regime == "Concession" and royalty_basis == "Revenues" else 0):
        revenue = price_array * production_array
        pretax_cf = revenue * ( 1 - royalty_rate/100) - capex_array - opex_array
        pretax_cf_cum = pretax_cf.cumsum()
        max_pretax_cf_cum_idx = np.argmax(pretax_cf_cum)
        flag = np.zeros_like(revenue)
        flag[:max_pretax_cf_cum_idx+1] = 1
        return flag


if show_tornado_chart:
    Capex_P10 = Capex_P50 * capex_p10_scalar
    Capex_P10_econ_flag = calc_econ_cutoff_flag(Capex_P10, Profile_price_base, Profile_production_base, Profile_opex_base)
    if fiscal_regime == "Production Sharing Contract":
        Capex_P10_NCF = calc_PSC(Capex_P10, Profile_production_base, Profile_price_base, Profile_opex_base, psc_cost_recovery_ceiling) * Capex_P10_econ_flag
        Capex_P10_DCF = Capex_P10_NCF * discount_factor
        Capex_P10_NPV = np.sum(Capex_P10_DCF)
    else:
        Capex_P10_NCF = calc_concession(Capex_P10, Profile_production_base, Profile_price_base, Profile_opex_base, royalty_rate) * Capex_P10_econ_flag
        Capex_P10_DCF = Capex_P10_NCF * discount_factor
        Capex_P10_NPV = np.sum(Capex_P10_DCF)
    
    Capex_P90 = Capex_P50 * capex_p90_scalar
    Capex_P90_econ_flag = calc_econ_cutoff_flag(Capex_P90, Profile_price_base, Profile_production_base, Profile_opex_base)
    if fiscal_regime == "Production Sharing Contract":
        Capex_P90_NCF = calc_PSC(Capex_P90, Profile_production_base, Profile_price_base, Profile_opex_base, psc_cost_recovery_ceiling) * Capex_P90_econ_flag
        Capex_P90_DCF = Capex_P90_NCF * discount_factor
        Capex_P90_NPV = np.sum(Capex_P90_DCF)
    else:
        Capex_P90_NCF = calc_concession(Capex_P90, Profile_production_base, Profile_price_base, Profile_opex_base, royalty_rate) * Capex_P90_econ_flag
        Capex_P90_DCF = Capex_P90_NCF * discount_factor
        Capex_P90_NPV = np.sum(Capex_P90_DCF)

    Opex_P10 = Opex_P50 * opex_p10_scalar
    Opex_P10_econ_flag = calc_econ_cutoff_flag(Profile_capex_base, Profile_price_base, Profile_production_base, Opex_P10)
    if fiscal_regime == "Production Sharing Contract":
        Opex_P10_NCF = calc_PSC(Profile_capex_base, Profile_production_base, Profile_price_base, Opex_P10, psc_cost_recovery_ceiling) * Opex_P10_econ_flag
        Opex_P10_DCF = Opex_P10_NCF * discount_factor
        Opex_P10_NPV = np.sum(Opex_P10_DCF)
    else:
        Opex_P10_NCF = calc_concession(Profile_capex_base, Profile_production_base, Profile_price_base, Opex_P10, royalty_rate) * Opex_P10_econ_flag
        Opex_P10_DCF = Opex_P10_NCF * discount_factor
        Opex_P10_NPV = np.sum(Opex_P10_DCF)

    Opex_P90 = Opex_P50 * opex_p90_scalar
    Opex_P90_econ_flag = calc_econ_cutoff_flag(Profile_capex_base, Profile_price_base, Profile_production_base, Opex_P90)
    if fiscal_regime == "Production Sharing Contract":
        Opex_P90_NCF = calc_PSC(Profile_capex_base, Profile_production_base, Profile_price_base, Opex_P90, psc_cost_recovery_ceiling) * Opex_P90_econ_flag
        Opex_P90_DCF = Opex_P90_NCF * discount_factor
        Opex_P90_NPV = np.sum(Opex_P90_DCF)
    else:
        Opex_P90_NCF = calc_concession(Profile_capex_base, Profile_production_base, Profile_price_base, Opex_P90, royalty_rate) * Opex_P90_econ_flag
        Opex_P90_DCF = Opex_P90_NCF * discount_factor
        Opex_P90_NPV = np.sum(Opex_P90_DCF)

    tornado_capex_label = str("Capex (P90|P10)") if Capex_P90_NPV < Capex_P10_NPV else str("Capex (P10|P90)")
    tornado_opex_label = str("Opex (P90|P10)") if Opex_P90_NPV < Opex_P10_NPV else str("Opex (P10|P90)")
    tornado_results = [
        {"Variable": tornado_capex_label, "Low": np.min([Capex_P90_NPV,Capex_P10_NPV]) , "High": np.max([Capex_P90_NPV,Capex_P10_NPV])},
        {"Variable": tornado_opex_label, "Low": np.min([Opex_P90_NPV, Opex_P10_NPV]), "High": np.max([Opex_P90_NPV,Opex_P10_NPV])},
    ]

    if prod_enabled:
        Production_P10 = gen_production_profile(initial_prod_p50, resource_decline_rate_p50, resource_p10, facility_capacity, Profile_production_base, decline_model, b=b_value if decline_model == "Hyperbolic" else 0.5)
        Prod_P10_econ_flag = calc_econ_cutoff_flag(Profile_capex_base, Profile_price_base, Production_P10, Profile_opex_base)
        if fiscal_regime == "Production Sharing Contract":
            Prod_P10_NCF = calc_PSC(Profile_capex_base, Production_P10, Profile_price_base, Profile_opex_base, psc_cost_recovery_ceiling) * Prod_P10_econ_flag
            Prod_P10_DCF = Prod_P10_NCF * discount_factor
            Prod_P10_NPV = np.sum(Prod_P10_DCF)
        else:
            Prod_P10_NCF = calc_concession(Profile_capex_base, Production_P10, Profile_price_base, Profile_opex_base, royalty_rate) * Prod_P10_econ_flag
            Prod_P10_DCF = Prod_P10_NCF * discount_factor
            Prod_P10_NPV = np.sum(Prod_P10_DCF)

        Production_P90 = gen_production_profile(initial_prod_p50, resource_decline_rate_p50, resource_p90, facility_capacity, Profile_production_base, decline_model, b=b_value if decline_model == "Hyperbolic" else 0.5)
        Prod_P90_econ_flag = calc_econ_cutoff_flag(Profile_capex_base, Profile_price_base, Production_P90, Profile_opex_base)
        if fiscal_regime == "Production Sharing Contract":
            Prod_P90_NCF = calc_PSC(Profile_capex_base, Production_P90, Profile_price_base, Profile_opex_base, psc_cost_recovery_ceiling) * Prod_P90_econ_flag
            Prod_P90_DCF = Prod_P90_NCF * discount_factor
            Prod_P90_NPV = np.sum(Prod_P90_DCF)
        else:
            Prod_P90_NCF = calc_concession(Profile_capex_base, Production_P90, Profile_price_base, Profile_opex_base, royalty_rate) * Prod_P10_econ_flag
            Prod_P90_DCF = Prod_P90_NCF * discount_factor
            Prod_P90_NPV = np.sum(Prod_P90_DCF)

        tornado_resource_label = str("Resource (P10|P90)") if Prod_P10_NPV < Prod_P90_NPV else str("Resource (P90|P10)")
        tornado_results.append({"Variable": tornado_resource_label, "Low": np.min([Prod_P10_NPV, Prod_P90_NPV]), "High": np.max([Prod_P10_NPV, Prod_P90_NPV])})
        

    tornado_df = pd.DataFrame(tornado_results)

# -----------------------------------------------------
# Show Tornado Chart 
# -----------------------------------------------------

    font_size("Tornado Chart", 18)
    set_chart_style(fig_height=4)
    plot_tornado(tornado_df, NPV_P50)
    st.markdown("---")

# -----------------------------------------------------
# Show S-Curves
# -----------------------------------------------------

# Function - Plot S-Curve
def plot_s_curve(df, column, kind='S-Curve', metric_label="NPV (USD mln)", pzero_threshold=0, percentiles_selected=None, quantile_clip=None):
    data = df[column].dropna().sort_values()
    n = len(data)

    if n == 0:
        st.warning(f"No valid data available to plot {column} S-curve.")
        return 
    
    cdf = np.arange(1, n+1) / n

    if kind == 'Probability of Exceedance':
        y = 1 - cdf 
        ylabel = 'Probability of Exceedance'
    else:
        y = cdf
        ylabel = 'Cumulative Probability (S-Curve)'

    mean = np.mean(data)
    
    if percentiles_selected is None:
        percentiles_selected = ['P10', 'P50', 'P90']

    percentile_map = {"P10": 10, "P25": 25, "P50": 50, "P75": 75, "P90": 90}
    percentiles_values = [percentile_map[p] for p in percentiles_selected]
    percentile_results = [np.percentile(data, p) for p in percentiles_values]
    quantiles = {p: val for p, val in zip(percentiles_selected, percentile_results)}
    quantiles['Mean'] = mean

    fig, ax = plt.subplots(figsize = (8, 3))

    ax.set_facecolor('white')

    for spine in ['top','right','left', 'bottom']: 
        ax.spines[spine].set_color('#cccccc')
        ax.spines[spine].set_linewidth(0.8)

    ax.plot(data, y, label='S-Curve' if kind=='S-Curve' else 'Probability of Exceedance')

    for label, val in quantiles.items():
        if kind == 'Probability of Exceedance':
            q_y = 1 - (np.searchsorted(data, val, side='right')/n)
        else: 
            q_y = (np.searchsorted(data, val, side='right')/n)
        ax.scatter(val, q_y, label=f'{label} = {val:.2f}', zorder=5)
        ax.axvline(x=val, linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel(metric_label)
    ax.set_ylabel(ylabel)
    
    title_text = 'Probability of Exceedance' if kind=='Probability of Exceedance' else 'S-Curve'
    ax.set_title(f"{metric_label} - {title_text}")
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.4)

    if quantile_clip is not None:
        x_min = np.quantile(data, 1 - quantile_clip)
        x_max = np.quantile(data, quantile_clip)
        ax.set_xlim(left=x_min, right=x_max)
    else:
        ax.set_xlim(left=data.min(), right=data.max())
    
    prob_gt_zero = (data > pzero_threshold).sum() / len(data)
    ax.axhline(y=prob_gt_zero, color='red', linestyle=':', linewidth=0.5, label=f'P(X >{pzero_threshold}) = {prob_gt_zero:.2f}')
    ax.legend(frameon=True, fontsize=6, framealpha=0.9, facecolor='white', edgecolor='#dddddd')
    plt.tight_layout()
    add_watermark(ax, "mc_app_base/Enerquill_Logo.webp", 1, 0.15)
    st.pyplot(fig)

if run_monte_carlo and run_mc_button: 
    if show_NPV_dist:
        plot_s_curve(df_results, "NPV", s_curve_type, "NPV (USD mln)")

    if show_IRR_dist:
        plot_s_curve(df_results, "IRR", s_curve_type, "IRR (%)")
