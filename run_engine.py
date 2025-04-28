from multiprocessing import Pool
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from backtesting import Backtest, Strategy
import random
from tqdm import tqdm
import warnings
import ast
import json
import ssl
from urllib.error import HTTPError
from urllib.request import urlopen
import certifi
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time

SYMBOL = 'BTCUSD'
N_MONTHS_TO_TEST = 12
N_ITER = 1000
# INTERVAL = '15min' # not changed

N_CORES = 8

# Global params for filter
MIN_SQN = 2
MIN_TRADES_1Y = 5

V3_URL = 'https://financialmodelingprep.com/api/v3/'
API_KEY = ''

def parse_json(url):
    data_received = False
    attempt = 1
    while data_received == False or attempt <= 10:
        try:
            response = urlopen(url, context=ssl.create_default_context(cafile=certifi.where()))
            data = response.read().decode('utf-8')
            data_received = True
            break

        except HTTPError as e:
            if e.code == 500:
                break
            else:
                attempt += 1
                time.sleep(60 - datetime.now().second + 1) # API lim resets when the minute changes

    if data_received == True:        
        return json.loads(data)
    else:
        return False
    
def get_intraday_prices(symbol, start='2022-09-30', end='2022-11-30', freq='15min'):
    url = f'{V3_URL}historical-chart/{freq}/{symbol}?from={start}&to={end}&apikey={API_KEY}'
    print(url)

    json_data = parse_json(url)
    data = pd.DataFrame(json_data)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date', ascending=True)
    data = data.set_index('date')
    data = data.rename({col: col.capitalize() for col in data.columns}, axis=1)
    
    return data

def get_15min_over_1_month(symbol, end='2025-01-01', months=2):
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    complete_data = pd.DataFrame()
    for month in tqdm(range(months)):
        sub_end = end_dt - relativedelta(months=month)
        sub_start = sub_end - relativedelta(months=1)

        sub_end_str = datetime.strftime(sub_end, '%Y-%m-%d')
        sub_start_str = datetime.strftime(sub_start, '%Y-%m-%d')
        
        data = get_intraday_prices(symbol, sub_start_str, sub_end_str)

        complete_data = pd.concat([data, complete_data], axis=0)
        
    return complete_data

class EvalSignals:
    def __init__(self, data):
        self.data = data
        self.signals = self.evaluate_signals()
        
    def evaluate_signals(self):
        signals = {}

        # ---------------------------
        # Volatility Indicators
        # ---------------------------
        # Bollinger Bands (using volatility_bbm, volatility_bbh, volatility_bbl)
        # Open Long: When the close price crosses from above to below the lower band.
        signals['bb_open_long'] = all([
            self.data['volatility_bbl'][-1] > self.data['Close'][-1],   # Now below lower band
            self.data['volatility_bbl'][-2] < self.data['Close'][-2],   # Previously above lower band
        ])
        # Close Long: When price crosses upward through the middle band.
        signals['bb_close_long'] = all([
            self.data['Close'][-1] > self.data['volatility_bbm'][-1],   # Now above middle band
            self.data['Close'][-2] < self.data['volatility_bbm'][-2],   # Previously below middle band
        ])
        # Open Short: When the close price crosses from below to above the upper band.
        signals['bb_open_short'] = all([
            self.data['Close'][-1] > self.data['volatility_bbh'][-1],   # Now above upper band
            self.data['Close'][-2] < self.data['volatility_bbh'][-2],   # Previously below upper band
        ])
        # Close Short: When the price crosses downward through the middle band.
        signals['bb_close_short'] = all([
            self.data['Close'][-1] < self.data['volatility_bbm'][-1],   # Now below middle band
            self.data['Close'][-2] > self.data['volatility_bbm'][-2],   # Previously above middle band
        ])

        # Average True Range (ATR)
        # (Assumes you compare current close with previous close plus/minus a multiple of ATR.)
        signals['atr_open_long'] = self.data['Close'][-1] > self.data['Close'][-2] + 1.5 * self.data['volatility_atr'][-2]
        signals['atr_open_short'] = self.data['Close'][-1] < self.data['Close'][-2] - 1.5 * self.data['volatility_atr'][-2]
        # For closing, we use an arbitrary recent high/low over the last 5 bars.
        signals['atr_close_long'] = self.data['Close'][-1] < max(self.data['Close'][-5:]) - 1 * self.data['volatility_atr'][-1]
        signals['atr_close_short'] = self.data['Close'][-1] > min(self.data['Close'][-5:]) + 1 * self.data['volatility_atr'][-1]

        # Ulcer Index (UI)
        signals['ui_open_long'] = self.data['volatility_ui'][-1] < 10
        signals['ui_close_long'] = self.data['volatility_ui'][-1] > 20
        signals['ui_open_short'] = self.data['volatility_ui'][-1] > 30
        signals['ui_close_short'] = self.data['volatility_ui'][-1] < 20

        # ---------------------------
        # Trend Indicators
        # ---------------------------
        # MACD (trend_macd and its essential signal line)
        signals['macd_open_long'] = all([
            self.data['trend_macd'][-1] > self.data['trend_macd_signal'][-1],
            self.data['trend_macd'][-2] <= self.data['trend_macd_signal'][-2]
        ])
        signals['macd_close_long'] = all([
            self.data['trend_macd'][-1] < self.data['trend_macd_signal'][-1],
            self.data['trend_macd'][-2] >= self.data['trend_macd_signal'][-2]
        ])
        signals['macd_open_short'] = all([
            self.data['trend_macd'][-1] < self.data['trend_macd_signal'][-1],
            self.data['trend_macd'][-2] >= self.data['trend_macd_signal'][-2]
        ])
        signals['macd_close_short'] = all([
            self.data['trend_macd'][-1] > self.data['trend_macd_signal'][-1],
            self.data['trend_macd'][-2] <= self.data['trend_macd_signal'][-2]
        ])

        # Moving Averages: Fast vs. Slow EMA
        signals['ema_open_long'] = all([
            self.data['trend_ema_fast'][-1] > self.data['trend_ema_slow'][-1],
            self.data['trend_ema_fast'][-2] <= self.data['trend_ema_slow'][-2]
        ])
        signals['ema_close_long'] = all([
            self.data['trend_ema_fast'][-1] < self.data['trend_ema_slow'][-1],
            self.data['trend_ema_fast'][-2] >= self.data['trend_ema_slow'][-2]
        ])
        signals['ema_open_short'] = all([
            self.data['trend_ema_fast'][-1] < self.data['trend_ema_slow'][-1],
            self.data['trend_ema_fast'][-2] >= self.data['trend_ema_slow'][-2]
        ])
        signals['ema_close_short'] = all([
            self.data['trend_ema_fast'][-1] > self.data['trend_ema_slow'][-1],
            self.data['trend_ema_fast'][-2] <= self.data['trend_ema_slow'][-2]
        ])

        # ADX (using ADX and its DI components)
        signals['adx_open_long'] = all([
            self.data['trend_adx'][-1] > 25,
            self.data['trend_adx_pos'][-1] > self.data['trend_adx_neg'][-1]
        ])
        signals['adx_close_long'] = any([
            self.data['trend_adx'][-1] < 25,
            self.data['trend_adx_pos'][-1] < self.data['trend_adx_neg'][-1]
        ])
        signals['adx_open_short'] = all([
            self.data['trend_adx'][-1] > 25,
            self.data['trend_adx_neg'][-1] > self.data['trend_adx_pos'][-1]
        ])
        signals['adx_close_short'] = any([
            self.data['trend_adx'][-1] < 25,
            self.data['trend_adx_neg'][-1] < self.data['trend_adx_pos'][-1]
        ])

        # Commodity Channel Index (CCI)
        signals['cci_open_long'] = self.data['trend_cci'][-1] < -100
        signals['cci_close_long'] = self.data['trend_cci'][-1] > 0
        signals['cci_open_short'] = self.data['trend_cci'][-1] > 100
        signals['cci_close_short'] = self.data['trend_cci'][-1] < 0

        # Vortex Indicator (using positive and negative vortex indicators)
        signals['vortex_open_long'] = all([
            self.data['trend_vortex_ind_pos'][-1] > self.data['trend_vortex_ind_neg'][-1],
            self.data['trend_vortex_ind_pos'][-2] <= self.data['trend_vortex_ind_neg'][-2]
        ])
        signals['vortex_close_long'] = all([
            self.data['trend_vortex_ind_pos'][-1] < self.data['trend_vortex_ind_neg'][-1],
            self.data['trend_vortex_ind_pos'][-2] >= self.data['trend_vortex_ind_neg'][-2]
        ])
        signals['vortex_open_short'] = all([
            self.data['trend_vortex_ind_neg'][-1] > self.data['trend_vortex_ind_pos'][-1],
            self.data['trend_vortex_ind_neg'][-2] <= self.data['trend_vortex_ind_pos'][-2]
        ])
        signals['vortex_close_short'] = all([
            self.data['trend_vortex_ind_neg'][-1] < self.data['trend_vortex_ind_pos'][-1],
            self.data['trend_vortex_ind_neg'][-2] >= self.data['trend_vortex_ind_pos'][-2]
        ])

        # Parabolic SAR (assuming trend_psar_down_indicator represents SAR dots)
        signals['psar_open_long'] = all([
            self.data['Close'][-1] > self.data['trend_psar_down_indicator'][-1],
            self.data['Close'][-2] <= self.data['trend_psar_down_indicator'][-2]
        ])
        signals['psar_close_long'] = all([
            self.data['Close'][-1] < self.data['trend_psar_down_indicator'][-1],
            self.data['Close'][-2] >= self.data['trend_psar_down_indicator'][-2]
        ])
        signals['psar_open_short'] = all([
            self.data['Close'][-1] < self.data['trend_psar_down_indicator'][-1],
            self.data['Close'][-2] >= self.data['trend_psar_down_indicator'][-2]
        ])
        signals['psar_close_short'] = all([
            self.data['Close'][-1] > self.data['trend_psar_down_indicator'][-1],
            self.data['Close'][-2] <= self.data['trend_psar_down_indicator'][-2]
        ])

        # ---------------------------
        # Momentum Indicators
        # ---------------------------
        # Relative Strength Index (RSI)
        signals['rsi_open_long'] = self.data['momentum_rsi'][-1] < 30
        signals['rsi_close_long'] = self.data['momentum_rsi'][-1] > 50
        signals['rsi_open_short'] = self.data['momentum_rsi'][-1] > 70
        signals['rsi_close_short'] = self.data['momentum_rsi'][-1] < 50

        # Stochastic RSI
        signals['stoch_rsi_open_long'] = self.data['momentum_stoch_rsi'][-1] < 0.2
        signals['stoch_rsi_close_long'] = self.data['momentum_stoch_rsi'][-1] > 0.5
        signals['stoch_rsi_open_short'] = self.data['momentum_stoch_rsi'][-1] > 0.8
        signals['stoch_rsi_close_short'] = self.data['momentum_stoch_rsi'][-1] < 0.5

        # Ultimate Oscillator (UO)
        signals['uo_open_long'] = self.data['momentum_uo'][-1] < 30
        signals['uo_close_long'] = self.data['momentum_uo'][-1] > 50
        signals['uo_open_short'] = self.data['momentum_uo'][-1] > 70
        signals['uo_close_short'] = self.data['momentum_uo'][-1] < 50

        # Williams %R
        signals['wr_open_long'] = self.data['momentum_wr'][-1] < -80
        signals['wr_close_long'] = self.data['momentum_wr'][-1] > -50
        signals['wr_open_short'] = self.data['momentum_wr'][-1] > -20
        signals['wr_close_short'] = self.data['momentum_wr'][-1] < -50

        # Awesome Oscillator (AO)
        signals['ao_open_long'] = all([
            self.data['momentum_ao'][-1] > 0,
            self.data['momentum_ao'][-2] <= 0
        ])
        signals['ao_close_long'] = all([
            self.data['momentum_ao'][-1] < 0,
            self.data['momentum_ao'][-2] >= 0
        ])
        signals['ao_open_short'] = all([
            self.data['momentum_ao'][-1] < 0,
            self.data['momentum_ao'][-2] >= 0
        ])
        signals['ao_close_short'] = all([
            self.data['momentum_ao'][-1] > 0,
            self.data['momentum_ao'][-2] <= 0
        ])

        # Rate of Change (ROC)
        signals['roc_open_long'] = self.data['momentum_roc'][-1] > 0.5  # e.g., > +0.5%
        signals['roc_close_long'] = self.data['momentum_roc'][-1] < 0     # reverting toward 0
        signals['roc_open_short'] = self.data['momentum_roc'][-1] < -0.5 # e.g., < -0.5%
        signals['roc_close_short'] = self.data['momentum_roc'][-1] > 0

        # Percentage Price Oscillator (PPO)
        signals['ppo_open_long'] = all([
            self.data['momentum_ppo'][-1] > self.data['momentum_ppo_signal'][-1],
            self.data['momentum_ppo'][-2] <= self.data['momentum_ppo_signal'][-2]
        ])
        signals['ppo_close_long'] = all([
            self.data['momentum_ppo'][-1] < self.data['momentum_ppo_signal'][-1],
            self.data['momentum_ppo'][-2] >= self.data['momentum_ppo_signal'][-2]
        ])
        signals['ppo_open_short'] = all([
            self.data['momentum_ppo'][-1] < self.data['momentum_ppo_signal'][-1],
            self.data['momentum_ppo'][-2] >= self.data['momentum_ppo_signal'][-2]
        ])
        signals['ppo_close_short'] = all([
            self.data['momentum_ppo'][-1] > self.data['momentum_ppo_signal'][-1],
            self.data['momentum_ppo'][-2] <= self.data['momentum_ppo_signal'][-2]
        ])

        # Percentage Volume Oscillator (PVO)
        # (Example thresholds; you may wish to adjust these based on volume conditions.)
        signals['pvo_open_long'] = self.data['momentum_pvo'][-1] < 0
        signals['pvo_close_long'] = self.data['momentum_pvo'][-1] >= 0
        signals['pvo_open_short'] = self.data['momentum_pvo'][-1] > 0
        signals['pvo_close_short'] = self.data['momentum_pvo'][-1] <= 0

        # Kaufman’s Adaptive Moving Average (KAMA)
        signals['kama_open_long'] = all([
            self.data['Close'][-1] > self.data['momentum_kama'][-1],
            self.data['Close'][-2] <= self.data['momentum_kama'][-2]
        ])
        signals['kama_close_long'] = all([
            self.data['Close'][-1] < self.data['momentum_kama'][-1],
            self.data['Close'][-2] >= self.data['momentum_kama'][-2]
        ])
        signals['kama_open_short'] = all([
            self.data['Close'][-1] < self.data['momentum_kama'][-1],
            self.data['Close'][-2] >= self.data['momentum_kama'][-2]
        ])
        signals['kama_close_short'] = all([
            self.data['Close'][-1] > self.data['momentum_kama'][-1],
            self.data['Close'][-2] <= self.data['momentum_kama'][-2]
        ])

        # Keltner Channel Signals
        # Assumes:
        # - volatility_kch: Keltner upper boundary
        # - volatility_kcl: Keltner lower boundary
        # - volatility_kcc: Keltner center line

        signals['keltner_open_long'] = all([
            self.data['volatility_kcl'][-1] > self.data['Close'][-1],   # Close price now below lower boundary
            self.data['volatility_kcl'][-2] < self.data['Close'][-2],   # Previous close above lower boundary (crossover downward)
        ])
        signals['keltner_close_long'] = all([
            self.data['Close'][-1] > self.data['volatility_kcc'][-1],    # Now above center line
            self.data['Close'][-2] < self.data['volatility_kcc'][-2],    # Previously below center line
        ])
        signals['keltner_open_short'] = all([
            self.data['Close'][-1] > self.data['volatility_kch'][-1],    # Now above upper boundary
            self.data['Close'][-2] < self.data['volatility_kch'][-2],    # Previously below upper boundary (crossover upward)
        ])
        signals['keltner_close_short'] = all([
            self.data['Close'][-1] < self.data['volatility_kcc'][-1],    # Now below center line
            self.data['Close'][-2] > self.data['volatility_kcc'][-2],    # Previously above center line
        ])

        # Donchian Channel Signals
        # Assumes:
        # - volatility_dch: Donchian upper boundary
        # - volatility_dcl: Donchian lower boundary
        # - volatility_dcm: Donchian middle value

        signals['donchian_open_long']= all([
            self.data['volatility_dcl'][-1] > self.data['Close'][-1],    # Now below lower boundary
            self.data['volatility_dcl'][-2] < self.data['Close'][-2],    # Previously above lower boundary (crossover downward)
        ])
        signals['donchian_close_long'] = all([
            self.data['Close'][-1] > self.data['volatility_dcm'][-1],    # Now above middle value
            self.data['Close'][-2] < self.data['volatility_dcm'][-2],    # Previously below middle value
        ])
        signals['donchian_open_short'] = all([
            self.data['Close'][-1] > self.data['volatility_dch'][-1],    # Now above upper boundary
            self.data['Close'][-2] < self.data['volatility_dch'][-2],    # Previously below upper boundary (crossover upward)
        ])
        signals['donchian_close_short'] = all([
            self.data['Close'][-1] < self.data['volatility_dcm'][-1],    # Now below middle value
            self.data['Close'][-2] > self.data['volatility_dcm'][-2],    # Previously above middle value
        ])
        
        return signals
    

class AutoStrat(Strategy):
    open_long_indicators = ['keltner']
    close_long_indicators = ['keltner']
    
    open_short_indicators = ['keltner']
    close_short_indicators = ['keltner']
    
    current_position_direction = None
    
    def init(self):
        ...

    def next(self):
        signals = EvalSignals(self.data).signals
        
        # Get entry criteria based on defined indicators
        open_long_criteria = []
        for indicator in self.open_long_indicators:
            open_long_criteria.append(signals[f'{indicator}_open_long'])
            
        close_long_criteria = []
        for indicator in self.close_long_indicators:
            close_long_criteria.append(signals[f'{indicator}_close_long'])
            
        open_short_criteria = []
        for indicator in self.open_short_indicators:
            open_short_criteria.append(signals[f'{indicator}_open_short'])
            
        close_short_criteria = []
        for indicator in self.close_short_indicators:
            close_short_criteria.append(signals[f'{indicator}_close_short'])
        
        # Strategy will close a position if close criteria met or if a position is open in the opposite direction.
        if all(open_long_criteria):
            if self.current_position_direction != 'long': # If currently short, then open long, else ignore the signal
                self.position.close()
                self.buy()
                self.current_position_direction = 'long'
                
        if all(close_long_criteria):
            if self.current_position_direction == 'long': # If currently long, then close long, else ignore the signal
                self.position.close()
                self.current_position_direction = None
                
        if all(open_short_criteria):
            if self.current_position_direction != 'short': # If currently long, then open short, else ignore the signal
                self.position.close()
                self.sell()
                self.current_position_direction = 'short'
                
        if all(close_short_criteria):
            if self.current_position_direction == 'short': # If currently short, then close short, else ignore the signal
                self.position.close()
                self.current_position_direction = None

def get_random_choice(l, k):
    # Regenerate until different
    choices = [0] * k
    while len(set(choices)) != k or choices == [0]:
        choices = random.choices(l, k=k)
        
    return choices

def select_random_indicators():
    indicators = [
        'bb',
        'atr',
        'ui',
        'macd',
        'ema', 
        'adx',
        'cci', 
        'vortex',
        'psar',
        'rsi',
        'stoch_rsi',
        'uo', 
        'wr',
        'ao', 
        'roc',
        'ppo',
        'pvo',
        'kama',
        'keltner',
        'donchian'
    ]

    # Select 1 to 3 indicators at once
    open_long_n_indicators = random.randint(1, 3)
    close_long_n_indicators = random.randint(1, 3)
    open_short_n_indicators = random.randint(1, 3)
    close_short_n_indicators = random.randint(1, 3)
    
    open_long_indicators = get_random_choice(indicators, open_long_n_indicators)
    close_long_indicators = get_random_choice(indicators, close_long_n_indicators)

    open_short_indicators = get_random_choice(indicators, open_short_n_indicators)
    close_short_indicators = get_random_choice(indicators, close_short_n_indicators)
    
    return open_long_indicators, close_long_indicators, open_short_indicators, close_short_indicators

def run_simulation(data, core, n=int(N_ITER / N_CORES)):
    # Take the best strategies from the most recent run and run these first in core 0. No filtering needed as we know only the best strats are in the output
    indicators_to_test = []
    try:
        if core == 0:
            previous_run = pd.read_csv(f'{SYMBOL}_last_run.csv')

            for row in range(len(previous_run)):
                open_long_indicators = ast.literal_eval(previous_run.iloc[row, :]['open_long_indicators'])
                close_long_indicators = ast.literal_eval(previous_run.iloc[row, :]['close_long_indicators'])
                open_short_indicators = ast.literal_eval(previous_run.iloc[row, :]['open_short_indicators'])
                close_short_indicators = ast.literal_eval(previous_run.iloc[row, :]['close_short_indicators'])
                
                indicators_to_test.append((open_long_indicators, close_long_indicators, open_short_indicators, close_short_indicators))
    except Exception as e:
        print(e)
        pass
    
    for i in tqdm(range(n), desc='Generating strategies'):
        open_long_indicators, close_long_indicators, open_short_indicators, close_short_indicators = select_random_indicators()
        indicators_to_test.append((open_long_indicators, close_long_indicators, open_short_indicators, close_short_indicators))

    all_results_table = pd.DataFrame()
    for open_long_indicators, close_long_indicators, open_short_indicators, close_short_indicators in tqdm(indicators_to_test):
        train_bt = Backtest(data.iloc[0:int(len(data) / 2), :], AutoStrat, cash=10_000_000, commission=.002, exclusive_orders=True, hedging=False)
        train_stats = train_bt.run(open_long_indicators=open_long_indicators, close_long_indicators=close_long_indicators, open_short_indicators=open_short_indicators, close_short_indicators=close_short_indicators)

        train_return = train_stats['Return [%]']
        train_buy_hold_return = train_stats['Buy & Hold Return [%]']
        train_avg_trade_return = train_stats['Avg. Trade [%]']
        train_avg_trade_duration = train_stats['Avg. Trade Duration']
        train_max_trade_duration = train_stats['Max. Trade Duration']
        train_max_drawdown = train_stats['Max. Drawdown [%]']
        train_sqn = train_stats['SQN']
        train_win_rate = train_stats['Win Rate [%]']
        train_n_trades = train_stats['# Trades']
        
        # Out of sample testing
        test_bt = Backtest(data.iloc[int(len(data) / 2):, :], AutoStrat, cash=10_000_000, commission=.002, exclusive_orders=True, hedging=False)
        test_stats = test_bt.run(open_long_indicators=open_long_indicators, close_long_indicators=close_long_indicators, open_short_indicators=open_short_indicators, close_short_indicators=close_short_indicators)
        
        test_return = test_stats['Return [%]']
        test_buy_hold_return = test_stats['Buy & Hold Return [%]']
        test_avg_trade_return = test_stats['Avg. Trade [%]']
        test_avg_trade_duration = test_stats['Avg. Trade Duration']
        test_max_trade_duration = test_stats['Max. Trade Duration']
        test_max_drawdown = test_stats['Max. Drawdown [%]']
        test_sqn = test_stats['SQN']
        test_win_rate = test_stats['Win Rate [%]']
        test_n_trades = test_stats['# Trades']

        # Results on entire data (used for filtering within app)
        bt = Backtest(data, AutoStrat, cash=10_000_000, commission=.002, exclusive_orders=True, hedging=False)
        stats = bt.run(open_long_indicators=open_long_indicators, close_long_indicators=close_long_indicators, open_short_indicators=open_short_indicators, close_short_indicators=close_short_indicators)
        
        total_return = stats['Return [%]']
        total_buy_hold_return = stats['Buy & Hold Return [%]']
        total_avg_trade_return = stats['Avg. Trade [%]']
        total_avg_trade_duration = stats['Avg. Trade Duration']
        total_max_trade_duration = stats['Max. Trade Duration']
        total_max_drawdown = stats['Max. Drawdown [%]']
        total_sqn = stats['SQN']
        total_win_rate = stats['Win Rate [%]']
        total_n_trades = stats['# Trades']

        results_table = pd.DataFrame([{
            'open_long_indicators': open_long_indicators,                                     
            'close_long_indicators': close_long_indicators,                                     
            'open_short_indicators': open_short_indicators,                                       
            'close_short_indicators': close_short_indicators,  
            
            'train_return': train_return,        
            'train_buy_hold_return': train_buy_hold_return,                            
            'train_avg_trade_return': train_avg_trade_return,
            'train_avg_trade_duration': train_avg_trade_duration,
            'train_max_trade_duration': train_max_trade_duration,
            'train_max_drawdown': train_max_drawdown,
            'train_sqn': train_sqn,
            'train_win_rate': train_win_rate,
            'train_n_trades': train_n_trades,
            
            'test_return': test_return,   
            'test_buy_hold_return': test_buy_hold_return,                                 
            'test_avg_trade_return': test_avg_trade_return,
            'test_avg_trade_duration': test_avg_trade_duration,
            'test_max_trade_duration': test_max_trade_duration,
            'test_max_drawdown': test_max_drawdown,
            'test_sqn': test_sqn,
            'test_win_rate': test_win_rate,
            'test_n_trades': test_n_trades,

            'total_return': total_return,
            'total_buy_hold_return': total_buy_hold_return,                                 
            'total_avg_trade_return': total_avg_trade_return,
            'total_avg_trade_duration': total_avg_trade_duration,
            'total_max_trade_duration': total_max_trade_duration,
            'total_max_drawdown': total_max_drawdown,
            'total_sqn': total_sqn,
            'total_win_rate': total_win_rate,
            'total_n_trades': total_n_trades,
        }])
        
        all_results_table = pd.concat([all_results_table, results_table], axis=0)

        # Checkpoint save
        all_results_table.to_csv(f'tmp/{core}_out.csv', index=False)
        
    return all_results_table

if __name__ == '__main__':
    data = get_15min_over_1_month(SYMBOL, end=datetime.strftime(datetime.today(), '%Y-%m-%d'), months=N_MONTHS_TO_TEST)
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])

    data_ta = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume')

    data_ta = data_ta.drop(['trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator'], axis=1)
    data_ta = data_ta.dropna(axis=0)

    with Pool(N_CORES) as p:
        out = p.starmap(run_simulation, [(data_ta, n) for n in range(N_CORES)])

    results = pd.concat(out, axis=0)
    
    # Filter results on criteria
    results = results[(results['train_sqn'] >= MIN_SQN) &
                      (results['test_sqn'] >= MIN_SQN) &
                      (results['total_sqn'] >= MIN_SQN) &
                      (results['total_n_trades'] > MIN_TRADES_1Y) &

                      # Ensure results are better than buy and hold
                      (results['train_return'] > results['train_buy_hold_return']) &
                      (results['test_return'] > results['test_buy_hold_return']) &
                      (results['total_return'] > results['total_buy_hold_return'])]

    results.to_csv(f'{SYMBOL}_last_run.csv', index=False)