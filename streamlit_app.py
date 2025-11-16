import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from datetime import datetime
import talib  # Using talib instead of ta for better compatibility

# Configure the page
st.set_page_config(
    page_title="ETH Trading Bot - Live",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 1000

class ETHTradingBot:
    def __init__(self):
        self.ema_period = 9
        self.resistance_buffer = 20
        self.stop_loss_percent = 1.2
        self.take_profit_percent = 2.5
    
    def get_eth_data(self, timeframe='1h', limit=100):
        """Fetch ETH/USD data from Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'ETHUSDT',
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types safely
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.dropna()
            
            return df if len(df) > 0 else None
            
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators using talib"""
        try:
            if df is None or len(df) < self.ema_period:
                return None, (0, 0)
            
            closes = df['close'].values
            ema_values = talib.EMA(closes, timeperiod=self.ema_period)
            
            current_ema = ema_values[-1] if not np.isnan(ema_values[-1]) else closes[-1]
            resistance_zone = (current_ema, current_ema + self.resistance_buffer)
            
            return df, resistance_zone
            
        except Exception as e:
            st.error(f"Indicator error: {str(e)}")
            return None, (0, 0)
    
    def analyze_market(self, df):
        """Generate trading signals"""
        try:
            if df is None:
                return self._default_analysis("No data")
            
            current_price = df['close'].iloc[-1]
            df_with_indicators, resistance_zone = self.calculate_indicators(df)
            
            if df_with_indicators is None:
                return self._default_analysis("Calculation failed", current_price)
            
            # Get EMA value
            closes = df['close'].values
            ema_values = talib.EMA(closes, timeperiod=self.ema_period)
            current_ema = ema_values[-1] if not np.isnan(ema_values[-1]) else current_price
            
            # Strategy logic
            trend = "BEARISH" if current_price < current_ema else "BULLISH"
            signal = "HOLD"
            reason = "Waiting for setup"
            
            if (trend == "BEARISH" and 
                resistance_zone[0] <= current_price <= resistance_zone[1]):
                signal = "SELL"
                reason = f"Price in resistance zone during downtrend"
            
            return {
                'signal': signal,
                'trend': trend,
                'current_price': current_price,
                'ema_9': current_ema,
                'resistance_zone': resistance_zone,
                'reason': reason,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return self._default_analysis(f"Error: {str(e)}")
    
    def _default_analysis(self, reason, current_price=0):
        """Default analysis for error cases"""
        return {
            'signal': "HOLD",
            'trend': "UNKNOWN",
            'current_price': current_price,
            'ema_9': 0,
            'resistance_zone': (0, 0),
            'reason': reason,
            'timestamp': datetime.now()
        }
    
    def execute_trade(self, analysis, account_balance=1000):
        """Execute trade with risk management"""
        if analysis['signal'] == "HOLD" or analysis['current_price'] <= 0:
            return None
        
        current_price = analysis['current_price']
        
        if analysis['signal'] == "SELL":
            stop_loss = current_price * (1 + self.stop_loss_percent / 100)
            take_profit = current_price * (1 - self.take_profit_percent / 100)
            
            trade = {
                'action': 'SELL',
                'entry_price': current_price,
                'timestamp': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': analysis['reason'],
                'status': 'EXECUTED'
            }
            
            return trade
        
        return None

# Initialize bot
bot = ETHTradingBot()

# UI Components
st.title("🤖 ETH Trading Bot")
st.markdown("Simple EMA-based trading strategy for ETH/USD")

# Sidebar
st.sidebar.header("Controls")

# Strategy parameters
ema_period = st.sidebar.slider("EMA Period", 5, 20, 9)
resistance_buffer = st.sidebar.slider("Resistance Buffer", 10, 50, 20)
stop_loss = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 1.2, 0.1)
take_profit = st.sidebar.slider("Take Profit %", 1.0, 10.0, 2.5, 0.1)

bot.ema_period = ema_period
bot.resistance_buffer = resistance_buffer
bot.stop_loss_percent = stop_loss
bot.take_profit_percent = take_profit

# Account balance
st.session_state.account_balance = st.sidebar.number_input(
    "Account Balance ($)", 100, 100000, st.session_state.account_balance
)

# Control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🚀 Start", use_container_width=True):
        st.session_state.bot_running = True
        st.success("Bot started!")
with col2:
    if st.button("🛑 Stop", use_container_width=True):
        st.session_state.bot_running = False
        st.warning("Bot stopped!")

# Main content
col1, col2, col3 = st.columns(3)

# Fetch data and analyze
df = bot.get_eth_data()
analysis = bot.analyze_market(df) if df is not None else bot._default_analysis("No data")

with col1:
    st.metric("ETH Price", f"${analysis['current_price']:.2f}")

with col2:
    signal_color = "red" if analysis['signal'] == "SELL" else "green"
    st.markdown(f"**Signal:** <span style='color:{signal_color}'>{analysis['signal']}</span>", 
                unsafe_allow_html=True)

with col3:
    st.metric("EMA 9", f"${analysis['ema_9']:.2f}")

# Chart
if df is not None:
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="ETH/USD"
    ))
    
    # EMA line
    closes = df['close'].values
    ema_values = talib.EMA(closes, timeperiod=ema_period)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=ema_values,
        line=dict(color='orange', width=2),
        name=f"EMA {ema_period}"
    ))
    
    fig.update_layout(
        title="ETH/USD Price Chart",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Trading panel
st.subheader("Trading Panel")
st.write(f"**Trend:** {analysis['trend']}")
st.write(f"**Resistance Zone:** ${analysis['resistance_zone'][0]:.2f} - ${analysis['resistance_zone'][1]:.2f}")
st.write(f"**Reason:** {analysis['reason']}")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Check Signal", use_container_width=True):
        st.rerun()

with col2:
    disabled = analysis['signal'] != 'SELL'
    if st.button("📥 Execute SELL", disabled=disabled, use_container_width=True):
        trade = bot.execute_trade(analysis, st.session_state.account_balance)
        if trade:
            st.session_state.trade_history.append(trade)
            st.success(f"SELL executed at ${trade['entry_price']:.2f}")

# Trade history
st.subheader("Trade History")
if st.session_state.trade_history:
    trades_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(trades_df)
else:
    st.info("No trades yet")

# Auto-trading
if st.session_state.bot_running:
    st.warning("🤖 Auto-trading active - checking signals...")
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("*Educational purpose only. Trade at your own risk.*")
