import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta

# Configure the page
st.set_page_config(
    page_title="ETH Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("🤖 ETH Automated Trading Bot")
st.markdown("""
This bot implements a **1-hour EMA pullback strategy** for ETH/USD.
The strategy sells when price pulls back to the EMA 9 resistance during a downtrend.
""")

# Initialize session state for tracking
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

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
            
            response = requests.get(url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=self.ema_period).ema_indicator()
        current_ema = df['ema_9'].iloc[-1]
        resistance_zone = (current_ema, current_ema + self.resistance_buffer)
        return df, resistance_zone
    
    def analyze_market(self, df):
        """Analyze market and generate signals"""
        current_price = df['close'].iloc[-1]
        df, resistance_zone = self.calculate_indicators(df)
        
        trend = "BEARISH" if current_price < df['ema_9'].iloc[-1] else "BULLISH"
        signal = "HOLD"
        reason = "Waiting for setup"
        
        if trend == "BEARISH" and resistance_zone[0] <= current_price <= resistance_zone[1]:
            signal = "SELL"
            reason = f"Price in resistance zone ({resistance_zone[0]:.2f}-{resistance_zone[1]:.2f}) during downtrend"
        
        return {
            'signal': signal,
            'trend': trend,
            'current_price': current_price,
            'ema_9': df['ema_9'].iloc[-1],
            'resistance_zone': resistance_zone,
            'reason': reason,
            'timestamp': datetime.now()
        }
    
    def execute_trade(self, analysis, account_balance=1000):
        """Execute trade based on analysis"""
        if analysis['signal'] == "HOLD":
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

# Create sidebar for controls
st.sidebar.header("🤖 Bot Controls")

# Strategy parameters
st.sidebar.subheader("Strategy Parameters")
ema_period = st.sidebar.slider("EMA Period", min_value=5, max_value=20, value=9)
resistance_buffer = st.sidebar.slider("Resistance Buffer ($)", min_value=10, max_value=50, value=20)
stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
take_profit = st.sidebar.slider("Take Profit (%)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

# Initialize bot
bot = ETHTradingBot()
bot.ema_period = ema_period
bot.resistance_buffer = resistance_buffer
bot.stop_loss_percent = stop_loss
bot.take_profit_percent = take_profit

# Control buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🚀 Start Bot", disabled=st.session_state.bot_running):
        st.session_state.bot_running = True
        st.success("Bot started!")

with col2:
    if st.button("🛑 Stop Bot"):
        st.session_state.bot_running = False
        st.warning("Bot stopped!")

st.sidebar.markdown("---")
st.sidebar.header("Account Info")
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=100, max_value=100000, value=1000)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Market Analysis")
    
    # Fetch and display current data
    df = bot.get_eth_data()
    
    if df is not None:
        analysis = bot.analyze_market(df)
        
        # Display current status
        status_color = "red" if analysis['signal'] == "SELL" else "green"
        st.metric(
            label="Current ETH Price",
            value=f"${analysis['current_price']:.2f}",
            delta=f"Signal: {analysis['signal']}",
            delta_color="off" if analysis['signal'] == "HOLD" else status_color
        )
        
        # Create price chart
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="ETH/USD"
        ))
        
        # Add EMA
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ema_9'],
            line=dict(color='orange', width=2),
            name=f"EMA {ema_period}"
        ))
        
        # Add resistance zone
        resistance_low = analysis['resistance_zone'][0]
        resistance_high = analysis['resistance_zone'][1]
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[resistance_low] * len(df),
            line=dict(color='red', width=2, dash='dash'),
            name="Resistance Zone"
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[resistance_high] * len(df),
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="ETH/USD Price Chart with EMA & Resistance",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("📈 Trading Signals")
    
    # Display analysis results
    st.subheader("Current Analysis")
    st.write(f"**Trend:** {analysis['trend']}")
    st.write(f"**Signal:** {analysis['signal']}")
    st.write(f"**EMA {ema_period}:** ${analysis['ema_9']:.2f}")
    st.write(f"**Resistance Zone:** ${analysis['resistance_zone'][0]:.2f} - ${analysis['resistance_zone'][1]:.2f}")
    st.write(f"**Reason:** {analysis['reason']}")
    
    # Manual trade execution
    st.subheader("Manual Control")
    if st.button("🔄 Check Signal", key="manual_check"):
        df = bot.get_eth_data()
        if df is not None:
            analysis = bot.analyze_market(df)
            trade = bot.execute_trade(analysis, account_balance)
            
            if trade:
                st.session_state.trade_history.append(trade)
                st.success(f"Trade Executed: {trade['action']} at ${trade['entry_price']:.2f}")
            else:
                st.info("No trade signal at this time")
    
    if st.button("📥 Execute SELL", type="primary", disabled=analysis['signal'] != 'SELL'):
        df = bot.get_eth_data()
        if df is not None:
            analysis = bot.analyze_market(df)
            trade = bot.execute_trade(analysis, account_balance)
            
            if trade:
                st.session_state.trade_history.append(trade)
                st.success(f"SELL Order Executed at ${trade['entry_price']:.2f}")
                st.write(f"Stop Loss: ${trade['stop_loss']:.2f}")
                st.write(f"Take Profit: ${trade['take_profit']:.2f}")

# Trade History
st.header("📋 Trade History")
if st.session_state.trade_history:
    trades_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(trades_df)
else:
    st.info("No trades executed yet")

# Auto-trading section
st.header("🤖 Auto-Trading Mode")
if st.session_state.bot_running:
    st.warning("Auto-trading is ACTIVE - Bot will check signals every 30 seconds")
    
    # Auto-trading logic
    if st.session_state.last_update is None or (datetime.now() - st.session_state.last_update).seconds >= 30:
        df = bot.get_eth_data()
        if df is not None:
            analysis = bot.analyze_market(df)
            trade = bot.execute_trade(analysis, account_balance)
            
            if trade:
                st.session_state.trade_history.append(trade)
                st.success(f"AUTO-TRADE: {trade['action']} at ${trade['entry_price']:.2f}")
            
            st.session_state.last_update = datetime.now()
            
        # Refresh the page to update
        time.sleep(1)
        st.rerun()
else:
    st.info("Auto-trading is INACTIVE - Click 'Start Bot' to begin")

# Strategy explanation
with st.expander("📚 Strategy Explanation"):
    st.markdown("""
    **EMA Pullback Strategy Rules:**
    
    1. **Trend Identification:** 
       - Price below EMA 9 = Bearish trend
       - Price above EMA 9 = Bullish trend
    
    2. **Entry Signal (SELL):**
       - Bearish trend AND
       - Price enters resistance zone (EMA 9 to EMA 9 + buffer)
    
    3. **Risk Management:**
       - Stop Loss: 1.2% above entry
       - Take Profit: 2.5% below entry
    
    4. **Exit Conditions:**
       - Stop Loss hit = Exit with loss
       - Take Profit hit = Exit with profit
    """)

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This is for educational purposes only. Trade at your own risk.")
