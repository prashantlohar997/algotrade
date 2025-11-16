import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the page
st.set_page_config(
    page_title="ETH Trading Bot - Live",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffa500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ETHTradingBot:
    def __init__(self):
        self.ema_period = 9
        self.resistance_buffer = 20
        self.stop_loss_percent = 1.2
        self.take_profit_percent = 2.5
        self.data_cache = None
        self.last_fetch_time = None
        
    def safe_api_call(self, url, params, max_retries=3):
        """Make safe API calls with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e
        return None
    
    def get_eth_data(self, timeframe='1h', limit=100):
        """Fetch ETH/USD data from Binance with robust error handling"""
        try:
            # Cache data for 1 minute to avoid rate limiting
            current_time = time.time()
            if (self.data_cache is not None and 
                self.last_fetch_time is not None and
                current_time - self.last_fetch_time < 60):
                return self.data_cache
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'ETHUSDT',
                'interval': timeframe,
                'limit': limit
            }
            
            data = self.safe_api_call(url, params)
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert and clean data
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.dropna()
            
            if len(df) == 0:
                return None
            
            # Cache the data
            self.data_cache = df
            self.last_fetch_time = current_time
            return df
            
        except Exception as e:
            logger.error(f"Error in get_eth_data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators with safety checks"""
        try:
            if df is None or len(df) < self.ema_period:
                return None, (0, 0)
            
            df = df.copy()
            
            # Calculate EMA
            df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=self.ema_period).ema_indicator()
            
            # Get the last valid values
            valid_ema = df['ema_9'].dropna()
            if len(valid_ema) == 0:
                return None, (0, 0)
                
            current_ema = valid_ema.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            resistance_zone = (current_ema, current_ema + self.resistance_buffer)
            
            return df, resistance_zone
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None, (0, 0)
    
    def generate_trading_signal(self, df):
        """Generate trading signals based on strategy rules"""
        try:
            if df is None or len(df) == 0:
                return self._create_default_analysis("No data available")
            
            current_price = df['close'].iloc[-1]
            df_with_indicators, resistance_zone = self.calculate_technical_indicators(df)
            
            if df_with_indicators is None:
                return self._create_default_analysis("Insufficient data for analysis", current_price)
            
            current_ema = df_with_indicators['ema_9'].iloc[-1]
            
            # Strategy logic
            trend = "BEARISH" if current_price < current_ema else "BULLISH"
            signal = "HOLD"
            reason = "Waiting for optimal setup"
            
            # SELL condition: Bearish trend AND price in resistance zone
            if (trend == "BEARISH" and 
                resistance_zone[0] <= current_price <= resistance_zone[1]):
                signal = "SELL"
                reason = (f"Price ${current_price:.2f} in resistance zone "
                         f"(${resistance_zone[0]:.2f}-${resistance_zone[1]:.2f}) during downtrend")
            
            return {
                'signal': signal,
                'trend': trend,
                'current_price': current_price,
                'ema_9': current_ema,
                'resistance_zone': resistance_zone,
                'reason': reason,
                'timestamp': datetime.now(),
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._create_default_analysis(f"Analysis error: {str(e)}")
    
    def _create_default_analysis(self, reason, current_price=0):
        """Create a default analysis response for error cases"""
        return {
            'signal': "HOLD",
            'trend': "UNKNOWN",
            'current_price': current_price,
            'ema_9': 0,
            'resistance_zone': (0, 0),
            'reason': reason,
            'timestamp': datetime.now(),
            'data_points': 0
        }
    
    def execute_trade(self, analysis, account_balance=1000):
        """Execute a trade based on analysis with proper risk management"""
        if (analysis['signal'] == "HOLD" or 
            analysis['current_price'] <= 0 or
            analysis['data_points'] == 0):
            return None
        
        try:
            current_price = analysis['current_price']
            
            if analysis['signal'] == "SELL":
                # Calculate risk management levels
                stop_loss = current_price * (1 + self.stop_loss_percent / 100)
                take_profit = current_price * (1 - self.take_profit_percent / 100)
                
                # Calculate position size (risk 1% of account)
                risk_amount = account_balance * 0.01
                stop_loss_distance = stop_loss - current_price
                position_size = risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
                
                trade = {
                    'action': 'SELL',
                    'entry_price': current_price,
                    'position_size': position_size,
                    'timestamp': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_reward_ratio': abs(take_profit - current_price) / abs(stop_loss - current_price),
                    'reason': analysis['reason'],
                    'status': 'EXECUTED',
                    'account_risk': f"${risk_amount:.2f}"
                }
                
                logger.info(f"Trade executed: {trade}")
                return trade
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'bot_running': False,
        'trade_history': [],
        'last_update': None,
        'last_signal': None,
        'error_message': None,
        'success_message': None,
        'account_balance': 1000,
        'total_trades': 0,
        'winning_trades': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize the bot and session state
initialize_session_state()
bot = ETHTradingBot()

# Header Section
st.markdown('<div class="main-header">🤖 ETH Automated Trading Bot</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
    <strong>Strategy:</strong> EMA Pullback | <strong>Timeframe:</strong> 1-Hour | <strong>Asset:</strong> ETH/USD
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Controls and Configuration
st.sidebar.header("🎮 Control Panel")

# Strategy Configuration
st.sidebar.subheader("⚙️ Strategy Parameters")
ema_period = st.sidebar.slider("EMA Period", 5, 20, 9, help="Period for Exponential Moving Average")
resistance_buffer = st.sidebar.slider("Resistance Buffer ($)", 10, 50, 20, help="Buffer around EMA for resistance zone")
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.2, 0.1, help="Stop loss percentage from entry")
take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 2.5, 0.1, help="Take profit percentage from entry")

# Update bot parameters
bot.ema_period = ema_period
bot.resistance_buffer = resistance_buffer
bot.stop_loss_percent = stop_loss
bot.take_profit_percent = take_profit

# Account Management
st.sidebar.subheader("💰 Account Settings")
st.session_state.account_balance = st.sidebar.number_input(
    "Account Balance ($)", 
    min_value=100, 
    max_value=100000, 
    value=st.session_state.account_balance,
    help="Your trading account balance for position sizing"
)

# Bot Controls
st.sidebar.subheader("🚦 Bot Controls")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🚀 START", use_container_width=True, type="primary"):
        st.session_state.bot_running = True
        st.session_state.success_message = "Trading bot started successfully!"
        st.session_state.error_message = None

with col2:
    if st.button("🛑 STOP", use_container_width=True, type="secondary"):
        st.session_state.bot_running = False
        st.session_state.success_message = "Trading bot stopped."
        st.session_state.error_message = None

# Performance Metrics
st.sidebar.subheader("📊 Performance")
if st.session_state.total_trades > 0:
    win_rate = (st.session_state.winning_trades / st.session_state.total_trades) * 100
    st.sidebar.metric("Win Rate", f"{win_rate:.1f}%")
    st.sidebar.metric("Total Trades", st.session_state.total_trades)
else:
    st.sidebar.info("No trades executed yet")

st.sidebar.markdown("---")
st.sidebar.info("**Note:** This is a demo application for educational purposes only.")

# Main Dashboard
col1, col2, col3 = st.columns(3)

# Fetch current market data
df = bot.get_eth_data()
analysis = bot.generate_trading_signal(df) if df is not None else bot._create_default_analysis("Fetching data...")

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="ETH/USD Price",
        value=f"${analysis['current_price']:.2f}",
        delta=f"{analysis['trend']} Trend"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    signal_class = f"signal-{analysis['signal'].lower()}"
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>Trading Signal</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {'red' if analysis['signal']=='SELL' else 'green' if analysis['signal']=='BUY' else 'orange'};'>{analysis['signal']}</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="EMA 9",
        value=f"${analysis['ema_9']:.2f}",
        delta=f"Resistance: ${analysis['resistance_zone'][1]:.2f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Display messages
if st.session_state.success_message:
    st.success(st.session_state.success_message)
    st.session_state.success_message = None

if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.session_state.error_message = None

# Chart and Analysis Section
tab1, tab2, tab3 = st.tabs(["📈 Live Chart", "🔍 Market Analysis", "📋 Trade History"])

with tab1:
    if df is not None and len(df) > 0:
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="ETH/USD"
        ))
        
        # EMA line
        if 'ema_9' in df.columns and not df['ema_9'].isna().all():
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['ema_9'],
                line=dict(color='orange', width=2),
                name=f"EMA {ema_period}"
            ))
        
        # Current price line
        current_price = analysis['current_price']
        fig.add_hline(y=current_price, line_dash="dot", line_color="blue", 
                     annotation_text=f"Current: ${current_price:.2f}")
        
        # Resistance zone
        res_low, res_high = analysis['resistance_zone']
        if res_low > 0 and res_high > 0:
            fig.add_hrect(y0=res_low, y1=res_high, line_width=0, fillcolor="red", opacity=0.2,
                         annotation_text="Resistance Zone")
        
        fig.update_layout(
            title="ETH/USD Price Chart with Trading Levels",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("📡 Connecting to market data... Please wait.")

with tab2:
    st.subheader("Market Analysis")
    
    if analysis['data_points'] > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Market Condition:**")
            st.write(f"- **Trend:** {analysis['trend']}")
            st.write(f"- **Signal:** {analysis['signal']}")
            st.write(f"- **Price:** ${analysis['current_price']:.2f}")
            st.write(f"- **EMA {ema_period}:** ${analysis['ema_9']:.2f}")
            st.write(f"- **Data Points:** {analysis['data_points']}")
        
        with col2:
            st.write("**Trading Levels:**")
            st.write(f"- **Resistance Zone:** ${analysis['resistance_zone'][0]:.2f} - ${analysis['resistance_zone'][1]:.2f}")
            st.write(f"- **Stop Loss:** {stop_loss}%")
            st.write(f"- **Take Profit:** {take_profit}%")
            st.write(f"- **Risk/Reward:** 1:{take_profit/stop_loss:.1f}")
        
        st.write("**Analysis Reasoning:**")
        st.info(analysis['reason'])
        
        # Manual trade execution
        st.subheader("Manual Trade Execution")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Check Signal", use_container_width=True):
                df = bot.get_eth_data()
                if df is not None:
                    analysis = bot.generate_trading_signal(df)
                    st.session_state.last_signal = analysis
                    st.rerun()
        
        with col2:
            disabled = analysis['signal'] != 'SELL'
            if st.button("📥 Execute SELL", use_container_width=True, disabled=disabled, type="primary"):
                trade = bot.execute_trade(analysis, st.session_state.account_balance)
                if trade:
                    st.session_state.trade_history.append(trade)
                    st.session_state.total_trades += 1
                    st.session_state.success_message = f"SELL order executed at ${trade['entry_price']:.2f}"
                    st.rerun()
    else:
        st.warning("Waiting for market data...")

with tab3:
    st.subheader("Trade History")
    
    if st.session_state.trade_history:
        trades_df = pd.DataFrame(st.session_state.trade_history)
        
        # Calculate PnL for each trade (simulated)
        trades_display = trades_df.copy()
        if not trades_display.empty:
            trades_display['timestamp'] = trades_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            trades_display['pnl'] = (trades_display['take_profit'] - trades_display['entry_price']) * trades_display['position_size']
            trades_display['pnl_pct'] = ((trades_display['take_profit'] - trades_display['entry_price']) / trades_display['entry_price']) * 100
            
            st.dataframe(
                trades_display[[
                    'timestamp', 'action', 'entry_price', 'position_size', 
                    'stop_loss', 'take_profit', 'pnl', 'pnl_pct', 'status'
                ]].style.format({
                    'entry_price': '${:.2f}',
                    'stop_loss': '${:.2f}',
                    'take_profit': '${:.2f}',
                    'pnl': '${:.2f}',
                    'pnl_pct': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.trade_history = []
                st.session_state.total_trades = 0
                st.session_state.winning_trades = 0
                st.rerun()
    else:
        st.info("No trades executed yet. Trade history will appear here.")

# Auto-trading Logic
if st.session_state.bot_running:
    st.markdown("---")
    st.warning("🤖 **AUTO-TRADING ACTIVE** - Bot is monitoring markets and will execute trades automatically")
    
    # Auto-refresh every 30 seconds when bot is running
    if (st.session_state.last_update is None or 
        (datetime.now() - st.session_state.last_update).seconds >= 30):
        
        st.session_state.last_update = datetime.now()
        
        # Fetch new data and check for signals
        df = bot.get_eth_data()
        if df is not None and len(df) > 0:
            analysis = bot.generate_trading_signal(df)
            
            # Auto-execute SELL signals
            if analysis['signal'] == 'SELL':
                trade = bot.execute_trade(analysis, st.session_state.account_balance)
                if trade:
                    st.session_state.trade_history.append(trade)
                    st.session_state.total_trades += 1
                    st.session_state.success_message = f"🤖 AUTO-TRADE: SELL at ${trade['entry_price']:.2f}"
        
        st.rerun()

# Strategy Documentation
with st.expander("📚 Strategy Documentation"):
    st.markdown("""
    ### EMA Pullback Trading Strategy
    
    **Strategy Logic:**
    1. **Trend Identification:**
       - Price < EMA 9 = Bearish Trend
       - Price > EMA 9 = Bullish Trend
    
    2. **Entry Condition (SELL):**
       - Bearish Trend ✅
       - Price in Resistance Zone (EMA 9 to EMA 9 + Buffer) ✅
    
    3. **Risk Management:**
       - Stop Loss: 1.2% above entry price
       - Take Profit: 2.5% below entry price
       - Position Size: Risk 1% of account per trade
    
    **Why This Works:**
    - Sells into strength during downtrends
    - Uses EMA as dynamic resistance
    - Favorable risk-reward ratio (1:2+)
    - Conservative position sizing
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Disclaimer:</strong> This is for educational and demonstration purposes only.</p>
    <p>Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for better user experience
if st.session_state.bot_running:
    time.sleep(2)
    st.rerun()
