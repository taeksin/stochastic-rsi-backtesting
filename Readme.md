# 간소화된 백테스팅 프로젝트 설계

## 프로젝트 구조 (Simplified)

```
crypto-backtesting/
├── README.md
├── pyproject.toml
├── .env
├── config.py                    # 모든 설정 통합
│
├── src/
│   ├── database.py              # DB 연결 및 데이터 로드
│   ├── candles.py               # 하이킨아시 변환
│   ├── indicators.py            # EMA + 스토캐스틱 RSI
│   ├── strategy.py              # 매매 로직 (롱/숏 조건)
│   ├── backtest.py              # 백테스팅 엔진 (TP/SL 포함)
│   └── metrics.py               # 성과 지표
│
├── app.py                       # Streamlit 메인 앱 (단일 파일)
│
└── tests/
    └── test_strategy.py
```

## 핵심 모듈 설계

### 1. config.py - 설정 관리

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class BacktestConfig:
    # Database
    db_host: str = "59.5.40.202"
    db_port: int = 5432
    db_name: str = "taeksin_DB"
    db_user: str = "taeksin_usr"
    db_password: str = ""
    db_schema: str = "crypto_backtesting"
    
    # Backtest Parameters (UI에서 수정 가능)
    initial_capital: float = 1_000_000  # $1M
    leverage: int = 1                    # 기본 1배
    timeframe: Literal['5min', '10min', '15min'] = '5min'
    
    # Candle Type
    use_heikin_ashi: bool = True        # 하이킨아시 사용 여부
    
    # Moving Average
    ma_type: Literal['EMA', 'SMA'] = 'EMA'
    ma_period: int = 200                 # 기본 200일
    
    # Stochastic RSI
    rsi_period: int = 14
    stoch_period: int = 14
    stoch_k: int = 3
    stoch_d: int = 3
    oversold: int = 20                   # 과매도 기준
    overbought: int = 80                 # 과매수 기준
    
    # Risk Management
    stop_loss_pct: float = 1.5          # 손절 1.5%
    take_profit_pct: float = 2.0        # 익절 2%
```

### 2. src/candles.py - 하이킨아시 계산[1][2]

```python
import pandas as pd

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    일반 캔들 -> 하이킨아시 캔들 변환
    
    HA_Close = (Open + High + Low + Close) / 4
    HA_Open = (이전 HA_Open + 이전 HA_Close) / 2
    HA_High = max(High, HA_Open, HA_Close)
    HA_Low = min(Low, HA_Open, HA_Close)
    """
    ha_df = df.copy()
    
    # HA Close
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # HA Open (초기값)
    ha_df['ha_open'] = (df['open'] + df['close']) / 2
    
    # 두 번째 캔들부터 계산
    for i in range(1, len(ha_df)):
        ha_df.loc[ha_df.index[i], 'ha_open'] = (
            ha_df.loc[ha_df.index[i-1], 'ha_open'] + 
            ha_df.loc[ha_df.index[i-1], 'ha_close']
        ) / 2
    
    # HA High & Low
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return ha_df

def check_ha_candle_strength(df: pd.DataFrame, idx: int, position_type: str) -> bool:
    """
    하이킨아시 캔들 강도 확인
    
    롱: 몸통이 이전보다 크고 + 아래꼬리 없음
    숏: 몸통이 이전보다 크고 + 윗꼬리 없음
    """
    if idx == 0:
        return False
    
    current = df.iloc[idx]
    previous = df.iloc[idx - 1]
    
    # 현재 캔들 몸통 크기
    current_body = abs(current['ha_close'] - current['ha_open'])
    previous_body = abs(previous['ha_close'] - previous['ha_open'])
    
    if position_type == 'LONG':
        # 몸통이 이전보다 크고
        body_bigger = current_body > previous_body
        # 아래꼬리 없음 (ha_low == ha_open or ha_close 중 작은 값)
        no_lower_wick = current['ha_low'] >= min(current['ha_open'], current['ha_close']) - 0.0001
        return body_bigger and no_lower_wick
    
    elif position_type == 'SHORT':
        # 몸통이 이전보다 크고
        body_bigger = current_body > previous_body
        # 윗꼬리 없음
        no_upper_wick = current['ha_high'] <= max(current['ha_open'], current['ha_close']) + 0.0001
        return body_bigger and no_upper_wick
    
    return False
```

### 3. src/indicators.py - 지표 계산

```python
import pandas as pd
import pandas_ta as ta

def calculate_ema(df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
    """EMA 계산"""
    df['ema'] = ta.ema(df['close'], length=period)
    return df

def calculate_stochastic_rsi(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    스토캐스틱 RSI 계산
    %K (파란선), %D (주황선)
    """
    stoch_rsi = ta.stochrsi(
        df['close'],
        length=config.stoch_period,
        rsi_length=config.rsi_period,
        k=config.stoch_k,
        d=config.stoch_d
    )
    
    df['stoch_k'] = stoch_rsi[f'STOCHRSIk_{config.rsi_period}_{config.stoch_period}_{config.stoch_k}_{config.stoch_d}']
    df['stoch_d'] = stoch_rsi[f'STOCHRSId_{config.rsi_period}_{config.stoch_period}_{config.stoch_k}_{config.stoch_d}']
    
    return df
```

### 4. src/strategy.py - 매매 전략

```python
import pandas as pd
from src.candles import check_ha_candle_strength

def generate_signals(df: pd.DataFrame, config, use_heikin_ashi: bool = True) -> pd.DataFrame:
    """
    매매 신호 생성
    
    롱 진입 조건:
    1. 200일선 위에 캔들
    2. 파란선(%K)이 주황선(%D)을 골든크로스 + RSI 20 이하
    3. 하이킨아시: 몸통 커지고 + 아래꼬리 없음
    
    숏 진입 조건:
    1. 200일선 아래에 캔들
    2. 파란선(%K)이 주황선(%D)을 데드크로스 + RSI 80 이상
    3. 하이킨아시: 몸통 커지고 + 윗꼬리 없음
    """
    df['signal'] = None
    
    for i in range(1, len(df)):
        close = df['ha_close'].iloc[i] if use_heikin_ashi else df['close'].iloc[i]
        ema = df['ema'].iloc[i]
        
        # 이전/현재 스토캐스틱 RSI
        prev_k = df['stoch_k'].iloc[i-1]
        prev_d = df['stoch_d'].iloc[i-1]
        curr_k = df['stoch_k'].iloc[i]
        curr_d = df['stoch_d'].iloc[i]
        
        # 골든크로스/데드크로스 확인
        golden_cross = (prev_k <= prev_d) and (curr_k > curr_d)
        death_cross = (prev_k >= prev_d) and (curr_k < curr_d)
        
        # 롱 진입 조건
        if (close > ema and 
            golden_cross and 
            curr_d < config.oversold):
            
            # 하이킨아시 캔들 조건 확인
            if use_heikin_ashi:
                if check_ha_candle_strength(df, i, 'LONG'):
                    df.loc[df.index[i], 'signal'] = 'LONG'
            else:
                df.loc[df.index[i], 'signal'] = 'LONG'
        
        # 숏 진입 조건
        elif (close < ema and 
              death_cross and 
              curr_d > config.overbought):
            
            # 하이킨아시 캔들 조건 확인
            if use_heikin_ashi:
                if check_ha_candle_strength(df, i, 'SHORT'):
                    df.loc[df.index[i], 'signal'] = 'SHORT'
            else:
                df.loc[df.index[i], 'signal'] = 'SHORT'
    
    return df
```

### 5. src/backtest.py - 백테스팅 엔진 (TP/SL 포함)[3][4]

```python
import pandas as pd

class BacktestEngine:
    def __init__(self, initial_capital: float, leverage: int, 
                 stop_loss_pct: float, take_profit_pct: float):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        
        self.position = None  # {'side': 'LONG/SHORT', 'entry_price': float, 'size': float}
        self.trades = []
        self.equity_curve = []
    
    def calculate_position_size(self, price: float) -> float:
        """포지션 크기 계산"""
        return (self.capital * self.leverage) / price
    
    def open_position(self, side: str, price: float, timestamp):
        """포지션 진입"""
        size = self.calculate_position_size(price)
        
        # Stop Loss & Take Profit 계산
        if side == 'LONG':
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
        else:  # SHORT
            stop_loss = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)
        
        self.position = {
            'side': side,
            'entry_price': price,
            'entry_time': timestamp,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def check_exit(self, high: float, low: float, close: float) -> tuple:
        """
        TP/SL 체크 (캔들의 고가/저가 확인)
        Returns: (exit_triggered, exit_price, exit_reason)
        """
        if not self.position:
            return False, None, None
        
        side = self.position['side']
        sl = self.position['stop_loss']
        tp = self.position['take_profit']
        
        if side == 'LONG':
            # 손절 먼저 체크 (보수적)
            if low <= sl:
                return True, sl, 'STOP_LOSS'
            # 익절 체크
            elif high >= tp:
                return True, tp, 'TAKE_PROFIT'
        
        else:  # SHORT
            # 손절 먼저 체크
            if high >= sl:
                return True, sl, 'STOP_LOSS'
            # 익절 체크
            elif low <= tp:
                return True, tp, 'TAKE_PROFIT'
        
        return False, None, None
    
    def close_position(self, exit_price: float, timestamp, reason: str):
        """포지션 청산"""
        entry_price = self.position['entry_price']
        size = self.position['size']
        side = self.position['side']
        
        # PnL 계산
        if side == 'LONG':
            pnl = (exit_price - entry_price) * size
        else:  # SHORT
            pnl = (entry_price - exit_price) * size
        
        pnl_pct = (pnl / self.capital) * 100
        
        # 자본 업데이트
        self.capital += pnl
        
        # 거래 기록
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'capital': self.capital
        })
        
        self.position = None
    
    def run(self, df: pd.DataFrame, use_heikin_ashi: bool = True):
        """백테스팅 실행"""
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # 캔들 데이터 선택
            if use_heikin_ashi:
                close = row['ha_close']
                high = row['ha_high']
                low = row['ha_low']
            else:
                close = row['close']
                high = row['high']
                low = row['low']
            
            # 포지션 있으면 TP/SL 체크
            if self.position:
                exit_triggered, exit_price, reason = self.check_exit(high, low, close)
                if exit_triggered:
                    self.close_position(exit_price, timestamp, reason)
            
            # 신호 확인 및 진입
            if pd.notna(row['signal']) and not self.position:
                self.open_position(row['signal'], close, timestamp)
            
            # 자산 곡선 기록
            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital,
                'return_pct': ((self.capital / self.initial_capital) - 1) * 100
            })
        
        return pd.DataFrame(self.trades), pd.DataFrame(self.equity_curve)
```

### 6. app.py - Streamlit UI (단일 파일)

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import BacktestConfig
from src.database import load_data, resample_timeframe
from src.candles import calculate_heikin_ashi
from src.indicators import calculate_ema, calculate_stochastic_rsi
from src.strategy import generate_signals
from src.backtest import BacktestEngine
from src.metrics import calculate_metrics

st.set_page_config(page_title="Crypto Backtesting", layout="wide")

# 사이드바 - 파라미터 설정
st.sidebar.title("⚙️ 백테스팅 설정")

# 날짜 범위
start_date = st.sidebar.date_input("시작일", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("종료일", value=pd.to_datetime("2024-12-31"))

# 캔들 타입
use_heikin_ashi = st.sidebar.checkbox("하이킨아시 캔들 사용", value=True)

# 타임프레임
timeframe = st.sidebar.selectbox("타임프레임", ['5min', '10min', '15min'], index=0)

# 이동평균선
ma_type = st.sidebar.selectbox("이동평균선 타입", ['EMA', 'SMA'], index=0)
ma_period = st.sidebar.slider("이동평균선 기간", 50, 300, 200)

# 레버리지 & 리스크 관리
leverage = st.sidebar.slider("레버리지", 1, 30, 1)
stop_loss = st.sidebar.slider("손절 (%)", 0.5, 10.0, 1.5, step=0.1)
take_profit = st.sidebar.slider("익절 (%)", 0.5, 15.0, 2.0, step=0.1)

# 스토캐스틱 RSI
with st.sidebar.expander("스토캐스틱 RSI 설정"):
    rsi_period = st.slider("RSI 기간", 5, 30, 14)
    stoch_period = st.slider("Stochastic 기간", 5, 30, 14)
    oversold = st.slider("과매도 기준", 10, 30, 20)
    overbought = st.slider("과매수 기준", 70, 90, 80)

# 설정 객체 생성
config = BacktestConfig(
    leverage=leverage,
    timeframe=timeframe,
    use_heikin_ashi=use_heikin_ashi,
    ma_type=ma_type,
    ma_period=ma_period,
    rsi_period=rsi_period,
    stoch_period=stoch_period,
    oversold=oversold,
    overbought=overbought,
    stop_loss_pct=stop_loss,
    take_profit_pct=take_profit
)

# 메인 화면
st.title("🚀 암호화폐 백테스팅 시스템")

if st.button("백테스팅 시작", type="primary"):
    with st.spinner("데이터 로딩 중..."):
        # 1. 데이터 로드
        df = load_data(start_date, end_date)
        df = resample_timeframe(df, timeframe)
        
        # 2. 하이킨아시 변환
        if use_heikin_ashi:
            df = calculate_heikin_ashi(df)
        
        # 3. 지표 계산
        df = calculate_ema(df, ma_period)
        df = calculate_stochastic_rsi(df, config)
        
        # 4. 신호 생성
        df = generate_signals(df, config, use_heikin_ashi)
        
        # 5. 백테스팅 실행
        engine = BacktestEngine(
            initial_capital=config.initial_capital,
            leverage=leverage,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit
        )
        trades_df, equity_df = engine.run(df, use_heikin_ashi)
        
        # 6. 성과 지표 계산
        metrics = calculate_metrics(trades_df, equity_df, config.initial_capital)
    
    # 결과 표시
    st.success("백테스팅 완료!")
    
    # 성과 요약 카드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 수익률", f"{metrics['total_return']:.2f}%")
    col2.metric("승률", f"{metrics['win_rate']:.2f}%")
    col3.metric("최대 낙폭", f"{metrics['max_drawdown']:.2f}%")
    col4.metric("총 거래", f"{len(trades_df)}회")
    
    # 자산 곡선 차트
    st.subheader("📈 자산 곡선")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df['timestamp'],
        y=equity_df['capital'],
        mode='lines',
        name='자산'
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # 가격 차트 + 신호
    st.subheader("💹 가격 차트 & 매매 신호")
    # ... 차트 구현
    
    # 거래 내역
    st.subheader("📊 거래 내역")
    st.dataframe(trades_df, use_container_width=True)
```

## README.md (간소화)

```markdown
# 🚀 Crypto Backtesting System

## 📋 주요 기능

- ✅ 하이킨아시 / 일반 캔들 전환
- ✅ EMA 200선 기반 추세 판단
- ✅ 스토캐스틱 RSI 크로스오버 시그널
- ✅ 롱/숏 + TP/SL 자동 설정
- ✅ 레버리지 1배 기본

## 🚀 실행

> ⚠️ Python 3.10–3.12 버전을 사용해야 합니다. `uv python install 3.12` 명령으로 호환되는 인터프리터를 설치한 뒤 `uv sync --python 3.12`를 실행하세요.

```
# 의존성 설치 및 동기화
uv sync

# Streamlit 앱 실행
uv run streamlit run app/main.py
```

## 📊 전략 로직

### 롱 진입 조건
1. 캔들이 200일 EMA 위에 위치
2. 스토캐스틱 RSI: %K선이 %D선을 골든크로스 + %D < 20
3. 하이킨아시: 몸통이 이전보다 크고 + 아래꼬리 없음

### 숏 진입 조건
1. 캔들이 200일 EMA 아래에 위치
2. 스토캐스틱 RSI: %K선이 %D선을 데드크로스 + %D > 80
3. 하이킨아시: 몸통이 이전보다 크고 + 윗꼬리 없음

### 청산 조건
- **Stop Loss**: 진입가 대비 -1.5% (기본)
- **Take Profit**: 진입가 대비 +2% (기본)
- 손익비 1.5% : 2%

## ⚙️ 설정 가능 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| 캔들 타입 | 하이킨아시 | 일반 캔들로 변경 가능 |
| 타임프레임 | 5분 | 5/10/15분 선택 |
| 이동평균선 | EMA 200 | 기간 조정 가능 |
| 레버리지 | 1배 | 1~30배 |
| 손절 | 1.5% | 조정 가능 |
| 익절 | 2% | 조정 가능 |

## 📁 구조

```
crypto-backtesting/
├── app/            # Streamlit UI
│   ├── main.py
│   └── pages/
├── config/         # 설정
└── src/
    ├── data/
    ├── indicators/
    ├── strategy/
    └── backtesting/
```
```

## pyproject.toml 요약

```toml
[project]
name = "stochastic-rsi-backtesting"
requires-python = ">=3.10,<3.13"
dependencies = [
  "backtesting==0.3.3",
  "matplotlib==3.8.2",
  "numpy==1.26.2",
  "pandas==2.1.3",
  "plotly==5.18.0",
  "psycopg2-binary==2.9.9",
  "python-dateutil==2.8.2",
  "python-dotenv==1.0.0",
  "pytz==2023.3",
  "sqlalchemy==2.0.23",
  "streamlit==1.29.0",
]
```
