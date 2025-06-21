import os
import time
import asyncio
import ccxt
from ib_insync import IB, Stock, MarketOrder
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantTrading")

# === 데이터 클래스 === #
@dataclass
class MarketData:
    symbol: str
    price: float
    rsi: float
    macd: float
    bb_upper: float
    bb_lower: float

# === Quant API Wrapper === #
class QuantAPIWrapper:
    def __init__(self, upbit_key, upbit_secret):
        self.upbit = ccxt.upbit({
            'apiKey': upbit_key,
            'secret': upbit_secret
        })
        
        self.ib = IB()
        logger.info("✅ API 초기화 완료")
        
    def connect_ibkr(self, host='127.0.0.1', port=7497, clientId=1):
        self.ib.connect(host, port, clientId)
        logger.info("✅ IBKR 연결 완료")
    
    async def fetch_upbit_data(self, symbol='BTC/KRW') -> MarketData:
        ohlcv = await asyncio.to_thread(self.upbit.fetch_ohlcv, symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # 기술 지표 계산
        df['rsi'] = ta_rsi(df['close'])
        df['macd'], _ = ta_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = ta_bollinger(df['close'])
        
        latest = df.iloc[-1]
        
        return MarketData(
            symbol=symbol,
            price=latest['close'],
            rsi=latest['rsi'],
            macd=latest['macd'],
            bb_upper=latest['bb_upper'],
            bb_lower=latest['bb_lower']
        )
    
    def place_upbit_order(self, symbol, side, amount):
        try:
            order = self.upbit.create_order(symbol, 'market', side, amount)
            logger.info(f"🚀 업비트 {side} 주문: {symbol} {amount}")
            return order
        except Exception as e:
            logger.error(f"❌ 업비트 주문 실패: {e}")
            return None
    
    def place_ibkr_order(self, symbol, side, quantity):
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(side, quantity)
        trade = self.ib.placeOrder(contract, order)
        logger.info(f"🚀 IBKR {side} 주문: {symbol} {quantity}")
        return trade

# === 기술 지표 === #
def ta_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def ta_macd(close):
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def ta_bollinger(close, window=20):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

# === 스케줄러 === #
async def scheduler(api: QuantAPIWrapper):
    symbols_upbit = ['BTC/KRW', 'ETH/KRW']
    symbols_ibkr = ['AAPL', 'TSLA']
    
    while True:
        for sym in symbols_upbit:
            data = await api.fetch_upbit_data(sym)
            if data.rsi < 30 or data.price < data.bb_lower:
                api.place_upbit_order(sym, 'buy', 0.001)
            elif data.rsi > 70 or data.price > data.bb_upper:
                api.place_upbit_order(sym, 'sell', 0.001)
        
        for sym in symbols_ibkr:
            # 간단화: 실제 데이터 fetch 생략, 샘플 매매 로직
            api.place_ibkr_order(sym, 'BUY', 1)
        
        logger.info("💡 주기적 매매 완료, 60초 대기")
        await asyncio.sleep(60)

# === 메인 === #
async def main():
    upbit_key = os.getenv('UPBIT_KEY')
    upbit_secret = os.getenv('UPBIT_SECRET')
    api = QuantAPIWrapper(upbit_key, upbit_secret)
    api.connect_ibkr()
    
    await scheduler(api)

if __name__ == "__main__":
    asyncio.run(main())
