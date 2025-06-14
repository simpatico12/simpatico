import pyupbit
import time
import os
from dotenv import load_dotenv
from utils import send_telegram, log_trade

load_dotenv()

UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

IS_LIVE = True
MAX_COIN_RATIO = 0.3
ALLOWED_RATIO = 0.9

def execute_trading_decision(coin, signal):
    ticker = f"KRW-{coin}"
    balance_krw = upbit.get_balance("KRW")
    balances = upbit.get_balances()
    coin_data = next((b for b in balances if b["currency"] == coin), {})
    coin_balance = float(coin_data.get("balance", 0))
    avg_price = float(coin_data.get("avg_buy_price", 0))
    now_price = pyupbit.get_current_price(ticker) or 1
    total_asset = balance_krw + (coin_balance * now_price)
    coin_value_ratio = (coin_balance * now_price) / total_asset if total_asset > 0 else 0

    if signal["confidence_score"] < 70:
        send_telegram(f"🚫 신뢰도 낮음({signal['confidence_score']}%), {coin} 매수 보류")
        return

    # 매수 로직
    if signal["decision"] == "buy" and balance_krw > 5000:
        if coin_value_ratio > MAX_COIN_RATIO:
            send_telegram(f"🚫 {coin} 매수 보류 (비중 초과)")
            return
        if (balance_krw / total_asset) > ALLOWED_RATIO:
            send_telegram(f"🚫 현금 비중 초과로 {coin} 매수 보류")
            return

        for i in range(2):
            unit = total_asset * 0.075
            if IS_LIVE:
                upbit.buy_market_order(ticker, unit)
            send_telegram(f"✅ [{coin}] {i+1}차 분할매수 - {unit:,.0f}원")
            time.sleep(1)

    # 매도 로직
    elif signal["decision"] == "sell" and coin_balance > 0:
        profit_rate = (now_price - avg_price) / avg_price

        # 변동성에 따른 익절/손절 조건
        if signal.get("volatility", "low") == "high":
            target_profit = 0.10
            target_loss = -0.05
        else:
            target_profit = 0.05
            target_loss = -0.03

        if profit_rate >= target_profit:
            sell_qty = coin_balance * 0.5
            for i in range(2):
                if IS_LIVE:
                    upbit.sell_market_order(ticker, sell_qty / 2)
                send_telegram(f"📈 익절 [{coin}] {i+1}차 매도 - {sell_qty/2:.6f}개")
                time.sleep(1)
        elif profit_rate <= target_loss:
            if IS_LIVE:
                upbit.sell_market_order(ticker, coin_balance)
            send_telegram(f"🛑 손절 [{coin}] 전체 매도 - {coin_balance:.6f}개")
        else:
            send_telegram(f"⏸️ {coin} 매도 보류 (익절/손절 조건 불충분)")

    # 거래 기록
    log_trade(coin, signal, coin_balance, balance_krw, avg_price, now_price)


