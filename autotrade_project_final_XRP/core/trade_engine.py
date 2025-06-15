import pyupbit
import os
import time
from dotenv import load_dotenv
from utils import send_telegram, log_trade

# IBKR 더미 함수 (실매매 차단)
def ibkr_buy(stock, amount):
    send_telegram(f"🚫 [BLOCKED] IBKR 매수 차단됨: {stock}, 금액: {amount:,.0f}")

def ibkr_sell(stock, qty):
    send_telegram(f"🚫 [BLOCKED] IBKR 매도 차단됨: {stock}, 수량: {qty:.4f}")

load_dotenv()

UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

IS_LIVE = True  # 기본 테스트 모드

MAX_ASSET_RATIO = 0.3
ALLOWED_ASSET_TOTAL_RATIO = 0.9
ALLOWED_CASH_RATIO = 0.1

def execute_trading_decision(asset, signal, asset_type="coin"):
    try:
        if asset_type == "coin":
            execute_coin_trade(asset, signal)
        else:
            # 일본/미국 매매 차단
            send_telegram(f"🚫 [{asset}] {asset_type.upper()} 매매 차단됨 (코인 전용 실행)")
    except Exception as e:
        send_telegram(f"❌ [{asset}] 매매 실행 오류: {e}")

def execute_coin_trade(coin, signal):
    ticker = f"KRW-{coin}"
    balance_krw = upbit.get_balance("KRW")
    balances = upbit.get_balances()
    coin_data = next((b for b in balances if b["currency"] == coin), {})
    coin_balance = float(coin_data.get("balance", 0))
    avg_price = float(coin_data.get("avg_buy_price", 0))
    now_price = pyupbit.get_current_price(ticker) or 1

    total_asset = balance_krw + sum(
        float(b["balance"]) * (pyupbit.get_current_price(f'KRW-{b["currency"]}') or 1)
        for b in balances if b["currency"] != "KRW"
    )

    coin_value = coin_balance * now_price
    coin_value_ratio = coin_value / total_asset if total_asset > 0 else 0
    total_coin_value = sum(
        float(b["balance"]) * (pyupbit.get_current_price(f'KRW-{b["currency"]}') or 1)
        for b in balances if b["currency"] != "KRW"
    )
    total_coin_ratio = total_coin_value / total_asset if total_asset > 0 else 0

    if signal["decision"] == "buy":
        if coin_value_ratio > MAX_ASSET_RATIO:
            send_telegram(f"🚫 {coin} 매수 보류 (개별 비중 초과)")
            return
        if total_coin_ratio > ALLOWED_ASSET_TOTAL_RATIO:
            send_telegram(f"🚫 {coin} 매수 보류 (전체 코인 비중 초과)")
            return
        if (balance_krw / total_asset) < ALLOWED_CASH_RATIO:
            send_telegram(f"🚫 {coin} 매수 보류 (현금 부족)")
            return
        if signal["confidence_score"] < 70:
            send_telegram(f"🚫 신뢰도 낮음 {coin} 매수 보류")
            return

        for i in range(2):
            unit = total_asset * 0.075
            if IS_LIVE:
                upbit.buy_market_order(ticker, unit)
                send_telegram(f"✅ [{coin}] {i+1}차 실매수 - {unit:,.0f}원")
            else:
                send_telegram(f"📝 [MOCK] {i+1}차 매수 예정 - {unit:,.0f}원")
            time.sleep(1)

    elif signal["decision"] == "sell" and coin_balance > 0:
        profit_rate = (now_price - avg_price) / avg_price
        target_profit = 0.10 if signal.get("volatility") == "high" else 0.05
        target_loss = -0.05 if signal.get("volatility") == "high" else -0.03

        if profit_rate >= target_profit:
            for i in range(2):
                qty = coin_balance * 0.5 / 2
                if IS_LIVE:
                    upbit.sell_market_order(ticker, qty)
                    send_telegram(f"📈 {i+1}차 실익절 매도 {qty:.6f}")
                else:
                    send_telegram(f"📝 [MOCK] {i+1}차 익절 매도 {qty:.6f}")
                time.sleep(1)
        elif profit_rate <= target_loss:
            if IS_LIVE:
                upbit.sell_market_order(ticker, coin_balance)
                send_telegram(f"🛑 실손절 전체 매도 {coin_balance:.6f}")
            else:
                send_telegram(f"📝 [MOCK] 손절 전체 매도 {coin_balance:.6f}")
        else:
            send_telegram(f"⏸️ {coin} 매도 보류 (익절/손절 조건 부족)")

    log_trade(coin, signal, coin_balance, balance_krw, avg_price, now_price)
