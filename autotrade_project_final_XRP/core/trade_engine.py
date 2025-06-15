import pyupbit
import os
import time
from dotenv import load_dotenv
from utils import send_telegram, get_price

load_dotenv()
UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

IS_LIVE = True

MAX_ASSET_RATIO = 0.45
ALLOWED_ASSET_TOTAL_RATIO = 0.9
ALLOWED_CASH_RATIO = 0.1

def ibkr_buy(stock, amount):
    send_telegram(f"✅ IBKR 매수 실행: {stock}, 금액: {amount:,.0f}")

def ibkr_sell(stock, qty):
    send_telegram(f"✅ IBKR 매도 실행: {stock}, 수량: {qty:.4f}")

def execute_trade(asset, asset_type, fg, sentiment, rsi, momentum, price_change,
                  ichimoku, candlestick, volume_spike, turnover,
                  pattern, sector_trend, earnings_near, volatility, upbit):

    now_price = get_price(asset, asset_type)
    decision = "hold"
    confidence = 60

    if asset_type == "coin":
        if momentum == "strong" and fg > 60 and price_change > 0:
            decision = "buy"
            confidence = 85
        elif price_change < -0.05:
            decision = "buy"
            confidence = 80
        elif price_change > 0.05:
            decision = "sell"
            confidence = 80
    else:
        if momentum == "strong" and volatility <= 0.05 and not earnings_near:
            decision = "buy"
            confidence = 85
        elif volatility > 0.07:
            decision = "sell"
            confidence = 80

    send_telegram(f"🔎 [{asset_type.upper()} {asset}] 결정: {decision} | 신뢰도: {confidence}%")

    if asset_type == "coin":
        ticker = f"KRW-{asset}"
        balance_krw = upbit.get_balance("KRW")
        balances = upbit.get_balances()
        coin_data = next((b for b in balances if b["currency"] == asset), {})
        coin_balance = float(coin_data.get("balance", 0))
        avg_price = float(coin_data.get("avg_buy_price", now_price))  # 첫 매수 시 현재가 사용

        total_asset = balance_krw + sum(
            float(b["balance"]) * (pyupbit.get_current_price(f'KRW-{b["currency"]}') or 1)
            for b in balances if b["currency"] != "KRW"
        )

        if decision == "buy":
            if (balance_krw / total_asset) < ALLOWED_CASH_RATIO:
                send_telegram(f"🚫 {asset} 매수 보류 (현금 부족)")
                return
            # 2% 하락마다 5% 매수 (최대 3회)
            target_prices = [now_price, now_price * 0.98, now_price * 0.96]
            bought_count = 0

            for target in target_prices:
                while True:
                    current = pyupbit.get_current_price(ticker)
                    if current <= target:
                        unit = total_asset * 0.05
                        if IS_LIVE:
                            upbit.buy_market_order(ticker, unit)
                            send_telegram(f"✅ {asset} {bought_count+1}차 매수: {unit:,.0f} KRW (가격: {current:,.0f})")
                        else:
                            send_telegram(f"📝 {asset} {bought_count+1}차 모의매수: {unit:,.0f} KRW (가격: {current:,.0f})")
                        bought_count += 1
                        time.sleep(1)
                        break
                    else:
                        time.sleep(5)  # 5초 간격으로 가격 확인

                    if bought_count >= 3:
                        break

        elif decision == "sell" and coin_balance > 0:
            profit_rate = (now_price - avg_price) / avg_price
            qty = coin_balance
            if profit_rate >= 0.05:
                if IS_LIVE:
                    upbit.sell_market_order(ticker, qty)
                    send_telegram(f"📈 {asset} 익절 매도 - {qty:.6f} (익절율: {profit_rate:.2%})")
                else:
                    send_telegram(f"📝 {asset} 모의 익절 - {qty:.6f} (익절율: {profit_rate:.2%})")
            elif profit_rate <= -0.03:
                if IS_LIVE:
                    upbit.sell_market_order(ticker, qty)
                    send_telegram(f"🛑 {asset} 손절 매도 - {qty:.6f} (손실율: {profit_rate:.2%})")
                else:
                    send_telegram(f"📝 {asset} 모의 손절 - {qty:.6f} (손실율: {profit_rate:.2%})")
            else:
                send_telegram(f"⏸️ {asset} 매도 보류 (익절/손절 조건 부족, 현재 수익률: {profit_rate:.2%})")

    elif asset_type in ["japan", "us"]:
        if decision == "buy":
            ibkr_buy(asset, 1000000)
        elif decision == "sell":
            ibkr_sell(asset, 10)
