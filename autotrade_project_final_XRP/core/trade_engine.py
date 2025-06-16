import pyupbit
import time
from utils import send_telegram, log_trade, get_price, save_trade
from ib_insync import IB, Stock, MarketOrder

IS_LIVE = True

MAX_ASSET_RATIO = {
    "coin": 0.3,
    "japan": 0.3,
    "us": 0.3
}
MAX_TOTAL_RATIO = {
    "coin": 0.9,
    "japan": 0.9,
    "us": 0.9
}
MIN_CASH_RATIO = 0.1

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # IBKR Gateway에 연결

def ibkr_buy(stock_code, amount):
    contract = Stock(stock_code, 'SMART', 'JPY' if stock_code.endswith('.T') else 'USD')
    order = MarketOrder('BUY', amount)
    ib.placeOrder(contract, order)
    send_telegram(f"✅ IBKR 매수: {stock_code}, 수량: {amount}")

def ibkr_sell(stock_code, amount):
    contract = Stock(stock_code, 'SMART', 'JPY' if stock_code.endswith('.T') else 'USD')
    order = MarketOrder('SELL', amount)
    ib.placeOrder(contract, order)
    send_telegram(f"✅ IBKR 매도: {stock_code}, 수량: {amount}")

def check_asset_ratio(asset, asset_type, asset_value, total_asset_value, cash_balance):
    asset_ratio = asset_value / total_asset_value if total_asset_value > 0 else 0
    if asset_ratio > MAX_ASSET_RATIO[asset_type]:
        send_telegram(f"🚫 {asset_type.upper()} {asset} 매수 보류 (개별 비중 초과)")
        return False
    cash_ratio = cash_balance / total_asset_value if total_asset_value > 0 else 0
    if cash_ratio < MIN_CASH_RATIO:
        send_telegram(f"🚫 {asset_type.upper()} {asset} 매수 보류 (현금 부족)")
        return False
    return True

def execute_trade(asset, asset_type, fg, sentiment, rsi, momentum, price_change,
                  ichimoku, candlestick, volume_spike, turnover,
                  pattern, sector_trend, earnings_near, volatility,
                  total_asset_value, cash_balance, upbit):
    
    now_price = get_price(asset, asset_type)
    asset_value = now_price * 1  # 보유 수량 곱하여 수정 필요 (IBKR API 연동 시)

    decision = "hold"
    confidence = 60

    if asset_type == "coin" and fg <= 70 and "부정" not in sentiment:
        decision = "buy"
        confidence = 85
    elif "부정" in sentiment:
        decision = "sell"
        confidence = 80

    send_telegram(f"🔎 [{asset_type.upper()} {asset}] 결정: {decision} | 신뢰도: {confidence}%")

    if decision == "buy":
        if not check_asset_ratio(asset, asset_type, asset_value, total_asset_value, cash_balance):
            return
        unit = total_asset_value * 0.3
        if asset_type == "coin":
            upbit.buy_market_order(f"KRW-{asset}", unit)
            send_telegram(f"✅ 코인 {asset} 매수 - {unit:,.0f} KRW")
        else:
            ibkr_buy(asset, unit / now_price)
        for i in range(3):
            time.sleep(1)
            current_price = get_price(asset, asset_type)
            if current_price <= now_price * (1 - 0.02 * (i + 1)):
                unit_split = total_asset_value * 0.05
                if asset_type == "coin":
                    upbit.buy_market_order(f"KRW-{asset}", unit_split)
                    send_telegram(f"✅ 코인 {asset} {i+1}차 분할매수 - {unit_split:,.0f} KRW")
                else:
                    ibkr_buy(asset, unit_split / current_price)

    elif decision == "sell":
        if asset_type == "coin":
            balance = upbit.get_balance(asset)
            upbit.sell_market_order(f"KRW-{asset}", balance)
            send_telegram(f"✅ 코인 {asset} 전량 매도")
        else:
            ibkr_sell(asset, 1)

    save_trade(asset, asset_type, {
        "decision": decision,
        "confidence_score": confidence
    }, {
        "asset_balance": 0,
        "cash_balance": cash_balance,
        "avg_price": now_price * 0.95,
        "total_asset": total_asset_value
    }, now_price)

    log_trade(asset, {
        "decision": decision,
        "confidence_score": confidence
    }, {
        "asset_balance": 0,
        "cash_balance": cash_balance,
        "avg_price": now_price * 0.95,
        "total_asset": total_asset_value
    }, now_price)
