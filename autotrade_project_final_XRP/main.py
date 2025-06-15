from trade_engine import execute_trade, upbit
from utils import (
    get_fear_greed_index, fetch_all_news, evaluate_news,
    send_telegram, is_holiday_or_weekend
)
from db_manager import init_db
import schedule
import random
import time

COINS = ["XRP", "ADA"]
JPN_STOCKS = ["7203.T", "6758.T"]
US_STOCKS = ["AAPL", "MSFT"]
NEWS_RESULTS = {}

def collect_news(asset_list, asset_type):
    if is_holiday_or_weekend() and asset_type != "coin":
        send_telegram(f"⏸️ {asset_type.upper()} 시장 휴장일 (주말/공휴일)")
        return
    for asset in asset_list:
        try:
            news_data = fetch_all_news(asset)
            sentiment = evaluate_news(news_data)
            NEWS_RESULTS[(asset_type, asset)] = sentiment
            send_telegram(f"📰 {asset_type.upper()} {asset} 뉴스 감성: {sentiment}")
        except Exception as e:
            send_telegram(f"❌ {asset_type.upper()} {asset} 뉴스 처리 오류: {e}")

def run_assets(asset_list, asset_type):
    if is_holiday_or_weekend() and asset_type != "coin":
        send_telegram(f"⏸️ {asset_type.upper()} 시장 휴장일 (주말/공휴일)")
        return
    for asset in asset_list:
        try:
            fg = get_fear_greed_index()
            sentiment = NEWS_RESULTS.get((asset_type, asset), "평가 없음")
            rsi = random.randint(30, 70)
            momentum = random.choice(["strong", "weak"])
            price_change = random.uniform(-0.15, 0.05)
            ichimoku = random.choice(["buy", "sell", "hold"])
            candlestick = random.choice(["강한양봉", "약한음봉"])
            volume_spike = random.choice([True, False])
            turnover = random.uniform(1.0, 3.0)
            pattern = random.choice(["상승형", "하락형"])
            sector_trend = random.choice(["positive", "negative"])
            earnings_near = random.choice([True, False])
            volatility = random.uniform(0.01, 0.1)

            execute_trade(
                asset, asset_type, fg, sentiment, rsi, momentum, price_change,
                ichimoku, candlestick, volume_spike, turnover,
                pattern, sector_trend, earnings_near, volatility, upbit
            )
        except Exception as e:
            send_telegram(f"❌ {asset_type.upper()} {asset} 매매 오류: {e}")

if __name__ == "__main__":
    init_db()
    send_telegram("✅ AI 자동매매 스케줄러 시작됨")

    # === 코인 ===
    schedule.every().day.at("00:30").do(collect_news, COINS, "coin")
    schedule.every().day.at("01:00").do(run_assets, COINS, "coin")
    schedule.every().day.at("08:30").do(collect_news, COINS, "coin")
    schedule.every().day.at("09:00").do(run_assets, COINS, "coin")
    schedule.every().day.at("14:30").do(collect_news, COINS, "coin")
    schedule.every().day.at("15:00").do(run_assets, COINS, "coin")
    schedule.every().day.at("20:30").do(collect_news, COINS, "coin")
    schedule.every().day.at("21:00").do(run_assets, COINS, "coin")

    # === 일본 ===
    schedule.every().day.at("10:00").do(collect_news, JPN_STOCKS, "japan")
    schedule.every().day.at("10:30").do(run_assets, JPN_STOCKS, "japan")
    schedule.every().day.at("13:30").do(collect_news, JPN_STOCKS, "japan")
    schedule.every().day.at("14:00").do(run_assets, JPN_STOCKS, "japan")

    # === 미국 ===
    schedule.every().day.at("22:30").do(collect_news, US_STOCKS, "us")
    schedule.every().day.at("23:00").do(run_assets, US_STOCKS, "us")
    schedule.every().day.at("02:30").do(collect_news, US_STOCKS, "us")
    schedule.every().day.at("03:00").do(run_assets, US_STOCKS, "us")

    while True:
        try:
            schedule.run_pending()
            time.sleep(10)
        except Exception as e:
            send_telegram(f"❌ 메인 루프 오류: {e}")
            time.sleep(10)
