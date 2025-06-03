import sqlite3
import os
import time
import requests
import pyupbit
import schedule
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1) .env 로드
load_dotenv()
UPBIT_ACCESS_KEY   = os.getenv("UPBIT_ACCESS_KEY")
UPBIT_SECRET_KEY   = os.getenv("UPBIT_SECRET_KEY")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
IS_LIVE            = os.getenv("IS_LIVE", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# 2) 디버그용: 환경 변수 출력 (None이면 .env 설정 확인)
print("DEBUG: UPBIT_ACCESS_KEY  =", UPBIT_ACCESS_KEY)
print("DEBUG: UPBIT_SECRET_KEY  =", UPBIT_SECRET_KEY)
print("DEBUG: TELEGRAM_TOKEN    =", TELEGRAM_TOKEN)
print("DEBUG: TELEGRAM_CHAT_ID  =", TELEGRAM_CHAT_ID)
print("DEBUG: IS_LIVE           =", IS_LIVE)
print("===============================================\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Upbit 인스턴스 생성 (IP 화이트리스트에 서버 IP 등록 필요)
upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 4) SQLite DB 설정
DB_PATH = "trading.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS trading_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    coin TEXT,
    decision TEXT,
    percentage INTEGER,
    confidence_score INTEGER,
    reason TEXT,
    reaction TEXT,
    coin_balance REAL,
    krw_balance REAL,
    avg_buy_price REAL,
    coin_price REAL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS trading_reflection (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trading_id INTEGER NOT NULL,
    reflection_date DATETIME NOT NULL,
    market_condition TEXT NOT NULL,
    decision_analysis TEXT NOT NULL,
    improvement_points TEXT NOT NULL,
    success_rate REAL NOT NULL,
    learning_points TEXT NOT NULL,
    FOREIGN KEY (trading_id) REFERENCES trading_history(id)
)
""")
conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# 5) Telegram 메시지 전송 함수
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("텔레그램 오류:", e)

# ─────────────────────────────────────────────────────────────────────────────
# 6) 거래 기록 저장 함수
def record_trade(coin, decision, percentage, confidence_score,
                 reason, reaction, coin_balance, krw_balance,
                 avg_price, coin_price):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO trading_history "
        "(timestamp, coin, decision, percentage, confidence_score, reason, reaction, coin_balance, krw_balance, avg_buy_price, coin_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, coin, decision, percentage, confidence_score,
         reason, reaction, coin_balance, krw_balance, avg_price, coin_price)
    )
    conn.commit()

# ─────────────────────────────────────────────────────────────────────────────
# 7) 공포·탐욕 지수 조회 함수 (Fear & Greed Index)
def get_fear_greed_index():
    """
    https://api.alternative.me/fng/ 의 JSON에서
    data[0]['value']를 int로 반환
    """
    try:
        resp = requests.get("https://api.alternative.me/fng/", timeout=5)
        data = resp.json()
        fng_str = data.get("data", [{}])[0].get("value", "50")
        return int(fng_str)
    except Exception as e:
        print("F&G 지수 조회 오류:", e)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 8) 워런 버핏식 가치 투자 (샘플 내재 가치 계산)
def calculate_intrinsic_value(coin):
    # 실제로는 펀더멘털 API(예: CoinGecko, Dune Analytics 등)로 계산해야 함
    return 1_000_000  # 예시: 모든 코인의 내재 가치를 1,000,000원으로 가정

def analyze_market_buffett(coin):
    intrinsic = calculate_intrinsic_value(coin)
    market_price = pyupbit.get_current_price(f"KRW-{coin}") or 0
    margin_of_safety = 0.3  # 30% 할인
    if market_price < intrinsic * (1 - margin_of_safety):
        return {
            "decision": "buy",
            "reason": f"저평가: 시장가 {market_price:,.0f} < 내재가 {intrinsic:,.0f}",
            "confidence_score": 90,
            "percentage": 50
        }
    elif intrinsic > 0 and market_price > intrinsic * 1.1:  # 10% 고평가 구간
        return {
            "decision": "sell",
            "reason": f"고평가: 시장가 {market_price:,.0f} > 내재가 {intrinsic:,.0f}",
            "confidence_score": 80,
            "percentage": 100
        }
    else:
        return {
            "decision": "hold",
            "reason": "적정가 구간",
            "confidence_score": 50,
            "percentage": 0
        }

# ─────────────────────────────────────────────────────────────────────────────
# 9) 제시 리버모어식 추세 추종 (이동평균 교차 예시)
def determine_trend(df):
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    latest = df.iloc[-1]
    if latest['ma20'] > latest['ma60']:
        return "uptrend"
    elif latest['ma20'] < latest['ma60']:
        return "downtrend"
    else:
        return "sideways"

def analyze_market_livermore(coin):
    try:
        df = pyupbit.get_ohlcv(f"KRW-{coin}", interval="day", count=100)
        if df is None or len(df) < 60:
            return {"decision": "hold", "reason": "데이터 부족", "confidence_score": 0, "percentage": 0}

        trend = determine_trend(df)
        if trend == "uptrend":
            return {
                "decision": "buy",
                "reason": "20일선 > 60일선(상승 추세)",
                "confidence_score": 75,
                "percentage": 50
            }
        elif trend == "downtrend":
            return {
                "decision": "sell",
                "reason": "20일선 < 60일선(하락 추세)",
                "confidence_score": 75,
                "percentage": 100
            }
        else:
            return {"decision": "hold", "reason": "추세 불명확", "confidence_score": 40, "percentage": 0}
    except Exception as e:
        return {"decision": "hold", "reason": f"리버모어 전략 오류: {e}", "confidence_score": 0, "percentage": 0}

# ─────────────────────────────────────────────────────────────────────────────
# 10) 워뇨띠 전략 (거래량 급증 예시)
def analyze_market_woonyoddi(coin):
    try:
        df = pyupbit.get_ohlcv(f"KRW-{coin}", interval="day", count=2)
        if df is None or len(df) < 2:
            return {"decision": "hold", "reason": "데이터 부족", "confidence_score": 0, "percentage": 0}

        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        if today['volume'] > yesterday['volume'] * 1.5:
            return {
                "decision": "buy",
                "reason": "거래량 급증 감지",
                "confidence_score": 60,
                "percentage": 30
            }
        else:
            return {"decision": "hold", "reason": "거래량 정상", "confidence_score": 30, "percentage": 0}
    except Exception as e:
        return {"decision": "hold", "reason": f"워뇨띠 전략 오류: {e}", "confidence_score": 0, "percentage": 0}

# ─────────────────────────────────────────────────────────────────────────────
# 11) 세 전략 통합
def analyze_market_combined(coin):
    buffett_signal = analyze_market_buffett(coin)
    if buffett_signal["decision"] in ["buy", "sell"]:
        return buffett_signal

    livermore_signal = analyze_market_livermore(coin)
    if livermore_signal["decision"] in ["buy", "sell"]:
        return livermore_signal

    woonyoddi_signal = analyze_market_woonyoddi(coin)
    return woonyoddi_signal

# ─────────────────────────────────────────────────────────────────────────────
# 12) 안전한 잔고 조회 헬퍼 (upbit 인스턴스 사용)
def get_krw_balance_safe():
    try:
        bal = upbit.get_balance("KRW")
        return float(bal) if bal is not None else None
    except:
        return None

def get_coin_balance_safe(coin_symbol: str):
    try:
        bal = upbit.get_balance(coin_symbol)
        return float(bal) if bal is not None else None
    except:
        return None

# ─────────────────────────────────────────────────────────────────────────────
def get_top10_volatile_coins():
    """
    KRW 마켓 티커 중 거래량 상위 10개를 뽑고,
    (고가 - 저가)/시가 ≥ 0.05인 코인 리스트 반환
    """
    tickers = pyupbit.get_tickers(fiat="KRW")
    vol_list = []
    for t in tickers:
        try:
            df = pyupbit.get_ohlcv(t, interval="day", count=1)
            if df is None or df.empty:
                continue
            today = df.iloc[-1]
            open_price = today['open']
            high_price = today['high']
            low_price = today['low']
            volatility = (high_price - low_price) / open_price if open_price > 0 else 0

            ticker_info = pyupbit.get_market_detail(t)
            volume_24h = ticker_info.get('acc_trade_volume_24h', 0)

            vol_list.append({
                "ticker": t.replace("KRW-", ""),
                "volatility": volatility,
                "volume_24h": volume_24h
            })
        except:
            continue

    df_vol = pd.DataFrame(vol_list)
    if df_vol.empty:
        return []
    df_top10 = df_vol.sort_values(by="volume_24h", ascending=False).head(10)
    result = df_top10[df_top10["volatility"] >= 0.05]["ticker"].tolist()
    return result

# ─────────────────────────────────────────────────────────────────────────────
def run_auto_trade():
    """
    09:00 / 15:00 실행
    1) 상위 10개 변동성 코인 선별
    2) 공포·탐욕 지수 조회
    3) 신뢰도 ≥70% & F&G ≥50 조건 확인
    4) 세 전략 통합 → 시그널
    5) 분할 매수(3회, 가격 하락) / 분할 매도(익절 +5~10%, 손절 −3%)
    6) DB 저장 + Telegram 알림
    """
    # 1) 상위 10개 변동성 코인
    coins_to_check = get_top10_volatile_coins()
    if not coins_to_check:
        send_telegram("⚠️ 상위 10개 변동성 코인 조회 실패 또는 없음")
        return

    # 2) 공포·탐욕 지수 조회
    fng_value = get_fear_greed_index()
    if fng_value is None:
        send_telegram("⚠️ F&G 지수 조회 실패 → 매매 보류")
        return

    for coin in coins_to_check:
        try:
            # 2-1) 잔고 조회
            krw_balance = get_krw_balance_safe()
            coin_balance = get_coin_balance_safe(coin)

            # 2-2) 현재가 & 평균 매수가
            price = pyupbit.get_current_price(f"KRW-{coin}") or 0
            avg_price = 0.0
            balances = upbit.get_balances()
            for b in balances:
                if b.get("currency") == coin and b.get("avg_buy_price"):
                    avg_price = float(b["avg_buy_price"])
                    break

            # 3) 세 전략 통합 시그널
            signal = analyze_market_combined(coin)

            # 3-1) 신뢰도 ≥ 70% 체크
            if signal["confidence_score"] < 70:
                reaction = f"신뢰도 {signal['confidence_score']}% < 70% → 보류"
                record_trade(
                    coin             = coin,
                    decision         = signal["decision"],
                    percentage       = signal["percentage"],
                    confidence_score = signal["confidence_score"],
                    reason           = signal["reason"],
                    reaction         = reaction,
                    coin_balance     = coin_balance if coin_balance is not None else 0,
                    krw_balance      = krw_balance if krw_balance is not None else 0,
                    avg_price        = avg_price,
                    coin_price       = price
                )
                send_telegram(f"⚠️ [{coin}] 신뢰도 {signal['confidence_score']}% 미만 → 매매 보류")
                continue

            # 3-2) F&G 지수 ≥ 50 체크
            if fng_value < 50:
                reaction = f"F&G 지수 {fng_value} < 50 → 보류"
                record_trade(
                    coin             = coin,
                    decision         = signal["decision"],
                    percentage       = signal["percentage"],
                    confidence_score = signal["confidence_score"],
                    reason           = signal["reason"],
                    reaction         = reaction,
                    coin_balance     = coin_balance if coin_balance is not None else 0,
                    krw_balance      = krw_balance if krw_balance is not None else 0,
                    avg_price        = avg_price,
                    coin_price       = price
                )
                send_telegram(f"⚠️ [{coin}] F&G 지수 {fng_value} 미만 → 매매 보류")
                continue

            # 4) 매매 로직
            ratio = signal["percentage"] / 100.0
            reaction = ""

            # ── 분할 매수(3회) 조건: decision=buy & krw*ratio>5000 & 가격 하락 ──
            if signal["decision"] == "buy" and krw_balance is not None and krw_balance * ratio > 5000:
                df_yesterday = pyupbit.get_ohlcv(f"KRW-{coin}", interval="day", count=2)
                if df_yesterday is not None and len(df_yesterday) >= 2:
                    prev_close = df_yesterday['close'].iloc[-2]
                    if price < prev_close:
                        unit = (krw_balance * ratio * 0.9995) / 3
                        for i in range(3):
                            if IS_LIVE:
                                upbit.buy_market_order(f"KRW-{coin}", unit)
                            send_telegram(f"💸 [{coin}] {i+1}차 분할매수 - {unit:,.0f}원 (현재가 {price:,.0f} < 직전 종가 {prev_close:,.0f})")
                            time.sleep(1)
                        reaction = "3회 분할매수 실행"
                    else:
                        reaction = "가격 하락 아님 → 매수 보류"
                else:
                    reaction = "직전 종가 조회 실패 → 매수 보류"

            # ── 분할 매도(익절/손절) 조건: decision=sell & coin_balance>0 & avg_price>0 ──
            elif signal["decision"] == "sell" and coin_balance is not None and coin_balance > 0 and avg_price > 0:
                gain_rate = (price - avg_price) / avg_price
                # 손절: −3% 이하
                if gain_rate <= -0.03:
                    if IS_LIVE:
                        upbit.sell_market_order(f"KRW-{coin}", coin_balance)
                    send_telegram(f"📉 [{coin}] 손절 매도 - 전량 {coin_balance:.6f}개 (손실 {(gain_rate*100):.2f}%)")
                    reaction = "손절(−3%) 전량 매도"
                # 익절 +10% 이상: 전량 매도
                elif gain_rate >= 0.10:
                    if IS_LIVE:
                        upbit.sell_market_order(f"KRW-{coin}", coin_balance)
                    send_telegram(f"🚀 [{coin}] 익절 매도(+10%) - 전량 {coin_balance:.6f}개 (수익 {(gain_rate*100):.2f}%)")
                    reaction = "익절(+10%) 전량 매도"
                # 익절 +5% 이상: 절반 매도
                elif gain_rate >= 0.05:
                    half_qty = coin_balance / 2
                    if IS_LIVE:
                        upbit.sell_market_order(f"KRW-{coin}", half_qty)
                    send_telegram(f"📈 [{coin}] 부분 익절(+5%) - {half_qty:.6f}개 매도 (수익 {(gain_rate*100):.2f}%)")
                    reaction = "부분 익절(+5%) 반절 매도"
                else:
                    reaction = "익절/손절 조건 미달 → 보류"
            else:
                if signal["decision"] == "sell":
                    reaction = "코인 미보유 → 매도 보류"
                else:
                    reaction = "매수 조건 미달 → 보류"

            # 5) 거래 기록 저장
            record_trade(
                coin             = coin,
                decision         = signal["decision"],
                percentage       = signal["percentage"],
                confidence_score = signal["confidence_score"],
                reason           = signal["reason"],
                reaction         = reaction,
                coin_balance     = coin_balance if coin_balance is not None else 0,
                krw_balance      = krw_balance if krw_balance is not None else 0,
                avg_price        = avg_price,
                coin_price       = price
            )

            # 6) Telegram 알림: 거래 기록 완료
            send_telegram(
                f"✅ [{coin}] 거래 기록 완료({signal['decision'].upper()}): 신뢰도 {signal['confidence_score']}%\n"
                f"사유: {signal['reason']}\n반응: {reaction}"
            )
            time.sleep(2)

        except Exception as e:
            send_telegram(f"❌ [{coin}] 오류 발생: {e}")

# ─────────────────────────────────────────────────────────────────────────────
def run_scheduler():
    """
    매일 09:00, 15:00에 run_auto_trade() 호출
    """
    schedule.every().day.at("09:00").do(run_auto_trade)
    schedule.every().day.at("15:00").do(run_auto_trade)
    send_telegram("✅ 자동매매 스케줄 시작됨 (09:00 / 15:00)")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    run_scheduler()


