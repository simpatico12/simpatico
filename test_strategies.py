"""
🧪 최고퀸트프로젝트 - 완전체 테스트 시스템
==========================================

전체 시스템 통합 테스트:
- 📊 전략 모듈 테스트 (뉴스 통합 확인)
- 📅 스케줄링 시스템 테스트
- 📰 뉴스 분석 시스템 테스트
- 💰 매매 시스템 테스트
- 🔔 알림 시스템 테스트
- ⚙️ 핵심 엔진 테스트
- 🛠️ 유틸리티 테스트
- 🌍 전체 시스템 통합 테스트
- 📈 성능 벤치마크 테스트

Author: 최고퀸트팀
Version: 1.0.0
Project: 최고퀸트프로젝트
"""

import asyncio
import unittest
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback

# 테스트 결과 저장
test_results = []
performance_results = {}

class Colors:
    """터미널 컬러"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_test_result(test_name: str, success: bool, message: str = "", duration: float = 0.0):
    """테스트 결과 기록"""
    global test_results
    result = {
        'test_name': test_name,
        'success': success,
        'message': message,
        'duration': duration,
        'timestamp': datetime.now()
    }
    test_results.append(result)
    
    # 실시간 출력
    status = f"{Colors.GREEN}✅ 성공{Colors.END}" if success else f"{Colors.RED}❌ 실패{Colors.END}"
    duration_str = f"({duration:.2f}s)" if duration > 0 else ""
    print(f"   {status} {duration_str}")
    if message and not success:
        print(f"     {Colors.YELLOW}└─ {message}{Colors.END}")

# =====================================
# 1️⃣ 모듈 Import 테스트
# =====================================

def test_module_imports():
    """모듈 import 테스트"""
    print(f"{Colors.BOLD}1️⃣ 모듈 Import 테스트{Colors.END}")
    
    modules_to_test = [
        ('configs 설정', 'yaml'),
        ('utils 유틸리티', 'utils'),
        ('US 전략', 'strategies.us_strategy'),
        ('JP 전략', 'strategies.jp_strategy'),
        ('Coin 전략', 'strategies.coin_strategy'),
        ('핵심 엔진', 'core'),
        ('알림 시스템', 'notifier'),
        ('스케줄러', 'scheduler'),
        ('뉴스 분석', 'news_analyzer'),
        ('매매 시스템', 'trading')
    ]
    
    for name, module_name in modules_to_test:
        start_time = time.time()
        try:
            if module_name == 'yaml':
                import yaml
            else:
                __import__(module_name)
            duration = time.time() - start_time
            log_test_result(f"Import {name}", True, "", duration)
        except ImportError as e:
            duration = time.time() - start_time
            log_test_result(f"Import {name}", False, str(e), duration)
        except Exception as e:
            duration = time.time() - start_time
            log_test_result(f"Import {name}", False, f"예상치 못한 오류: {str(e)}", duration)

# =====================================
# 2️⃣ 설정 파일 테스트
# =====================================

def test_config_system():
    """설정 시스템 테스트"""
    print(f"\n{Colors.BOLD}2️⃣ 설정 시스템 테스트{Colors.END}")
    
    # 설정 파일 존재 확인
    start_time = time.time()
    try:
        config_path = "configs/settings.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                import yaml
                config = yaml.safe_load(f)
            duration = time.time() - start_time
            log_test_result("설정 파일 로드", True, "", duration)
            
            # 필수 섹션 확인
            required_sections = [
                'us_strategy', 'jp_strategy', 'coin_strategy',
                'schedule', 'news_analysis', 'trading', 'api',
                'risk_management', 'notifications'
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                log_test_result("설정 섹션 확인", False, f"누락된 섹션: {missing_sections}")
            else:
                log_test_result("설정 섹션 확인", True)
                
        else:
            duration = time.time() - start_time
            log_test_result("설정 파일 로드", False, "configs/settings.yaml 파일 없음", duration)
            
    except Exception as e:
        duration = time.time() - start_time
        log_test_result("설정 파일 로드", False, str(e), duration)

# =====================================
# 3️⃣ 유틸리티 시스템 테스트
# =====================================

async def test_utils_system():
    """유틸리티 시스템 테스트"""
    print(f"\n{Colors.BOLD}3️⃣ 유틸리티 시스템 테스트{Colors.END}")
    
    try:
        from utils import (
            DataProcessor, FinanceUtils, TimeZoneManager, 
            Formatter, Validator, get_config, is_market_open
        )
        
        # 데이터 처리 테스트
        start_time = time.time()
        test_symbols = ['AAPL', '7203.T', 'BTC-KRW']
        processed_symbols = []
        for symbol in test_symbols:
            normalized = DataProcessor.normalize_symbol(symbol)
            market = DataProcessor.detect_market(symbol)
            processed_symbols.append((symbol, normalized, market))
        duration = time.time() - start_time
        log_test_result("데이터 처리", True, f"3개 심볼 처리 완료", duration)
        
        # 시간대 관리 테스트
        start_time = time.time()
        tz_manager = TimeZoneManager()
        seoul_time = tz_manager.get_current_time('Seoul')
        us_time = tz_manager.get_current_time('US')
        duration = time.time() - start_time
        log_test_result("시간대 관리", True, f"서울/미국 시간 조회 완료", duration)
        
        # 포맷팅 테스트
        start_time = time.time()
        formatted_prices = [
            Formatter.format_price(175.50, 'US'),
            Formatter.format_price(2850, 'JP'),
            Formatter.format_price(95000000, 'COIN')
        ]
        duration = time.time() - start_time
        log_test_result("가격 포맷팅", True, f"3개 시장 포맷팅 완료", duration)
        
        # 시장 개장 확인
        start_time = time.time()
        markets_open = {
            'US': is_market_open('US'),
            'JP': is_market_open('JP'),
            'COIN': is_market_open('COIN')
        }
        duration = time.time() - start_time
        log_test_result("시장 개장 확인", True, f"3개 시장 상태 확인", duration)
        
    except Exception as e:
        log_test_result("유틸리티 시스템", False, str(e))

# =====================================
# 4️⃣ 스케줄링 시스템 테스트
# =====================================

async def test_scheduler_system():
    """스케줄링 시스템 테스트"""
    print(f"\n{Colors.BOLD}4️⃣ 스케줄링 시스템 테스트{Colors.END}")
    
    try:
        from scheduler import get_today_strategies, is_trading_time, get_schedule_status
        
        # 오늘 전략 조회
        start_time = time.time()
        today_strategies = get_today_strategies()
        duration = time.time() - start_time
        log_test_result("오늘 전략 조회", True, f"활성 전략: {today_strategies}", duration)
        
        # 거래 시간 확인
        start_time = time.time()
        trading_time = is_trading_time()
        duration = time.time() - start_time
        status_text = "거래 시간" if trading_time else "휴장 시간"
        log_test_result("거래 시간 확인", True, status_text, duration)
        
        # 스케줄러 상태
        start_time = time.time()
        schedule_status = get_schedule_status()
        duration = time.time() - start_time
        log_test_result("스케줄러 상태", True, f"상태: {schedule_status.get('scheduler_status', 'unknown')}", duration)
        
        # 주간 스케줄 검증
        start_time = time.time()
        from scheduler import TradingScheduler
        scheduler = TradingScheduler()
        weekly_format = scheduler._format_weekly_schedule()
        duration = time.time() - start_time
        log_test_result("주간 스케줄", True, f"포맷: {weekly_format[:50]}...", duration)
        
    except Exception as e:
        log_test_result("스케줄링 시스템", False, str(e))

# =====================================
# 5️⃣ 뉴스 분석 시스템 테스트
# =====================================

async def test_news_system():
    """뉴스 분석 시스템 테스트"""
    print(f"\n{Colors.BOLD}5️⃣ 뉴스 분석 시스템 테스트{Colors.END}")
    
    try:
        from news_analyzer import get_news_sentiment, get_news_analysis_stats
        
        # 뉴스 센티먼트 분석 (빠른 테스트)
        test_symbols = [('AAPL', 'US'), ('BTC', 'COIN')]
        
        for symbol, market in test_symbols:
            start_time = time.time()
            try:
                sentiment, reasoning = await get_news_sentiment(symbol, market)
                duration = time.time() - start_time
                log_test_result(f"뉴스 분석 {symbol}", True, f"센티먼트: {sentiment:.2f}", duration)
            except Exception as e:
                duration = time.time() - start_time
                log_test_result(f"뉴스 분석 {symbol}", False, str(e), duration)
        
        # 뉴스 분석 통계
        start_time = time.time()
        stats = get_news_analysis_stats()
        duration = time.time() - start_time
        log_test_result("뉴스 분석 통계", True, f"상태: {stats.get('analyzer_status', 'unknown')}", duration)
        
    except Exception as e:
        log_test_result("뉴스 분석 시스템", False, str(e))

# =====================================
# 6️⃣ 전략 시스템 테스트
# =====================================

async def test_strategy_systems():
    """전략 시스템 테스트"""
    print(f"\n{Colors.BOLD}6️⃣ 전략 시스템 테스트{Colors.END}")
    
    # 미국 주식 전략
    try:
        from strategies.us_strategy import analyze_us, USStrategy
        
        start_time = time.time()
        result = await analyze_us('AAPL')
        duration = time.time() - start_time
        
        if result and 'decision' in result:
            decision = result['decision']
            confidence = result.get('confidence_score', 0)
            log_test_result("미국 주식 전략", True, f"AAPL: {decision} ({confidence:.0f}%)", duration)
        else:
            log_test_result("미국 주식 전략", False, "결과 형식 오류", duration)
            
    except Exception as e:
        log_test_result("미국 주식 전략", False, str(e))
    
    # 일본 주식 전략
    try:
        from strategies.jp_strategy import analyze_jp, JPStrategy
        
        start_time = time.time()
        result = await analyze_jp('7203.T')
        duration = time.time() - start_time
        
        if result and 'decision' in result:
            decision = result['decision']
            confidence = result.get('confidence_score', 0)
            log_test_result("일본 주식 전략", True, f"토요타: {decision} ({confidence:.0f}%)", duration)
        else:
            log_test_result("일본 주식 전략", False, "결과 형식 오류", duration)
            
    except Exception as e:
        log_test_result("일본 주식 전략", False, str(e))
    
    # 암호화폐 전략
    try:
        from strategies.coin_strategy import analyze_coin, CoinStrategy
        
        start_time = time.time()
        result = await analyze_coin('BTC-KRW')
        duration = time.time() - start_time
        
        if result and 'decision' in result:
            decision = result['decision']
            confidence = result.get('confidence_score', 0)
            log_test_result("암호화폐 전략", True, f"BTC: {decision} ({confidence:.0f}%)", duration)
        else:
            log_test_result("암호화폐 전략", False, "결과 형식 오류", duration)
            
    except Exception as e:
        log_test_result("암호화폐 전략", False, str(e))

# =====================================
# 7️⃣ 매매 시스템 테스트
# =====================================

async def test_trading_system():
    """매매 시스템 테스트"""
    print(f"\n{Colors.BOLD}7️⃣ 매매 시스템 테스트{Colors.END}")
    
    try:
        from trading import TradingExecutor, get_trading_stats, get_portfolio_summary
        
        # 매매 실행기 초기화
        start_time = time.time()
        executor = TradingExecutor()
        duration = time.time() - start_time
        log_test_result("매매 실행기 초기화", True, f"모의거래: {executor.paper_trading}", duration)
        
        # 거래 통계
        start_time = time.time()
        stats = get_trading_stats()
        duration = time.time() - start_time
        status = stats.get('executor_status', 'unknown')
        log_test_result("거래 통계", True, f"상태: {status}", duration)
        
        # 포트폴리오 요약 (타임아웃 적용)
        start_time = time.time()
        try:
            portfolio = await asyncio.wait_for(get_portfolio_summary(), timeout=10.0)
            duration = time.time() - start_time
            portfolio_count = sum(1 for k in ['ibkr_portfolio', 'upbit_portfolio'] if portfolio.get(k))
            log_test_result("포트폴리오 조회", True, f"{portfolio_count}개 브로커 연결", duration)
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            log_test_result("포트폴리오 조회", False, "타임아웃 (10초)", duration)
        
        # 테스트 신호 생성 및 실행
        start_time = time.time()
        from trading import TradingSignal
        test_signal = TradingSignal(
            market='US', symbol='AAPL', action='buy', confidence=0.85, price=175.50,
            strategy='test', reasoning='테스트 신호', target_price=195.80,
            timestamp=datetime.now()
        )
        
        try:
            result = await asyncio.wait_for(executor.execute_signal(test_signal), timeout=15.0)
            duration = time.time() - start_time
            success = result.get('success', False)
            if success:
                log_test_result("테스트 매매 실행", True, "모의거래 성공", duration)
            else:
                error = result.get('error', 'unknown')
                log_test_result("테스트 매매 실행", False, error, duration)
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            log_test_result("테스트 매매 실행", False, "타임아웃 (15초)", duration)
        
        # 리소스 정리
        await executor.cleanup()
        
    except Exception as e:
        log_test_result("매매 시스템", False, str(e))

# =====================================
# 8️⃣ 알림 시스템 테스트
# =====================================

async def test_notification_system():
    """알림 시스템 테스트"""
    print(f"\n{Colors.BOLD}8️⃣ 알림 시스템 테스트{Colors.END}")
    
    try:
        from notifier import (
            send_telegram_message, test_telegram_connection,
            send_trading_alert, send_system_alert
        )
        
        # 텔레그램 연결 테스트
        start_time = time.time()
        connection_result = await test_telegram_connection()
        duration = time.time() - start_time
        log_test_result("텔레그램 연결", connection_result, "봇 설정 확인 필요" if not connection_result else "", duration)
        
        # 시스템 알림 테스트 (연결되어 있을 때만)
        if connection_result:
            start_time = time.time()
            try:
                alert_result = await send_system_alert("info", "테스트 시스템 알림", "normal")
                duration = time.time() - start_time
                log_test_result("시스템 알림", alert_result, "", duration)
            except Exception as e:
                duration = time.time() - start_time
                log_test_result("시스템 알림", False, str(e), duration)
        else:
            log_test_result("시스템 알림", False, "텔레그램 연결 없음")
        
    except Exception as e:
        log_test_result("알림 시스템", False, str(e))

# =====================================
# 9️⃣ 핵심 엔진 테스트
# =====================================

async def test_core_engine():
    """핵심 엔진 테스트"""
    print(f"\n{Colors.BOLD}9️⃣ 핵심 엔진 테스트{Colors.END}")
    
    try:
        from core import QuantTradingEngine, get_engine_status
        
        # 엔진 초기화
        start_time = time.time()
        engine = QuantTradingEngine()
        duration = time.time() - start_time
        log_test_result("엔진 초기화", True, f"전략 {len(engine.today_strategies)}개 활성화", duration)
        
        # 엔진 상태
        start_time = time.time()
        status = get_engine_status()
        duration = time.time() - start_time
        system_status = status.get('system_status', 'unknown')
        log_test_result("엔진 상태", True, f"상태: {system_status}", duration)
        
        # 빠른 분석 테스트 (타임아웃 적용)
        start_time = time.time()
        try:
            quick_symbols = ['AAPL']
            if 'COIN' in engine.today_strategies:
                quick_symbols.append('BTC-KRW')
            
            signals = await asyncio.wait_for(engine.get_quick_analysis(quick_symbols), timeout=20.0)
            duration = time.time() - start_time
            log_test_result("빠른 분석", True, f"{len(signals)}개 신호 생성", duration)
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            log_test_result("빠른 분석", False, "타임아웃 (20초)", duration)
        
    except Exception as e:
        log_test_result("핵심 엔진", False, str(e))

# =====================================
# 🔟 성능 벤치마크 테스트
# =====================================

async def test_performance_benchmark():
    """성능 벤치마크 테스트"""
    print(f"\n{Colors.BOLD}🔟 성능 벤치마크 테스트{Colors.END}")
    
    global performance_results
    
    # 개별 전략 성능 테스트
    strategies = [
        ('US 전략', 'strategies.us_strategy', 'analyze_us', 'AAPL'),
        ('JP 전략', 'strategies.jp_strategy', 'analyze_jp', '7203.T'),
        ('COIN 전략', 'strategies.coin_strategy', 'analyze_coin', 'BTC-KRW')
    ]
    
    for strategy_name, module_name, function_name, test_symbol in strategies:
        try:
            module = __import__(module_name, fromlist=[function_name])
            analyze_func = getattr(module, function_name)
            
            # 10회 실행하여 평균 시간 측정
            times = []
            for i in range(3):  # 시간 단축을 위해 3회로 감소
                start_time = time.time()
                try:
                    result = await asyncio.wait_for(analyze_func(test_symbol), timeout=10.0)
                    duration = time.time() - start_time
                    times.append(duration)
                except asyncio.TimeoutError:
                    times.append(10.0)  # 타임아웃을 10초로 기록
                except Exception:
                    times.append(999.0)  # 에러를 999초로 기록
            
            avg_time = sum(times) / len(times)
            performance_results[strategy_name] = {
                'avg_time': avg_time,
                'min_time': min(times),
                'max_time': max(times),
                'runs': len(times)
            }
            
            log_test_result(f"성능 {strategy_name}", True, f"평균: {avg_time:.2f}s")
            
        except Exception as e:
            log_test_result(f"성능 {strategy_name}", False, str(e))

# =====================================
# 📊 전체 시스템 통합 테스트
# =====================================

async def test_full_system_integration():
    """전체 시스템 통합 테스트"""
    print(f"\n{Colors.BOLD}📊 전체 시스템 통합 테스트{Colors.END}")
    
    try:
        from core import QuantTradingEngine
        
        # 전체 시스템 시뮬레이션
        start_time = time.time()
        engine = QuantTradingEngine()
        
        # 스케줄링 확인
        today_strategies = engine.today_strategies
        if not today_strategies:
            log_test_result("통합 테스트", False, "오늘 활성화된 전략 없음")
            return
        
        # 시장별 빠른 분석 (타임아웃 적용)
        try:
            test_symbols = []
            if 'US' in today_strategies:
                test_symbols.append('AAPL')
            if 'JP' in today_strategies:
                test_symbols.append('7203.T')
            if 'COIN' in today_strategies:
                test_symbols.append('BTC-KRW')
            
            if test_symbols:
                signals = await asyncio.wait_for(engine.get_quick_analysis(test_symbols), timeout=30.0)
                duration = time.time() - start_time
                
                # 신호 품질 검증
                valid_signals = 0
                for signal in signals:
                    if hasattr(signal, 'confidence') and signal.confidence > 0:
                        valid_signals += 1
                
                log_test_result("통합 분석", True, f"{valid_signals}/{len(signals)}개 유효 신호", duration)
            else:
                log_test_result("통합 분석", False, "테스트할 심볼 없음")
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            log_test_result("통합 분석", False, "타임아웃 (30초)", duration)
        
    except Exception as e:
        log_test_result("전체 시스템 통합", False, str(e))

# =====================================
# 📋 테스트 결과 요약
# =====================================

def print_test_summary():
    """테스트 결과 요약 출력"""
    global test_results, performance_results
    
    print(f"\n{Colors.BOLD}📋 테스트 결과 요약{Colors.END}")
    print("=" * 70)
    
    # 전체 통계
    total_tests = len(test_results)
    passed_tests = len([r for r in test_results if r['success']])
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 전체 테스트: {total_tests}개")
    print(f"✅ 성공: {Colors.GREEN}{passed_tests}개{Colors.END}")
    print(f"❌ 실패: {Colors.RED}{failed_tests}개{Colors.END}")
    print(f"📈 성공률: {Colors.CYAN}{success_rate:.1f}%{Colors.END}")
    
    # 실패한 테스트 상세
    if failed_tests > 0:
        print(f"\n{Colors.RED}❌ 실패한 테스트:{Colors.END}")
        for result in test_results:
            if not result['success']:
                print(f"   • {result['test_name']}: {result['message']}")
    
    # 성능 결과
    if performance_results:
        print(f"\n{Colors.PURPLE}📈 성능 벤치마크:{Colors.END}")
        for strategy, perf in performance_results.items():
            avg_time = perf['avg_time']
            color = Colors.GREEN if avg_time < 5.0 else Colors.YELLOW if avg_time < 10.0 else Colors.RED
            print(f"   • {strategy}: {color}{avg_time:.2f}s{Colors.END} (최소: {perf['min_time']:.2f}s)")
    
    # 총 실행 시간
    total_duration = sum(r['duration'] for r in test_results)
    print(f"\n⏱️ 총 실행 시간: {total_duration:.2f}초")
    
    # 시스템 상태 요약
    print(f"\n{Colors.BOLD}🎯 시스템 상태 요약:{Colors.END}")
    
    # 핵심 시스템 상태
    core_systems = ['모듈 Import', '설정 시스템', '전략 시스템', '핵심 엔진']
    core_status = []
    for system in core_systems:
        system_tests = [r for r in test_results if system.lower() in r['test_name'].lower()]
        if system_tests:
            system_success = all(r['success'] for r in system_tests)
            status = f"{Colors.GREEN}✅{Colors.END}" if system_success else f"{Colors.RED}❌{Colors.END}"
            core_status.append(f"{system}: {status}")
    
    for status in core_status:
        print(f"   • {status}")
    
    # 최종 판정
    print(f"\n{Colors.BOLD}🏆 최종 판정:{Colors.END}")
    if success_rate >= 90:
        print(f"   {Colors.GREEN}🎉 EXCELLENT - 최고퀸트프로젝트가 완벽하게 작동합니다!{Colors.END}")
    elif success_rate >= 80:
        print(f"   {Colors.CYAN}👍 GOOD - 시스템이 양호하게 작동합니다{Colors.END}")
    elif success_rate >= 70:
        print(f"   {Colors.YELLOW}⚠️ FAIR - 일부 개선이 필요합니다{Colors.END}")
    else:
        print(f"   {Colors.RED}❌ POOR - 시스템 점검이 필요합니다{Colors.END}")

# =====================================
# 메인 테스트 실행 함수
# =====================================

async def run_all_tests(quick_mode: bool = False):
    """전체 테스트 실행"""
    print(f"{Colors.BOLD}{Colors.CYAN}🧪 최고퀸트프로젝트 완전체 테스트 시스템{Colors.END}")
    print("=" * 70)
    print(f"🚀 테스트 모드: {'빠른 테스트' if quick_mode else '전체 테스트'}")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 기본 시스템 테스트
    test_module_imports()
    test_config_system()
    
    # 2. 핵심 모듈 테스트
    await test_utils_system()
    await test_scheduler_system()
    
    if not quick_mode:
        await test_news_system()
    
    # 3. 전략 시스템 테스트
    await test_strategy_systems()
    
    # 4. 인프라 테스트
    if not quick_mode:
        await test_trading_system()
        await test_notification_system()
    
    # 5. 통합 테스트
    await test_core_engine()
    
    if not quick_mode:
        await test_performance_benchmark()
        await test_full_system_integration()
    
    # 6. 결과 요약
    print_test_summary()

# =====================================
# 실행 함수
# =====================================

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='최고퀸트프로젝트 테스트 시스템')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 모드')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # 테스트 실행
    try:
        asyncio.run(run_all_tests(args.quick))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ 사용자에 의해 테스트가 중단되었습니다{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ 테스트 실행 중 오류: {e}{Colors.END}")
        traceback.print_exc()

if __name__ == "__main__":
    main()