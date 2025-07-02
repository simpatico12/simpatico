#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 최고퀸트프로젝트 - 파일 기반 전략 테스트 시스템
====================================================================

3개 전략 파일들을 동적으로 로드하고 테스트:
- 📊 jp_strategy.py (일본 주식 전략)
- 📈 us_strategy.py (미국 주식 전략)  
- 🪙 coin_strategy.py (암호화폐 전략)

테스트 항목:
- 📁 파일 기반 동적 로딩
- 🔧 설정 파일 연동 검증
- ⚡ 비동기 함수 호출 및 타임아웃
- 📊 전략별 결과 형식 검증
- 🎯 성능 벤치마크
- 🛡️ 에러 핸들링 및 복구

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
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import traceback
from pathlib import Path
import json

# 테스트 결과 저장
test_results = []
performance_results = {}
strategy_results = {}

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
# 🎯 전략 파일 정보 정의
# =====================================

STRATEGY_FILES = {
    'jp_strategy': {
        'file_path': 'strategies/jp_strategy.py',
        'backup_paths': ['jp_strategy.py', './jp_strategy.py'],
        'analyze_function': 'analyze_jp',
        'strategy_class': 'JPStrategy',
        'test_symbol': '7203.T',
        'expected_keys': ['decision', 'confidence_score', 'reasoning', 'current_price'],
        'config_type': 'ConfigLoader',
        'description': '일본 주식 전략 (엔화 기반 + 기술분석)',
        'timeout': 30.0,
        'async_function': True
    },
    'us_strategy': {
        'file_path': 'strategies/us_strategy.py',
        'backup_paths': ['us_strategy.py', './us_strategy.py'],
        'analyze_function': 'analyze_us',
        'strategy_class': 'AdvancedUSStrategy',
        'test_symbol': 'AAPL',
        'expected_keys': ['decision', 'confidence_score', 'reasoning', 'target_price'],
        'config_type': 'ConfigManager',
        'description': '미국 주식 전략 (4가지 전략 융합 + VIX)',
        'timeout': 30.0,
        'async_function': True
    },
    'coin_strategy': {
        'file_path': 'strategies/coin_strategy.py',
        'backup_paths': ['coin_strategy.py', './coin_strategy.py'],
        'analyze_function': 'analyze_single_coin',
        'strategy_class': 'UltimateCoinStrategy',
        'test_symbol': 'KRW-BTC',
        'expected_keys': ['decision', 'confidence_percent', 'total_score', 'scores'],
        'config_type': 'ConfigManager',
        'description': '암호화폐 전략 (6단계 필터링 + AI 품질평가)',
        'timeout': 45.0,
        'async_function': True
    }
}

# =====================================
# 📁 파일 기반 동적 로더
# =====================================

class StrategyFileLoader:
    """전략 파일 동적 로더"""
    
    def __init__(self):
        self.loaded_modules = {}
        self.loaded_functions = {}
        self.loaded_classes = {}
    
    def find_strategy_file(self, strategy_name: str) -> Optional[str]:
        """전략 파일 경로 찾기"""
        strategy_info = STRATEGY_FILES.get(strategy_name)
        if not strategy_info:
            return None
        
        # 우선순위별로 파일 경로 확인
        all_paths = [strategy_info['file_path']] + strategy_info['backup_paths']
        
        for path in all_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_strategy_module(self, strategy_name: str) -> Optional[Any]:
        """전략 모듈 동적 로드"""
        try:
            file_path = self.find_strategy_file(strategy_name)
            if not file_path:
                raise FileNotFoundError(f"{strategy_name} 파일을 찾을 수 없습니다")
            
            # 모듈 이름 생성
            module_name = f"dynamic_{strategy_name}_{int(time.time())}"
            
            # 파일을 모듈로 로드
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"{file_path} 스펙 생성 실패")
            
            module = importlib.util.module_from_spec(spec)
            
            # sys.modules에 추가 (순환 import 방지)
            sys.modules[module_name] = module
            
            # 모듈 실행
            spec.loader.exec_module(module)
            
            self.loaded_modules[strategy_name] = module
            return module
            
        except Exception as e:
            print(f"❌ {strategy_name} 모듈 로드 실패: {e}")
            return None
    
    def get_strategy_function(self, strategy_name: str) -> Optional[Callable]:
        """전략 분석 함수 추출"""
        try:
            if strategy_name not in self.loaded_modules:
                module = self.load_strategy_module(strategy_name)
                if not module:
                    return None
            else:
                module = self.loaded_modules[strategy_name]
            
            strategy_info = STRATEGY_FILES[strategy_name]
            function_name = strategy_info['analyze_function']
            
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                self.loaded_functions[strategy_name] = func
                return func
            else:
                raise AttributeError(f"{function_name} 함수가 {strategy_name}에 없습니다")
                
        except Exception as e:
            print(f"❌ {strategy_name} 함수 추출 실패: {e}")
            return None
    
    def get_strategy_class(self, strategy_name: str) -> Optional[type]:
        """전략 클래스 추출"""
        try:
            if strategy_name not in self.loaded_modules:
                module = self.load_strategy_module(strategy_name)
                if not module:
                    return None
            else:
                module = self.loaded_modules[strategy_name]
            
            strategy_info = STRATEGY_FILES[strategy_name]
            class_name = strategy_info['strategy_class']
            
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                self.loaded_classes[strategy_name] = cls
                return cls
            else:
                raise AttributeError(f"{class_name} 클래스가 {strategy_name}에 없습니다")
                
        except Exception as e:
            print(f"❌ {strategy_name} 클래스 추출 실패: {e}")
            return None
    
    def check_function_signature(self, strategy_name: str) -> Dict[str, Any]:
        """함수 시그니처 검증"""
        try:
            func = self.get_strategy_function(strategy_name)
            if not func:
                return {'valid': False, 'error': '함수 로드 실패'}
            
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            is_async = inspect.iscoroutinefunction(func)
            
            strategy_info = STRATEGY_FILES[strategy_name]
            expected_async = strategy_info['async_function']
            
            return {
                'valid': True,
                'is_async': is_async,
                'expected_async': expected_async,
                'parameters': params,
                'signature_match': is_async == expected_async,
                'param_count': len(params)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

# =====================================
# 🔧 설정 시스템 테스트
# =====================================

def test_strategy_file_loading():
    """전략 파일 로딩 테스트"""
    print(f"{Colors.BOLD}1️⃣ 전략 파일 로딩 테스트{Colors.END}")
    
    loader = StrategyFileLoader()
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n📁 {strategy_name} 테스트:")
        
        # 파일 존재 확인
        start_time = time.time()
        file_path = loader.find_strategy_file(strategy_name)
        duration = time.time() - start_time
        
        if file_path:
            log_test_result(f"{strategy_name} 파일 찾기", True, f"경로: {file_path}", duration)
        else:
            log_test_result(f"{strategy_name} 파일 찾기", False, "파일 없음", duration)
            continue
        
        # 모듈 로드
        start_time = time.time()
        module = loader.load_strategy_module(strategy_name)
        duration = time.time() - start_time
        
        if module:
            log_test_result(f"{strategy_name} 모듈 로드", True, "", duration)
        else:
            log_test_result(f"{strategy_name} 모듈 로드", False, "모듈 로드 실패", duration)
            continue
        
        # 함수 추출
        start_time = time.time()
        func = loader.get_strategy_function(strategy_name)
        duration = time.time() - start_time
        
        if func:
            log_test_result(f"{strategy_name} 함수 추출", True, f"함수: {strategy_info['analyze_function']}", duration)
        else:
            log_test_result(f"{strategy_name} 함수 추출", False, "함수 추출 실패", duration)
        
        # 클래스 추출
        start_time = time.time()
        cls = loader.get_strategy_class(strategy_name)
        duration = time.time() - start_time
        
        if cls:
            log_test_result(f"{strategy_name} 클래스 추출", True, f"클래스: {strategy_info['strategy_class']}", duration)
        else:
            log_test_result(f"{strategy_name} 클래스 추출", False, "클래스 추출 실패", duration)
        
        # 함수 시그니처 검증
        start_time = time.time()
        sig_info = loader.check_function_signature(strategy_name)
        duration = time.time() - start_time
        
        if sig_info['valid'] and sig_info['signature_match']:
            async_status = "비동기" if sig_info['is_async'] else "동기"
            log_test_result(f"{strategy_name} 시그니처 검증", True, f"{async_status}, 파라미터 {sig_info['param_count']}개", duration)
        else:
            error_msg = sig_info.get('error', '시그니처 불일치')
            log_test_result(f"{strategy_name} 시그니처 검증", False, error_msg, duration)
    
    return loader

# =====================================
# 🔧 설정 파일 연동 테스트
# =====================================

def test_config_integration(loader: StrategyFileLoader):
    """설정 파일 연동 테스트"""
    print(f"\n{Colors.BOLD}2️⃣ 설정 파일 연동 테스트{Colors.END}")
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n🔧 {strategy_name} 설정 테스트:")
        
        try:
            # 모듈에서 설정 관련 클래스 찾기
            start_time = time.time()
            module = loader.loaded_modules.get(strategy_name)
            if not module:
                log_test_result(f"{strategy_name} 설정 테스트", False, "모듈 없음")
                continue
            
            config_type = strategy_info['config_type']
            config_class = None
            
            # 설정 클래스 찾기
            if hasattr(module, config_type):
                config_class = getattr(module, config_type)
            elif hasattr(module, 'ConfigLoader'):
                config_class = getattr(module, 'ConfigLoader')
            elif hasattr(module, 'ConfigManager'):
                config_class = getattr(module, 'ConfigManager')
            
            duration = time.time() - start_time
            
            if config_class:
                log_test_result(f"{strategy_name} 설정 클래스 발견", True, f"클래스: {config_class.__name__}", duration)
                
                # 설정 인스턴스 생성 테스트
                start_time = time.time()
                try:
                    if config_type == 'ConfigLoader':
                        config_instance = config_class()
                    else:
                        config_instance = config_class()
                    
                    duration = time.time() - start_time
                    log_test_result(f"{strategy_name} 설정 인스턴스 생성", True, "", duration)
                    
                    # 설정 메서드 확인
                    methods_to_check = ['get', 'get_config', 'load_config', 'get_section']
                    available_methods = []
                    
                    for method in methods_to_check:
                        if hasattr(config_instance, method):
                            available_methods.append(method)
                    
                    if available_methods:
                        log_test_result(f"{strategy_name} 설정 메서드 확인", True, f"메서드: {', '.join(available_methods)}")
                    else:
                        log_test_result(f"{strategy_name} 설정 메서드 확인", False, "설정 메서드 없음")
                        
                except Exception as e:
                    duration = time.time() - start_time
                    log_test_result(f"{strategy_name} 설정 인스턴스 생성", False, str(e), duration)
            else:
                log_test_result(f"{strategy_name} 설정 클래스 발견", False, f"{config_type} 클래스 없음", duration)
                
        except Exception as e:
            log_test_result(f"{strategy_name} 설정 테스트", False, str(e))

# =====================================
# ⚡ 비동기 함수 호출 테스트
# =====================================

async def test_async_function_calls(loader: StrategyFileLoader):
    """비동기 함수 호출 테스트"""
    print(f"\n{Colors.BOLD}3️⃣ 비동기 함수 호출 테스트{Colors.END}")
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n⚡ {strategy_name} 비동기 테스트:")
        
        try:
            func = loader.get_strategy_function(strategy_name)
            if not func:
                log_test_result(f"{strategy_name} 함수 호출", False, "함수 없음")
                continue
            
            test_symbol = strategy_info['test_symbol']
            timeout = strategy_info['timeout']
            
            # 타임아웃 적용하여 함수 호출
            start_time = time.time()
            try:
                if inspect.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(test_symbol), timeout=timeout)
                else:
                    # 동기 함수인 경우 스레드에서 실행
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(func, test_symbol)
                        result = await asyncio.wait_for(
                            asyncio.wrap_future(future), timeout=timeout
                        )
                
                duration = time.time() - start_time
                
                if result:
                    log_test_result(f"{strategy_name} 함수 호출", True, f"심볼: {test_symbol}", duration)
                    
                    # 결과 저장
                    strategy_results[strategy_name] = {
                        'result': result,
                        'duration': duration,
                        'symbol': test_symbol,
                        'success': True
                    }
                else:
                    log_test_result(f"{strategy_name} 함수 호출", False, "결과 없음", duration)
                    
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                log_test_result(f"{strategy_name} 함수 호출", False, f"타임아웃 ({timeout}초)", duration)
                
            except Exception as e:
                duration = time.time() - start_time
                log_test_result(f"{strategy_name} 함수 호출", False, str(e), duration)
                
        except Exception as e:
            log_test_result(f"{strategy_name} 비동기 테스트", False, str(e))

# =====================================
# 📊 결과 형식 검증 테스트
# =====================================

def test_result_format_validation():
    """결과 형식 검증 테스트"""
    print(f"\n{Colors.BOLD}4️⃣ 결과 형식 검증 테스트{Colors.END}")
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n📊 {strategy_name} 결과 검증:")
        
        if strategy_name not in strategy_results:
            log_test_result(f"{strategy_name} 결과 검증", False, "결과 없음")
            continue
        
        result_data = strategy_results[strategy_name]
        result = result_data['result']
        expected_keys = strategy_info['expected_keys']
        
        start_time = time.time()
        
        try:
            # 결과가 딕셔너리인지 확인
            if not isinstance(result, dict):
                log_test_result(f"{strategy_name} 타입 검증", False, f"예상: dict, 실제: {type(result)}")
                continue
            
            # 필수 키 존재 확인
            missing_keys = []
            present_keys = []
            
            for key in expected_keys:
                if key in result:
                    present_keys.append(key)
                else:
                    missing_keys.append(key)
            
            duration = time.time() - start_time
            
            if not missing_keys:
                log_test_result(f"{strategy_name} 필수 키 검증", True, f"키 {len(present_keys)}개 확인", duration)
            else:
                log_test_result(f"{strategy_name} 필수 키 검증", False, f"누락 키: {missing_keys}", duration)
            
            # 데이터 유효성 검증
            start_time = time.time()
            validation_results = []
            
            # decision 검증
            if 'decision' in result:
                decision = result['decision']
                if decision in ['buy', 'sell', 'hold']:
                    validation_results.append(f"decision: {decision} ✓")
                else:
                    validation_results.append(f"decision: {decision} ✗")
            
            # confidence 검증
            confidence_keys = ['confidence_score', 'confidence_percent', 'confidence']
            for conf_key in confidence_keys:
                if conf_key in result:
                    confidence = result[conf_key]
                    if isinstance(confidence, (int, float)) and 0 <= confidence <= 100:
                        validation_results.append(f"{conf_key}: {confidence} ✓")
                    else:
                        validation_results.append(f"{conf_key}: {confidence} ✗")
                    break
            
            # reasoning 검증
            if 'reasoning' in result:
                reasoning = result['reasoning']
                if isinstance(reasoning, str) and len(reasoning) > 0:
                    validation_results.append(f"reasoning: {len(reasoning)}글자 ✓")
                else:
                    validation_results.append("reasoning: 비어있음 ✗")
            
            duration = time.time() - start_time
            valid_count = len([r for r in validation_results if '✓' in r])
            total_count = len(validation_results)
            
            if valid_count == total_count:
                log_test_result(f"{strategy_name} 데이터 유효성 검증", True, f"{valid_count}/{total_count} 통과", duration)
            else:
                log_test_result(f"{strategy_name} 데이터 유효성 검증", False, f"{valid_count}/{total_count} 통과", duration)
                for validation in validation_results:
                    if '✗' in validation:
                        print(f"     {Colors.YELLOW}└─ {validation}{Colors.END}")
            
            # 결과 요약 저장
            strategy_results[strategy_name]['validation'] = {
                'format_valid': not missing_keys,
                'data_valid': valid_count == total_count,
                'missing_keys': missing_keys,
                'validation_details': validation_results
            }
            
        except Exception as e:
            duration = time.time() - start_time
            log_test_result(f"{strategy_name} 결과 검증", False, str(e), duration)

# =====================================
# 🎯 성능 벤치마크 테스트
# =====================================

async def test_performance_benchmark(loader: StrategyFileLoader):
    """성능 벤치마크 테스트"""
    print(f"\n{Colors.BOLD}5️⃣ 성능 벤치마크 테스트{Colors.END}")
    
    global performance_results
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n🎯 {strategy_name} 성능 테스트:")
        
        try:
            func = loader.get_strategy_function(strategy_name)
            if not func:
                log_test_result(f"{strategy_name} 성능 테스트", False, "함수 없음")
                continue
            
            test_symbol = strategy_info['test_symbol']
            timeout = min(strategy_info['timeout'], 15.0)  # 벤치마크는 15초 제한
            
            # 3회 실행하여 평균 시간 측정
            times = []
            success_count = 0
            
            for i in range(3):
                start_time = time.time()
                try:
                    if inspect.iscoroutinefunction(func):
                        result = await asyncio.wait_for(func(test_symbol), timeout=timeout)
                    else:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, test_symbol)
                            result = await asyncio.wait_for(
                                asyncio.wrap_future(future), timeout=timeout
                            )
                    
                    duration = time.time() - start_time
                    times.append(duration)
                    
                    if result:
                        success_count += 1
                        
                except asyncio.TimeoutError:
                    times.append(timeout)
                except Exception:
                    times.append(999.0)  # 에러를 999초로 기록
                
                # 테스트 간 간격
                await asyncio.sleep(0.5)
            
            # 성능 메트릭 계산
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            success_rate = success_count / 3 * 100
            
            performance_results[strategy_name] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'success_rate': success_rate,
                'runs': 3,
                'timeout': timeout
            }
            
            # 성능 등급 결정
            if avg_time < 5.0 and success_rate >= 100:
                grade = "우수"
                color = Colors.GREEN
            elif avg_time < 10.0 and success_rate >= 66:
                grade = "양호"
                color = Colors.CYAN
            elif avg_time < 15.0 and success_rate >= 33:
                grade = "보통"
                color = Colors.YELLOW
            else:
                grade = "개선 필요"
                color = Colors.RED
            
            log_test_result(f"{strategy_name} 성능 테스트", success_rate > 0, 
                          f"{color}{grade}{Colors.END} - 평균: {avg_time:.2f}s, 성공률: {success_rate:.0f}%")
            
        except Exception as e:
            log_test_result(f"{strategy_name} 성능 테스트", False, str(e))

# =====================================
# 🛡️ 에러 핸들링 테스트
# =====================================

async def test_error_handling(loader: StrategyFileLoader):
    """에러 핸들링 테스트"""
    print(f"\n{Colors.BOLD}6️⃣ 에러 핸들링 테스트{Colors.END}")
    
    error_test_cases = [
        ('잘못된 심볼', 'INVALID_SYMBOL'),
        ('빈 심볼', ''),
        ('None 심볼', None),
        ('숫자 심볼', 12345)
    ]
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n🛡️ {strategy_name} 에러 테스트:")
        
        func = loader.get_strategy_function(strategy_name)
        if not func:
            log_test_result(f"{strategy_name} 에러 테스트", False, "함수 없음")
            continue
        
        for test_case, test_symbol in error_test_cases:
            start_time = time.time()
            try:
                if inspect.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(test_symbol), timeout=10.0)
                else:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(func, test_symbol)
                        result = await asyncio.wait_for(
                            asyncio.wrap_future(future), timeout=10.0
                        )
                
                duration = time.time() - start_time
                
                # 에러가 발생하지 않고 결과가 있으면 우아한 처리
                if result:
                    log_test_result(f"{strategy_name} {test_case} 처리", True, "우아한 에러 처리", duration)
                else:
                    log_test_result(f"{strategy_name} {test_case} 처리", True, "None 반환", duration)
                    
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                log_test_result(f"{strategy_name} {test_case} 처리", False, "타임아웃", duration)
                
            except Exception as e:
                duration = time.time() - start_time
                # 예외 발생은 정상적인 에러 처리로 간주
                log_test_result(f"{strategy_name} {test_case} 처리", True, f"예외 처리: {type(e).__name__}", duration)

# =====================================
# 📋 통합 테스트 및 결과 요약
# =====================================

def print_comprehensive_test_summary():
    """종합 테스트 결과 요약"""
    global test_results, performance_results, strategy_results
    
    print(f"\n{Colors.BOLD}📋 파일 기반 전략 테스트 결과 요약{Colors.END}")
    print("=" * 80)
    
    # 전체 통계
    total_tests = len(test_results)
    passed_tests = len([r for r in test_results if r['success']])
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 전체 테스트: {total_tests}개")
    print(f"✅ 성공: {Colors.GREEN}{passed_tests}개{Colors.END}")
    print(f"❌ 실패: {Colors.RED}{failed_tests}개{Colors.END}")
    print(f"📈 성공률: {Colors.CYAN}{success_rate:.1f}%{Colors.END}")
    
    # 전략별 상세 결과
    print(f"\n{Colors.BOLD}🎯 전략별 상세 결과:{Colors.END}")
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n📁 {Colors.BOLD}{strategy_name.upper()}{Colors.END} ({strategy_info['description']}):")
        
        # 파일 로딩 상태
        strategy_tests = [r for r in test_results if strategy_name in r['test_name']]
        strategy_success = len([r for r in strategy_tests if r['success']])
        strategy_total = len(strategy_tests)
        
        if strategy_total > 0:
            strategy_rate = (strategy_success / strategy_total) * 100
            color = Colors.GREEN if strategy_rate >= 80 else Colors.YELLOW if strategy_rate >= 60 else Colors.RED
            print(f"   📊 테스트 성공률: {color}{strategy_rate:.1f}%{Colors.END} ({strategy_success}/{strategy_total})")
        
        # 함수 호출 결과
        if strategy_name in strategy_results:
            result_data = strategy_results[strategy_name]
            if result_data['success']:
                print(f"   ✅ 함수 호출: 성공 ({result_data['duration']:.2f}s)")
                print(f"   🎯 테스트 심볼: {result_data['symbol']}")
                
                # 결과 검증
                if 'validation' in result_data:
                    validation = result_data['validation']
                    format_status = "✅" if validation['format_valid'] else "❌"
                    data_status = "✅" if validation['data_valid'] else "❌"
                    print(f"   📋 결과 형식: {format_status}")
                    print(f"   📊 데이터 유효성: {data_status}")
            else:
                print(f"   ❌ 함수 호출: 실패")
        
        # 성능 결과
        if strategy_name in performance_results:
            perf = performance_results[strategy_name]
            avg_time = perf['avg_time']
            success_rate = perf['success_rate']
            
            time_color = Colors.GREEN if avg_time < 5.0 else Colors.YELLOW if avg_time < 10.0 else Colors.RED
            rate_color = Colors.GREEN if success_rate >= 100 else Colors.YELLOW if success_rate >= 66 else Colors.RED
            
            print(f"   ⚡ 평균 속도: {time_color}{avg_time:.2f}s{Colors.END}")
            print(f"   🎯 성공률: {rate_color}{success_rate:.0f}%{Colors.END}")
    
    # 실패한 테스트 상세
    if failed_tests > 0:
        print(f"\n{Colors.RED}❌ 실패한 테스트 상세:{Colors.END}")
        failed_by_strategy = {}
        
        for result in test_results:
            if not result['success']:
                strategy = None
                for name in STRATEGY_FILES.keys():
                    if name in result['test_name']:
                        strategy = name
                        break
                
                if strategy:
                    if strategy not in failed_by_strategy:
                        failed_by_strategy[strategy] = []
                    failed_by_strategy[strategy].append(result)
        
        for strategy, failures in failed_by_strategy.items():
            print(f"\n   📁 {strategy}:")
            for failure in failures:
                print(f"      • {failure['test_name']}: {failure['message']}")
    
    # 성능 벤치마크 요약
    if performance_results:
        print(f"\n{Colors.PURPLE}📈 성능 벤치마크 요약:{Colors.END}")
        
        # 성능 순위
        sorted_perf = sorted(performance_results.items(), key=lambda x: x[1]['avg_time'])
        
        for i, (strategy, perf) in enumerate(sorted_perf, 1):
            avg_time = perf['avg_time']
            success_rate = perf['success_rate']
            
            # 성능 등급
            if avg_time < 5.0 and success_rate >= 100:
                grade = f"{Colors.GREEN}⭐ 우수{Colors.END}"
            elif avg_time < 10.0 and success_rate >= 66:
                grade = f"{Colors.CYAN}👍 양호{Colors.END}"
            elif avg_time < 15.0 and success_rate >= 33:
                grade = f"{Colors.YELLOW}⚠️ 보통{Colors.END}"
            else:
                grade = f"{Colors.RED}🔧 개선필요{Colors.END}"
            
            print(f"   {i}. {strategy}: {grade} (평균 {avg_time:.2f}s, 성공률 {success_rate:.0f}%)")
    
    # 종합 평가
    print(f"\n{Colors.BOLD}🏆 종합 평가:{Colors.END}")
    
    if success_rate >= 90:
        overall_grade = f"{Colors.GREEN}🎉 EXCELLENT{Colors.END}"
        message = "모든 전략 파일이 완벽하게 작동합니다!"
    elif success_rate >= 80:
        overall_grade = f"{Colors.CYAN}👍 GOOD{Colors.END}"
        message = "전략 파일들이 양호하게 작동합니다"
    elif success_rate >= 70:
        overall_grade = f"{Colors.YELLOW}⚠️ FAIR{Colors.END}"
        message = "일부 개선이 필요합니다"
    else:
        overall_grade = f"{Colors.RED}❌ POOR{Colors.END}"
        message = "전략 파일 점검이 필요합니다"
    
    print(f"   등급: {overall_grade}")
    print(f"   평가: {message}")
    
    # 권장사항
    print(f"\n{Colors.BOLD}💡 권장사항:{Colors.END}")
    
    recommendations = []
    
    # 실패한 전략별 권장사항
    for strategy_name in STRATEGY_FILES.keys():
        strategy_tests = [r for r in test_results if strategy_name in r['test_name']]
        strategy_failures = [r for r in strategy_tests if not r['success']]
        
        if strategy_failures:
            recommendations.append(f"📁 {strategy_name}: {len(strategy_failures)}개 문제 해결 필요")
    
    # 성능 개선 권장사항
    for strategy, perf in performance_results.items():
        if perf['avg_time'] > 10.0:
            recommendations.append(f"⚡ {strategy}: 성능 최적화 필요 (현재 {perf['avg_time']:.1f}초)")
        if perf['success_rate'] < 100:
            recommendations.append(f"🎯 {strategy}: 안정성 개선 필요 (성공률 {perf['success_rate']:.0f}%)")
    
    if recommendations:
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print(f"   ✅ 모든 전략이 우수한 상태입니다!")
    
    # 총 실행 시간
    total_duration = sum(r['duration'] for r in test_results)
    print(f"\n⏱️ 총 실행 시간: {total_duration:.2f}초")

# =====================================
# 📊 결과 내보내기 기능
# =====================================

def export_test_results_to_json(filename: str = None) -> str:
    """테스트 결과를 JSON으로 내보내기"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"strategy_test_results_{timestamp}.json"
    
    export_data = {
        'test_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_results),
            'passed_tests': len([r for r in test_results if r['success']]),
            'failed_tests': len([r for r in test_results if not r['success']]),
            'success_rate': (len([r for r in test_results if r['success']]) / len(test_results) * 100) if test_results else 0
        },
        'strategy_files': STRATEGY_FILES,
        'test_results': test_results,
        'performance_results': performance_results,
        'strategy_results': strategy_results
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 테스트 결과가 {filename}에 저장되었습니다.")
        return filename
        
    except Exception as e:
        print(f"\n❌ 결과 저장 실패: {e}")
        return ""

def generate_test_report_summary() -> str:
    """테스트 리포트 요약 생성"""
    total_tests = len(test_results)
    passed_tests = len([r for r in test_results if r['success']])
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    report = []
    report.append("🧪 파일 기반 전략 테스트 리포트 요약")
    report.append("=" * 50)
    report.append(f"📊 전체 테스트: {total_tests}개")
    report.append(f"✅ 성공: {passed_tests}개")
    report.append(f"❌ 실패: {total_tests - passed_tests}개")
    report.append(f"📈 성공률: {success_rate:.1f}%")
    report.append("")
    
    # 전략별 요약
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        if strategy_name in strategy_results:
            result_data = strategy_results[strategy_name]
            status = "✅ 성공" if result_data['success'] else "❌ 실패"
            report.append(f"📁 {strategy_name}: {status} ({result_data['duration']:.2f}s)")
    
    if performance_results:
        report.append("")
        report.append("📈 성능 요약:")
        for strategy, perf in performance_results.items():
            report.append(f"   {strategy}: {perf['avg_time']:.2f}s (성공률 {perf['success_rate']:.0f}%)")
    
    return "\n".join(report)

# =====================================
# 🚀 메인 테스트 실행 함수
# =====================================

async def run_all_strategy_tests(export_results: bool = True):
    """전체 전략 테스트 실행"""
    print(f"{Colors.BOLD}{Colors.CYAN}🧪 최고퀸트프로젝트 - 파일 기반 전략 테스트 시스템{Colors.END}")
    print("=" * 80)
    print(f"🎯 테스트 대상: {len(STRATEGY_FILES)}개 전략 파일")
    print(f"📁 일본 주식, 미국 주식, 암호화폐 전략")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. 파일 로딩 테스트
        loader = test_strategy_file_loading()
        
        # 2. 설정 연동 테스트
        test_config_integration(loader)
        
        # 3. 비동기 함수 호출 테스트
        await test_async_function_calls(loader)
        
        # 4. 결과 형식 검증 테스트
        test_result_format_validation()
        
        # 5. 성능 벤치마크 테스트
        await test_performance_benchmark(loader)
        
        # 6. 에러 핸들링 테스트
        await test_error_handling(loader)
        
        # 7. 결과 요약 출력
        print_comprehensive_test_summary()
        
        # 8. 결과 내보내기
        if export_results:
            export_test_results_to_json()
        
        # 9. 간단 요약 생성
        summary = generate_test_report_summary()
        print(f"\n{Colors.BOLD}📋 테스트 완료 요약:{Colors.END}")
        print(summary)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ 사용자에 의해 테스트가 중단되었습니다{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ 테스트 실행 중 치명적 오류: {e}{Colors.END}")
        traceback.print_exc()

async def run_quick_strategy_test():
    """빠른 전략 테스트 (핵심 기능만)"""
    print(f"{Colors.BOLD}{Colors.CYAN}⚡ 빠른 전략 테스트{Colors.END}")
    print("=" * 50)
    
    loader = StrategyFileLoader()
    
    for strategy_name, strategy_info in STRATEGY_FILES.items():
        print(f"\n📁 {strategy_name} 빠른 테스트:")
        
        # 파일 로드
        start_time = time.time()
        module = loader.load_strategy_module(strategy_name)
        if module:
            log_test_result(f"{strategy_name} 로드", True, "", time.time() - start_time)
            
            # 함수 호출
            func = loader.get_strategy_function(strategy_name)
            if func:
                start_time = time.time()
                try:
                    test_symbol = strategy_info['test_symbol']
                    timeout = 15.0  # 빠른 테스트는 15초 제한
                    
                    if inspect.iscoroutinefunction(func):
                        result = await asyncio.wait_for(func(test_symbol), timeout=timeout)
                    else:
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, test_symbol)
                            result = await asyncio.wait_for(
                                asyncio.wrap_future(future), timeout=timeout
                            )
                    
                    if result:
                        log_test_result(f"{strategy_name} 실행", True, f"결과 수신", time.time() - start_time)
                    else:
                        log_test_result(f"{strategy_name} 실행", False, "결과 없음", time.time() - start_time)
                        
                except asyncio.TimeoutError:
                    log_test_result(f"{strategy_name} 실행", False, "타임아웃", time.time() - start_time)
                except Exception as e:
                    log_test_result(f"{strategy_name} 실행", False, str(e), time.time() - start_time)
            else:
                log_test_result(f"{strategy_name} 함수", False, "함수 로드 실패")
        else:
            log_test_result(f"{strategy_name} 로드", False, "모듈 로드 실패")

# =====================================
# 📱 CLI 인터페이스
# =====================================

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='파일 기반 전략 테스트 시스템')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 모드')
    parser.add_argument('--no-export', action='store_true', help='결과 내보내기 비활성화')
    parser.add_argument('--strategy', type=str, help='특정 전략만 테스트 (jp_strategy, us_strategy, coin_strategy)')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # 특정 전략만 테스트
    if args.strategy:
        if args.strategy in STRATEGY_FILES:
            global STRATEGY_FILES
            STRATEGY_FILES = {args.strategy: STRATEGY_FILES[args.strategy]}
            print(f"🎯 특정 전략 테스트: {args.strategy}")
        else:
            print(f"❌ 알 수 없는 전략: {args.strategy}")
            print(f"사용 가능한 전략: {', '.join(STRATEGY_FILES.keys())}")
            return
    
    # 테스트 실행
    try:
        if args.quick:
            asyncio.run(run_quick_strategy_test())
        else:
            asyncio.run(run_all_strategy_tests(not args.no_export))
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ 테스트가 중단되었습니다{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ 테스트 실행 실패: {e}{Colors.END}")
        if args.verbose:
            traceback.print_exc()

if __name__ == "__main__":
    main()

# =====================================
# 📚 사용 예시 및 도움말
# =====================================

"""
🧪 파일 기반 전략 테스트 시스템 사용법

기본 실행:
    python test_strategies.py

빠른 테스트:
    python test_strategies.py --quick

특정 전략 테스트:
    python test_strategies.py --strategy jp_strategy
    python test_strategies.py --strategy us_strategy
    python test_strategies.py --strategy coin_strategy

결과 내보내기 비활성화:
    python test_strategies.py --no-export

상세 출력:
    python test_strategies.py --verbose

🎯 테스트 항목:
1. 📁 파일 로딩 및 모듈 import
2. 🔧 설정 파일 연동 검증
3. ⚡ 비동기 함수 호출 및 타임아웃
4. 📊 결과 형식 및 데이터 유효성 검증
5. 🎯 성능 벤치마크 (3회 실행 평균)
6. 🛡️ 에러 핸들링 테스트

📋 지원하는 전략:
- jp_strategy.py: 일본 주식 전략 (엔화 기반 + 기술분석)
- us_strategy.py: 미국 주식 전략 (4가지 전략 융합 + VIX)
- coin_strategy.py: 암호화폐 전략 (6단계 필터링 + AI 품질평가)

💾 출력 파일:
- strategy_test_results_YYYYMMDD_HHMMSS.json: 상세 테스트 결과
"""
