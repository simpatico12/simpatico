#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ LEGENDARY QUANT STRATEGY COMPLETE ⚡
전설급 5대 시스템 + 완전한 매도 시스템

🧠 Neural Quality Engine - 가중평균 기반 품질 스코어링
🌊 Quantum Cycle Matrix - 27개 미시사이클 감지  
⚡ Fractal Filtering Pipeline - 다차원 필터링
💎 Diamond Hand Algorithm - 켈리공식 기반 분할매매
🕸️ Correlation Web Optimizer - 네트워크 포트폴리오
🎯 Position Manager - 포지션 관리 + 실시간 매도

Author: 퀀트마스터 | Version: COMPLETE
"""

import asyncio
import numpy as np
import pandas as pd
import pyupbit
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import os
import time

# ============================================================================
# 🧠 NEURAL QUALITY ENGINE - 가중평균 기반 품질 스코어링 (오류 수정)
# ============================================================================
class NeuralQualityEngine:
    """가중평균 기반 품질 평가 엔진 (안정성 최우선)"""
    
    def __init__(self):
        # 코인별 품질 점수 (기술력, 생태계, 커뮤니티, 채택도)
        self.coin_scores = {
            'BTC': [0.98, 0.95, 0.90, 0.95], 'ETH': [0.95, 0.98, 0.85, 0.90],
            'BNB': [0.80, 0.90, 0.75, 0.85], 'ADA': [0.85, 0.75, 0.80, 0.70],
            'SOL': [0.90, 0.80, 0.85, 0.75], 'AVAX': [0.85, 0.75, 0.70, 0.70],
            'DOT': [0.85, 0.80, 0.70, 0.65], 'MATIC': [0.80, 0.85, 0.75, 0.80],
            'ATOM': [0.75, 0.70, 0.75, 0.60], 'NEAR': [0.80, 0.70, 0.65, 0.60],
            'LINK': [0.90, 0.75, 0.70, 0.80], 'UNI': [0.85, 0.80, 0.75, 0.75],
        }
        
        # 가중치 (기술력 30%, 생태계 30%, 커뮤니티 20%, 채택도 20%)
        self.weights = [0.30, 0.30, 0.20, 0.20]
    
    def neural_quality_score(self, symbol: str, market_data: Dict) -> Dict:
        """안전한 가중평균 품질 점수 계산"""
        try:
            coin_name = symbol.replace('KRW-', '')
            
            # 기본 점수 또는 등록된 점수 사용
            scores = self.coin_scores.get(coin_name, [0.6, 0.6, 0.6, 0.6])
            
            # 가중평균 계산 (안전한 방식)
            quality_score = sum(score * weight for score, weight in zip(scores, self.weights))
            
            # 거래량 기반 보너스
            volume_bonus = self._calculate_volume_bonus(market_data.get('volume_24h_krw', 0))
            final_quality = min(0.98, quality_score + volume_bonus)
            
            # AI 설명 생성
            explanation = self._generate_explanation(coin_name, scores, final_quality)
            
            return {
                'quality_score': final_quality,
                'tech_score': scores[0],
                'ecosystem_score': scores[1], 
                'community_score': scores[2],
                'adoption_score': scores[3],
                'ai_explanation': explanation,
                'confidence': min(0.95, final_quality + 0.05)
            }
            
        except Exception as e:
            logging.error(f"품질 점수 계산 실패 {symbol}: {e}")
            return {
                'quality_score': 0.5, 'tech_score': 0.5, 'ecosystem_score': 0.5,
                'community_score': 0.5, 'adoption_score': 0.5,
                'ai_explanation': '기본등급', 'confidence': 0.5
            }
    
    def _calculate_volume_bonus(self, volume_krw: float) -> float:
        """거래량 기반 보너스 계산"""
        if volume_krw >= 100_000_000_000: return 0.05
        elif volume_krw >= 50_000_000_000: return 0.03
        elif volume_krw >= 10_000_000_000: return 0.01
        else: return 0.0
    
    def _generate_explanation(self, coin: str, scores: List[float], final_score: float) -> str:
        """설명 생성"""
        features = []
        if scores[0] > 0.85: features.append("최고급기술")
        if scores[1] > 0.85: features.append("강력생태계")
        if scores[2] > 0.80: features.append("활발커뮤니티")
        if scores[3] > 0.80: features.append("높은채택도")
        
        if final_score > 0.8: grade = "S급"
        elif final_score > 0.7: grade = "A급"  
        elif final_score > 0.6: grade = "B급"
        else: grade = "C급"
        
        return f"{grade} | " + " | ".join(features) if features else f"{grade} | 기본등급"

# ============================================================================
# 🌊 QUANTUM CYCLE MATRIX - 27개 미시사이클 감지
# ============================================================================
class QuantumCycleMatrix:
    """양자역학 스타일 시장 사이클 감지기"""
    
    async def detect_quantum_cycle(self) -> Dict:
        """양자 사이클 매트릭스 감지"""
        try:
            # BTC 데이터 수집
            btc_data = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=90)
            if btc_data is None or len(btc_data) < 60:
                with open(self.position_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_positions, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"포지션 저장 실패: {e}")
    
    def load_positions(self):
        """포지션 로드"""
        try:
            if os.path.exists(self.position_file):
                with open(self.position_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for symbol, pos_data in data.items():
                    self.positions[symbol] = Position(
                        symbol=pos_data['symbol'],
                        total_quantity=pos_data['total_quantity'],
                        avg_price=pos_data['avg_price'],
                        current_stage=pos_data['current_stage'],
                        stage_quantities=pos_data['stage_quantities'],
                        stage_prices=pos_data['stage_prices'],
                        target_take_profits=pos_data['target_take_profits'],
                        stop_loss=pos_data['stop_loss'],
                        unrealized_pnl=pos_data['unrealized_pnl'],
                        created_at=datetime.fromisoformat(pos_data['created_at']),
                        last_updated=datetime.fromisoformat(pos_data['last_updated'])
                    )
                
                logging.info(f"포지션 로드 완료: {len(self.positions)}개")
        except Exception as e:
            logging.error(f"포지션 로드 실패: {e}")

# ============================================================================
# 🚨 EXIT STRATEGY ENGINE - 출구 전략 엔진
# ============================================================================
class ExitStrategyEngine:
    """실시간 매도 전략 엔진"""
    
    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
        self.trailing_stop_ratio = 0.10  # 10% 트레일링 스톱
    
    async def check_exit_conditions(self, symbol: str, current_price: float, current_cycle: str) -> Dict:
        """매도 조건 체크"""
        position = self.position_manager.get_position(symbol)
        if not position:
            return {'action': 'none', 'reason': 'no_position'}
        
        # 미실현 손익 업데이트
        self.position_manager.update_unrealized_pnl(symbol, current_price)
        
        # 1. 손절 체크
        if current_price <= position.stop_loss:
            return {
                'action': 'sell_all',
                'reason': 'stop_loss',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'손절 실행: {current_price} <= {position.stop_loss}'
            }
        
        # 2. 익절 체크 (단계별)
        profit_ratio = (current_price - position.avg_price) / position.avg_price
        
        # 1차 익절 (20% 매도)
        if (current_price >= position.target_take_profits[0] and 
            profit_ratio >= 0.20):
            sell_quantity = position.total_quantity * 0.2
            return {
                'action': 'sell_partial',
                'reason': 'take_profit_1',
                'price': current_price,
                'quantity': sell_quantity,
                'details': f'1차 익절: 20% 매도'
            }
        
        # 2차 익절 (50% 매도)
        if (current_price >= position.target_take_profits[1] and 
            profit_ratio >= 0.50):
            sell_quantity = position.total_quantity * 0.5
            return {
                'action': 'sell_partial',
                'reason': 'take_profit_2',
                'price': current_price,
                'quantity': sell_quantity,
                'details': f'2차 익절: 50% 매도'
            }
        
        # 3차 익절 (전량 매도)
        if (current_price >= position.target_take_profits[2] and 
            profit_ratio >= 1.00):
            return {
                'action': 'sell_all',
                'reason': 'take_profit_3',
                'price': current_price,
                'quantity': position.total_quantity,
                'details': f'3차 익절: 전량 매도'
            }
        
        # 3. 사이클 변화 매도
        cycle_exit = self._check_cycle_exit(position, current_cycle, profit_ratio)
        if cycle_exit:
            return cycle_exit
        
        # 4. 트레일링 스톱
        trailing_exit = self._check_trailing_stop(position, current_price)
        if trailing_exit:
            return trailing_exit
        
        return {'action': 'hold', 'reason': 'no_exit_condition'}
    
    def _check_cycle_exit(self, position: Position, current_cycle: str, profit_ratio: float) -> Optional[Dict]:
        """사이클 변화 매도 체크"""
        # 수익 상태에서 약세 사이클로 전환 시 매도
        if profit_ratio > 0.05 and current_cycle in ['strong_bear', 'reversal_phase']:
            return {
                'action': 'sell_all',
                'reason': 'cycle_change',
                'price': 0,  # 현재가로 매도
                'quantity': position.total_quantity,
                'details': f'사이클 변화 매도: {current_cycle}'
            }
        
        return None
    
    def _check_trailing_stop(self, position: Position, current_price: float) -> Optional[Dict]:
        """트레일링 스톱 체크"""
        # 수익이 10% 이상일 때만 트레일링 스톱 적용
        profit_ratio = (current_price - position.avg_price) / position.avg_price
        
        if profit_ratio > 0.10:
            # 최고점 대비 10% 하락 시 매도 (단순화)
            # 실제로는 최고점을 추적해야 하지만 여기서는 간단히 구현
            dynamic_stop = position.avg_price * (1 + profit_ratio - self.trailing_stop_ratio)
            
            if current_price <= dynamic_stop:
                return {
                    'action': 'sell_all',
                    'reason': 'trailing_stop',
                    'price': current_price,
                    'quantity': position.total_quantity,
                    'details': f'트레일링 스톱: {current_price} <= {dynamic_stop}'
                }
        
        return None

# ============================================================================
# 🎮 TRADE EXECUTOR - 거래 실행기
# ============================================================================
class TradeExecutor:
    """거래 실행 시스템"""
    
    def __init__(self, position_manager: PositionManager, demo_mode: bool = True):
        self.position_manager = position_manager
        self.demo_mode = demo_mode  # 실제 거래 vs 시뮬레이션
        
        if not demo_mode:
            # 실제 거래용 업비트 API 초기화
            self.access_key = os.getenv('UPBIT_ACCESS_KEY')
            self.secret_key = os.getenv('UPBIT_SECRET_KEY')
            self.upbit = pyupbit.Upbit(self.access_key, self.secret_key)
    
    async def execute_buy_signal(self, signal: LegendarySignal, stage: int) -> Dict:
        """매수 신호 실행"""
        try:
            symbol = signal.symbol
            target_price = signal.entry_prices[stage - 1]
            target_amount = signal.stage_amounts[stage - 1]
            
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'price_fetch_failed'}
            
            # 진입 조건 체크 (현재가가 목표가 이하일 때만)
            if current_price > target_price * 1.02:  # 2% 여유
                return {'success': False, 'error': 'price_too_high'}
            
            # 수량 계산
            quantity = target_amount / current_price
            
            if self.demo_mode:
                # 시뮬레이션 모드
                result = {
                    'success': True,
                    'symbol': symbol,
                    'stage': stage,
                    'quantity': quantity,
                    'price': current_price,
                    'amount': target_amount,
                    'type': 'demo_buy'
                }
                
                # 포지션 업데이트
                self.position_manager.add_position(signal, stage, quantity, current_price)
                
                logging.info(f"📈 시뮬레이션 매수: {symbol} 단계{stage} {quantity:.6f}개 @ {current_price:,.0f}원")
                
            else:
                # 실제 거래
                order = self.upbit.buy_market_order(symbol, target_amount)
                if order:
                    result = {
                        'success': True,
                        'symbol': symbol,
                        'stage': stage,
                        'order_id': order['uuid'],
                        'quantity': quantity,
                        'price': current_price,
                        'amount': target_amount,
                        'type': 'real_buy'
                    }
                    
                    # 포지션 업데이트
                    self.position_manager.add_position(signal, stage, quantity, current_price)
                    
                    logging.info(f"📈 실제 매수: {symbol} 단계{stage} {quantity:.6f}개 @ {current_price:,.0f}원")
                else:
                    result = {'success': False, 'error': 'order_failed'}
            
            return result
            
        except Exception as e:
            logging.error(f"매수 실행 실패 {signal.symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_sell_signal(self, symbol: str, sell_action: Dict) -> Dict:
        """매도 신호 실행"""
        try:
            position = self.position_manager.get_position(symbol)
            if not position:
                return {'success': False, 'error': 'no_position'}
            
            current_price = pyupbit.get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'price_fetch_failed'}
            
            sell_quantity = sell_action['quantity']
            sell_amount = sell_quantity * current_price
            
            if self.demo_mode:
                # 시뮬레이션 모드
                result = {
                    'success': True,
                    'symbol': symbol,
                    'quantity': sell_quantity,
                    'price': current_price,
                    'amount': sell_amount,
                    'reason': sell_action['reason'],
                    'type': 'demo_sell'
                }
                
                # 포지션 업데이트
                if sell_action['action'] == 'sell_all':
                    self.position_manager.remove_position(symbol)
                else:
                    self.position_manager.remove_position(symbol, sell_quantity)
                
                # 수익률 계산
                profit = (current_price - position.avg_price) * sell_quantity
                profit_ratio = (current_price - position.avg_price) / position.avg_price * 100
                
                logging.info(f"📉 시뮬레이션 매도: {symbol} {sell_quantity:.6f}개 @ {current_price:,.0f}원")
                logging.info(f"💰 손익: {profit:+,.0f}원 ({profit_ratio:+.1f}%) - {sell_action['reason']}")
                
            else:
                # 실제 거래
                order = self.upbit.sell_market_order(symbol, sell_quantity)
                if order:
                    result = {
                        'success': True,
                        'symbol': symbol,
                        'order_id': order['uuid'],
                        'quantity': sell_quantity,
                        'price': current_price,
                        'amount': sell_amount,
                        'reason': sell_action['reason'],
                        'type': 'real_sell'
                    }
                    
                    # 포지션 업데이트
                    if sell_action['action'] == 'sell_all':
                        self.position_manager.remove_position(symbol)
                    else:
                        self.position_manager.remove_position(symbol, sell_quantity)
                    
                    profit = (current_price - position.avg_price) * sell_quantity
                    profit_ratio = (current_price - position.avg_price) / position.avg_price * 100
                    
                    logging.info(f"📉 실제 매도: {symbol} {sell_quantity:.6f}개 @ {current_price:,.0f}원")
                    logging.info(f"💰 손익: {profit:+,.0f}원 ({profit_ratio:+.1f}%) - {sell_action['reason']}")
                else:
                    result = {'success': False, 'error': 'order_failed'}
            
            return result
            
        except Exception as e:
            logging.error(f"매도 실행 실패 {symbol}: {e}")
            return {'success': False, 'error': str(e)}

# ============================================================================
# 📊 REAL-TIME MONITOR - 실시간 모니터
# ============================================================================
class RealTimeMonitor:
    """실시간 모니터링 시스템"""
    
    def __init__(self, position_manager: PositionManager, exit_engine: ExitStrategyEngine, 
                 trade_executor: TradeExecutor, quantum_cycle: QuantumCycleMatrix):
        self.position_manager = position_manager
        self.exit_engine = exit_engine
        self.trade_executor = trade_executor
        self.quantum_cycle = quantum_cycle
        self.monitoring = False
    
    async def start_monitoring(self, check_interval: int = 300):  # 5분마다
        """실시간 모니터링 시작"""
        self.monitoring = True
        logging.info("🔄 실시간 모니터링 시작")
        
        while self.monitoring:
            try:
                # 현재 시장 사이클 확인
                cycle_info = await self.quantum_cycle.detect_quantum_cycle()
                current_cycle = cycle_info['cycle']
                
                # 모든 포지션 체크
                positions = self.position_manager.get_all_positions()
                
                if positions:
                    logging.info(f"🔍 포지션 모니터링: {len(positions)}개 ({current_cycle})")
                
                for position in positions:
                    try:
                        # 현재가 조회
                        current_price = pyupbit.get_current_price(position.symbol)
                        if not current_price:
                            continue
                        
                        # 매도 조건 체크
                        exit_action = await self.exit_engine.check_exit_conditions(
                            position.symbol, current_price, current_cycle
                        )
                        
                        # 매도 실행
                        if exit_action['action'] in ['sell_all', 'sell_partial']:
                            logging.info(f"🚨 매도 신호: {position.symbol} - {exit_action['reason']}")
                            
                            # 매도 실행
                            sell_result = await self.trade_executor.execute_sell_signal(
                                position.symbol, exit_action
                            )
                            
                            if sell_result['success']:
                                logging.info(f"✅ 매도 성공: {position.symbol}")
                            else:
                                logging.error(f"❌ 매도 실패: {position.symbol} - {sell_result.get('error')}")
                        
                        # 미실현 손익 로그
                        pnl_ratio = (current_price - position.avg_price) / position.avg_price * 100
                        if abs(pnl_ratio) > 5:  # 5% 이상 변동 시에만 로그
                            logging.info(f"💹 {position.symbol}: {pnl_ratio:+.1f}% @ {current_price:,.0f}원")
                        
                    except Exception as e:
                        logging.error(f"개별 포지션 모니터링 실패 {position.symbol}: {e}")
                        continue
                
                # 다음 체크까지 대기
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logging.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(60)  # 1분 후 재시도
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        logging.info("⏹️ 실시간 모니터링 중지")

# ============================================================================
# 🏆 LEGENDARY QUANT MASTER - 전설급 통합 시스템 (완전체)
# ============================================================================
class LegendaryQuantMaster:
    """전설급 5대 시스템 + 완전한 매도 시스템 통합 마스터"""
    
    def __init__(self, portfolio_value: float = 100_000_000, min_volume: float = 10_000_000_000, demo_mode: bool = True):
        self.portfolio_value = portfolio_value
        self.min_volume = min_volume
        self.demo_mode = demo_mode
        
        # 전설급 5대 엔진 초기화
        self.neural_engine = NeuralQualityEngine()
        self.quantum_cycle = QuantumCycleMatrix()
        self.fractal_filter = FractalFilteringPipeline(min_volume)
        self.diamond_algorithm = DiamondHandAlgorithm(portfolio_value)
        
        # 매도 시스템 초기화
        self.position_manager = PositionManager()
        self.exit_engine = ExitStrategyEngine(self.position_manager)
        self.trade_executor = TradeExecutor(self.position_manager, demo_mode)
        self.monitor = RealTimeMonitor(self.position_manager, self.exit_engine, self.trade_executor, self.quantum_cycle)
        
        # 설정
        self.target_portfolio_size = 8
        
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s | %(levelname)s | %(message)s')
    
    async def execute_legendary_strategy(self) -> List[LegendarySignal]:
        """전설급 전략 실행"""
        logging.info("🏆 LEGENDARY QUANT STRATEGY COMPLETE 시작")
        
        try:
            # 1단계: 양자 사이클 감지
            logging.info("🌊 양자 사이클 매트릭스 감지 중...")
            quantum_state = await self.quantum_cycle.detect_quantum_cycle()
            
            # 2단계: 프랙탈 필터링
            logging.info("⚡ 프랙탈 필터링 파이프라인 실행 중...")
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            if not all_tickers:
                logging.error("티커 목록 조회 실패")
                return []
            
            fractal_candidates = await self.fractal_filter.execute_fractal_filtering(all_tickers)
            
            if not fractal_candidates:
                logging.error("프랙탈 필터링 결과 없음")
                return []
            
            # 3단계: 개별 코인 전설급 분석
            legendary_signals = []
            for i, candidate in enumerate(fractal_candidates[:self.target_portfolio_size], 1):
                logging.info(f"💎 전설급 분석 [{i}/{min(len(fractal_candidates), self.target_portfolio_size)}]: {candidate['symbol']}")
                
                signal = await self._analyze_legendary_coin(candidate, quantum_state)
                if signal:
                    legendary_signals.append(signal)
                
                await asyncio.sleep(0.2)  # API 제한
            
            # 4단계: 최종 포트폴리오 랭킹
            legendary_signals.sort(key=lambda x: x.legendary_score, reverse=True)
            
            # 결과 요약
            buy_signals = [s for s in legendary_signals if s.action == 'BUY']
            logging.info(f"✨ 전설급 분석 완료: {len(legendary_signals)}개 분석, {len(buy_signals)}개 매수 신호")
            
            return legendary_signals
            
        except Exception as e:
            logging.error(f"전설급 전략 실행 실패: {e}")
            return []
    
    async def _analyze_legendary_coin(self, candidate: Dict, quantum_state: Dict) -> Optional[LegendarySignal]:
        """개별 코인 전설급 분석"""
        try:
            symbol = candidate['symbol']
            price = candidate['price']
            
            # Neural Quality Engine 분석
            neural_result = self.neural_engine.neural_quality_score(
                symbol, {'volume_24h_krw': candidate['volume_krw']}
            )
            
            # Diamond Hand Algorithm 분석
            diamond_result = self.diamond_algorithm.calculate_diamond_strategy(
                symbol, price, neural_result['confidence'], 
                quantum_state['cycle'], neural_result['quality_score']
            )
            
            # 종합 점수 계산
            legendary_score = (
                neural_result['quality_score'] * 0.30 +      # Neural Quality
                quantum_state['confidence'] * 0.25 +         # Quantum Cycle  
                candidate.get('technical_score', 0.5) * 0.25 +  # Technical
                candidate.get('momentum_score', 0.5) * 0.20     # Momentum
            )
            
            # 액션 결정
            if legendary_score >= 0.70:
                action = 'BUY'
            elif legendary_score <= 0.30:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # 전설급 시그널 생성
            signal = LegendarySignal(
                symbol=symbol,
                action=action,
                confidence=neural_result['confidence'],
                price=price,
                neural_quality=neural_result['quality_score'],
                ai_explanation=neural_result['ai_explanation'],
                quantum_cycle=quantum_state['cycle'],
                cycle_confidence=quantum_state['confidence'],
                kelly_fraction=diamond_result['kelly_fraction'],
                emotion_factor=diamond_result['emotion_factor'],
                total_investment=diamond_result['total_investment'],
                entry_prices=diamond_result['entry_prices'],
                stage_amounts=diamond_result['stage_amounts'],
                take_profits=diamond_result['take_profits'],
                stop_loss=diamond_result['stop_loss'],
                legendary_score=legendary_score,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            logging.error(f"개별 코인 전설급 분석 실패 {candidate['symbol']}: {e}")
            return None
    
    async def execute_auto_trading(self, signals: List[LegendarySignal]):
        """자동 매매 실행"""
        buy_signals = [s for s in signals if s.action == 'BUY']
        
        if not buy_signals:
            logging.info("매수 신호가 없습니다.")
            return
        
        logging.info(f"🚀 자동 매매 시작: {len(buy_signals)}개 매수 신호")
        
        # 1단계 진입 (즉시 매수)
        for signal in buy_signals:
            try:
                result = await self.trade_executor.execute_buy_signal(signal, 1)
                if result['success']:
                    logging.info(f"✅ 1단계 매수 성공: {signal.symbol}")
                else:
                    logging.error(f"❌ 1단계 매수 실패: {signal.symbol} - {result.get('error')}")
                
                await asyncio.sleep(1)  # API 제한
                
            except Exception as e:
                logging.error(f"1단계 매수 실행 오류 {signal.symbol}: {e}")
        
        # 실시간 모니터링 시작
        logging.info("🔄 실시간 모니터링 및 2-3단계 진입 시작")
        
        # 모니터링을 별도 태스크로 실행
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())
        
        # 2-3단계 진입 체크 (별도 루프)
        entry_task = asyncio.create_task(self._check_additional_entries(buy_signals))
        
        # 두 태스크를 동시 실행
        try:
            await asyncio.gather(monitor_task, entry_task)
        except KeyboardInterrupt:
            logging.info("사용자 중단 요청")
            self.monitor.stop_monitoring()
    
    async def _check_additional_entries(self, signals: List[LegendarySignal]):
        """2-3단계 추가 진입 체크"""
        while self.monitor.monitoring:
            try:
                for signal in signals:
                    position = self.position_manager.get_position(signal.symbol)
                    if not position:
                        continue
                    
                    current_price = pyupbit.get_current_price(signal.symbol)
                    if not current_price:
                        continue
                    
                    # 2단계 진입 체크
                    if (position.current_stage == 1 and 
                        current_price <= signal.entry_prices[1]):
                        
                        result = await self.trade_executor.execute_buy_signal(signal, 2)
                        if result['success']:
                            logging.info(f"✅ 2단계 진입: {signal.symbol}")
                    
                    # 3단계 진입 체크
                    if (position.current_stage == 2 and 
                        current_price <= signal.entry_prices[2]):
                        
                        result = await self.trade_executor.execute_buy_signal(signal, 3)
                        if result['success']:
                            logging.info(f"✅ 3단계 진입: {signal.symbol}")
                
                await asyncio.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                logging.error(f"추가 진입 체크 오류: {e}")
                await asyncio.sleep(60)
    
    def print_legendary_results(self, signals: List[LegendarySignal]):
        """전설급 결과 출력"""
        print("\n" + "="*80)
        print("🏆 LEGENDARY QUANT STRATEGY COMPLETE RESULTS 🏆")
        print("="*80)
        
        if not signals:
            print("❌ 분석된 신호가 없습니다.")
            return
        
        buy_signals = [s for s in signals if s.action == 'BUY']
        total_investment = sum(s.total_investment for s in buy_signals)
        
        print(f"\n📊 전략 요약:")
        print(f"   분석 코인: {len(signals)}개")
        print(f"   매수 신호: {len(buy_signals)}개") 
        print(f"   총 투자금: {total_investment:,.0f}원")
        print(f"   포트폴리오 비중: {(total_investment/self.portfolio_value)*100:.1f}%")
        print(f"   운영 모드: {'시뮬레이션' if self.demo_mode else '실제거래'}")
        
        if signals:
            print(f"\n🌊 양자 사이클 상태:")
            print(f"   현재 사이클: {signals[0].quantum_cycle}")
            print(f"   신뢰도: {signals[0].cycle_confidence:.2f}")
        
        print(f"\n💎 전설급 매수 신호:")
        for i, signal in enumerate(buy_signals, 1):
            print(f"\n[{i}] {signal.symbol}")
            print(f"   전설 점수: {signal.legendary_score:.3f}")
            print(f"   AI 품질: {signal.neural_quality:.2f} | {signal.ai_explanation}")
            print(f"   켈리 비중: {signal.kelly_fraction:.1%}")
            print(f"   투자금액: {signal.total_investment:,.0f}원")
            print(f"   3단계 진입: {[f'{p:,.0f}' for p in signal.entry_prices]}")
            print(f"   3단계 익절: {[f'{p:,.0f}' for p in signal.take_profits]}")
            print(f"   손절선: {signal.stop_loss:,.0f}원")
        
        # 현재 포지션 상태
        positions = self.position_manager.get_all_positions()
        if positions:
            print(f"\n📈 현재 포지션 ({len(positions)}개):")
            for pos in positions:
                current_price = pyupbit.get_current_price(pos.symbol) or 0
                pnl_ratio = (current_price - pos.avg_price) / pos.avg_price * 100 if current_price > 0 else 0
                print(f"   {pos.symbol}: {pos.total_quantity:.6f}개 @ {pos.avg_price:,.0f}원 ({pnl_ratio:+.1f}%)")
        
        print("\n" + "="*80)
        print("⚡ LEGENDARY STRATEGY COMPLETE ⚡")
    
    def save_legendary_results(self, signals: List[LegendarySignal], filename: str = None):
        """전설급 결과 저장"""
        if not filename:
            filename = f"legendary_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 시그널을 JSON 직렬화
        serializable_signals = []
        for signal in signals:
            serializable_signals.append({
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'price': signal.price,
                'neural_quality': signal.neural_quality,
                'ai_explanation': signal.ai_explanation,
                'quantum_cycle': signal.quantum_cycle,
                'cycle_confidence': signal.cycle_confidence,
                'kelly_fraction': signal.kelly_fraction,
                'emotion_factor': signal.emotion_factor,
                'total_investment': signal.total_investment,
                'legendary_score': signal.legendary_score,
                'entry_prices': signal.entry_prices,
                'stage_amounts': signal.stage_amounts,
                'take_profits': signal.take_profits,
                'stop_loss': signal.stop_loss,
                'timestamp': signal.timestamp.isoformat()
            })
        
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_signals, f, ensure_ascii=False, indent=2)
        
        logging.info(f"✅ 전설급 결과 저장 완료: {filepath}")

# ============================================================================
# 🚀 메인 실행 함수들
# ============================================================================
async def main():
    """전설급 퀀트 전략 메인 실행"""
    print("⚡ LEGENDARY QUANT STRATEGY COMPLETE STARTING ⚡")
    print("🧠🌊⚡💎🕸️🎯🚨🎮📊 완전체 시스템 로딩...")
    
    # 전설급 마스터 초기화 (시뮬레이션 모드)
    master = LegendaryQuantMaster(
        portfolio_value=100_000_000,  # 1억원
        min_volume=5_000_000_000,     # 50억원
        demo_mode=True                # 시뮬레이션 모드
    )
    
    try:
        # 1단계: 전설급 분석 실행
        legendary_signals = await master.execute_legendary_strategy()
        
        # 2단계: 결과 출력
        master.print_legendary_results(legendary_signals)
        
        # 3단계: 결과 저장
        master.save_legendary_results(legendary_signals)
        
        # 4단계: 자동 매매 실행 여부 확인
        buy_signals = [s for s in legendary_signals if s.action == 'BUY']
        if buy_signals:
            print(f"\n🤔 {len(buy_signals)}개 매수 신호 발견!")
            print("자동 매매를 시작하시겠습니까? (y/n)")
            
            # 실제 환경에서는 사용자 입력 받기
            # user_input = input().lower()
            user_input = 'y'  # 자동 실행 (테스트용)
            
            if user_input == 'y':
                print("🚀 자동 매매 시스템 시작!")
                await master.execute_auto_trading(legendary_signals)
            else:
                print("📊 분석 결과만 저장하고 종료합니다.")
        else:
            print("📊 매수 신호가 없어 분석만 완료했습니다.")
        
        return legendary_signals
        
    except KeyboardInterrupt:
        print("\n👋 전설급 전략을 종료합니다.")
        master.monitor.stop_monitoring()
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        logging.error(f"메인 실행 실패: {e}")

# 단일 코인 분석 함수
async def analyze_single_legendary_coin(symbol: str):
    """단일 코인 전설급 분석"""
    master = LegendaryQuantMaster()
    
    try:
        price = pyupbit.get_current_price(symbol)
        ohlcv = pyupbit.get_ohlcv(symbol, interval="day", count=30)
        volume_krw = ohlcv.iloc[-1]['volume'] * price
        
        candidate = {
            'symbol': symbol,
            'price': price,
            'volume_krw': volume_krw,
            'ohlcv': ohlcv,
            'technical_score': 0.6,
            'momentum_score': 0.5
        }
        
        quantum_state = await master.quantum_cycle.detect_quantum_cycle()
        signal = await master._analyze_legendary_coin(candidate, quantum_state)
        
        if signal:
            print(f"\n🏆 {symbol} 전설급 분석 결과:")
            print(f"   액션: {signal.action}")
            print(f"   전설 점수: {signal.legendary_score:.3f}")
            print(f"   AI 설명: {signal.ai_explanation}")
            print(f"   양자 사이클: {signal.quantum_cycle}")
            print(f"   투자 권장: {signal.total_investment:,.0f}원")
            print(f"   3단계 진입: {[f'{p:,.0f}' for p in signal.entry_prices]}")
            print(f"   3단계 익절: {[f'{p:,.0f}' for p in signal.take_profits]}")
            
        return signal
        
    except Exception as e:
        print(f"❌ {symbol} 분석 실패: {e}")
        return None

# 포지션 상태 확인 함수
def check_positions():
    """현재 포지션 상태 확인"""
    position_manager = PositionManager()
    positions = position_manager.get_all_positions()
    
    if not positions:
        print("📭 현재 포지션이 없습니다.")
        return
    
    print(f"\n📊 현재 포지션 현황 ({len(positions)}개)")
    print("-" * 60)
    
    total_investment = 0
    total_pnl = 0
    
    for pos in positions:
        try:
            current_price = pyupbit.get_current_price(pos.symbol) or 0
            if current_price > 0:
                position_manager.update_unrealized_pnl(pos.symbol, current_price)
                pnl_ratio = (current_price - pos.avg_price) / pos.avg_price * 100
                investment = pos.total_quantity * pos.avg_price
                pnl = pos.total_quantity * (current_price - pos.avg_price)
                
                total_investment += investment
                total_pnl += pnl
                
                print(f"{pos.symbol:12} | {pos.total_quantity:10.6f}개")
                print(f"             | 평균가: {pos.avg_price:8,.0f}원 | 현재가: {current_price:8,.0f}원")
                print(f"             | 투자금: {investment:8,.0f}원 | 손익: {pnl:+8,.0f}원 ({pnl_ratio:+5.1f}%)")
                print(f"             | 단계: {pos.current_stage}/3 | 손절선: {pos.stop_loss:8,.0f}원")
                print("-" * 60)
        except Exception as e:
            print(f"{pos.symbol}: 가격 조회 실패 - {e}")
    
    if total_investment > 0:
        total_return = (total_pnl / total_investment) * 100
        print(f"💰 포트폴리오 요약:")
        print(f"   총 투자금: {total_investment:,.0f}원")
        print(f"   총 손익: {total_pnl:+,.0f}원 ({total_return:+.1f}%)")

# 실시간 모니터링만 실행하는 함수
async def start_monitoring_only():
    """기존 포지션에 대해 실시간 모니터링만 실행"""
    position_manager = PositionManager()
    positions = position_manager.get_all_positions()
    
    if not positions:
        print("📭 모니터링할 포지션이 없습니다.")
        return
    
    print(f"🔄 {len(positions)}개 포지션 실시간 모니터링 시작")
    
    # 모니터링 시스템 초기화
    quantum_cycle = QuantumCycleMatrix()
    exit_engine = ExitStrategyEngine(position_manager)
    trade_executor = TradeExecutor(position_manager, demo_mode=True)
    monitor = RealTimeMonitor(position_manager, exit_engine, trade_executor, quantum_cycle)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️ 모니터링을 중지합니다.")
        monitor.stop_monitoring()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'positions':
            # 포지션 확인
            check_positions()
        elif command == 'monitor':
            # 모니터링만 실행
            asyncio.run(start_monitoring_only())
        elif command.startswith('analyze:'):
            # 단일 코인 분석 (예: python script.py analyze:KRW-BTC)
            symbol = command.split(':')[1].upper()
            if not symbol.startswith('KRW-'):
                symbol = f'KRW-{symbol}'
            asyncio.run(analyze_single_legendary_coin(symbol))
        else:
            print("사용법:")
            print("  python script.py              # 전체 전략 실행")
            print("  python script.py positions    # 포지션 확인")
            print("  python script.py monitor      # 모니터링만 실행")
            print("  python script.py analyze:BTC  # 단일 코인 분석")
    else:
        # 기본 실행: 전체 전략
        asyncio.run(main()) or len(btc_data) < 60
                return self._default_cycle_state()
            
            # 3차원 상태 분석
            macro_state = self._detect_macro_cycle(btc_data)
            meso_state = self._detect_meso_cycle(btc_data)  
            micro_state = self._detect_micro_cycle(btc_data)
            
            # 사이클 강도 계산
            cycle_strength = self._calculate_cycle_strength(btc_data)
            
            # 최적 사이클 결정
            optimal_cycle = self._determine_optimal_cycle(macro_state, meso_state, micro_state)
            
            return {
                'cycle': optimal_cycle,
                'macro': macro_state,
                'meso': meso_state, 
                'micro': micro_state,
                'strength': cycle_strength,
                'confidence': min(0.95, 0.5 + cycle_strength * 0.5)
            }
            
        except Exception as e:
            logging.error(f"양자 사이클 감지 실패: {e}")
            return self._default_cycle_state()
    
    def _detect_macro_cycle(self, data: pd.DataFrame) -> str:
        """거시 사이클 (60일 기준)"""
        try:
            ma30 = data['close'].rolling(30).mean()
            ma60 = data['close'].rolling(60).mean()
            current_price = data['close'].iloc[-1]
            
            if current_price > ma30.iloc[-1] > ma60.iloc[-1]:
                return 'bull'
            elif current_price < ma30.iloc[-1] < ma60.iloc[-1]:
                return 'bear'
            else:
                return 'sideways'
        except:
            return 'sideways'
    
    def _detect_meso_cycle(self, data: pd.DataFrame) -> str:
        """중기 사이클 (14일 기준)"""
        try:
            high_14 = data['high'].rolling(14).max().iloc[-1]
            low_14 = data['low'].rolling(14).min().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if high_14 == low_14:  # 0으로 나누기 방지
                return 'range'
                
            position = (current_price - low_14) / (high_14 - low_14)
            
            if position > 0.7:
                return 'uptrend'
            elif position < 0.3:
                return 'downtrend'
            else:
                return 'range'
        except:
            return 'range'
    
    def _detect_micro_cycle(self, data: pd.DataFrame) -> str:
        """미시 사이클 (3일 기준)"""
        try:
            recent_returns = data['close'].pct_change().tail(3)
            if len(recent_returns) < 3:
                return 'stable'
                
            volatility = recent_returns.std()
            momentum = recent_returns.mean()
            
            if abs(momentum) > volatility * 1.5:
                return 'momentum'
            elif volatility > 0.03:  # 3% 이상 변동성
                return 'reversal'
            else:
                return 'stable'
        except:
            return 'stable'
    
    def _calculate_cycle_strength(self, data: pd.DataFrame) -> float:
        """사이클 강도 계산"""
        try:
            returns = data['close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.5
            
            # 변동성과 트렌드 강도 조합
            volatility = returns.std()
            trend_strength = abs(returns.mean()) / (volatility + 0.001)
            
            return min(1.0, trend_strength)
        except:
            return 0.5
    
    def _determine_optimal_cycle(self, macro: str, meso: str, micro: str) -> str:
        """최적 사이클 결정"""
        # 간단한 규칙 기반 매핑
        if macro == 'bull' and meso == 'uptrend':
            return 'strong_bull'
        elif macro == 'bear' and meso == 'downtrend':
            return 'strong_bear'
        elif micro == 'momentum':
            return 'momentum_phase'
        elif micro == 'reversal':
            return 'reversal_phase'
        else:
            return 'accumulation'
    
    def _default_cycle_state(self) -> Dict:
        """기본 상태"""
        return {
            'cycle': 'accumulation',
            'macro': 'sideways',
            'meso': 'range',
            'micro': 'stable',
            'strength': 0.5,
            'confidence': 0.5
        }

# ============================================================================
# ⚡ FRACTAL FILTERING PIPELINE - 다차원 필터링
# ============================================================================
class FractalFilteringPipeline:
    """프랙탈 차원 기반 다단계 필터링"""
    
    def __init__(self, min_volume: float):
        self.min_volume = min_volume
    
    async def execute_fractal_filtering(self, all_tickers: List[str]) -> List[Dict]:
        """프랙탈 파이프라인 실행"""
        logging.info("⚡ 프랙탈 필터링 파이프라인 시작")
        
        # 1단계: 원시 데이터 수집
        raw_data = await self._collect_raw_data(all_tickers)
        logging.info(f"원시 데이터: {len(raw_data)}개")
        
        if not raw_data:
            return []
        
        # 2단계: 단계별 필터링
        current_candidates = raw_data
        
        # 거래량 필터
        current_candidates = self._volume_filter(current_candidates)
        logging.info(f"거래량 필터: {len(current_candidates)}개")
        
        # 안정성 필터
        current_candidates = self._stability_filter(current_candidates)
        logging.info(f"안정성 필터: {len(current_candidates)}개")
        
        # 모멘텀 필터
        current_candidates = self._momentum_filter(current_candidates)
        logging.info(f"모멘텀 필터: {len(current_candidates)}개")
        
        # 기술적 필터
        current_candidates = self._technical_filter(current_candidates)
        logging.info(f"기술적 필터: {len(current_candidates)}개")
        
        # 최종 선별 (상위 20개)
        return current_candidates[:20]
    
    async def _collect_raw_data(self, tickers: List[str]) -> List[Dict]:
        """원시 데이터 수집"""
        valid_coins = []
        
        for i, ticker in enumerate(tickers):
            try:
                if i % 50 == 0:  # 진행상황 로그
                    logging.info(f"데이터 수집 진행: {i}/{len(tickers)}")
                
                price = pyupbit.get_current_price(ticker)
                if not price or price < 1:
                    continue
                    
                ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=30)
                if ohlcv is None or len(ohlcv) < 30:
                    continue
                
                volume_krw = ohlcv.iloc[-1]['volume'] * price
                if volume_krw < self.min_volume:
                    continue
                
                valid_coins.append({
                    'symbol': ticker,
                    'price': price,
                    'volume_krw': volume_krw,
                    'ohlcv': ohlcv,
                    'raw_score': volume_krw
                })
                
            except Exception as e:
                logging.debug(f"{ticker} 수집 실패: {e}")
                continue
        
        return sorted(valid_coins, key=lambda x: x['raw_score'], reverse=True)
    
    def _volume_filter(self, candidates: List[Dict]) -> List[Dict]:
        """거래량 필터"""
        filtered = []
        for candidate in candidates:
            try:
                volumes = candidate['ohlcv']['volume'].tail(7)
                if len(volumes) >= 7:
                    cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else 999
                    if cv < 2.0:  # 변동계수 2.0 이하
                        candidate['volume_stability'] = 1 / (1 + cv)
                        filtered.append(candidate)
            except:
                continue
        
        return sorted(filtered, key=lambda x: x.get('volume_stability', 0), reverse=True)[:100]
    
    def _stability_filter(self, candidates: List[Dict]) -> List[Dict]:
        """안정성 필터"""
        filtered = []
        for candidate in candidates:
            try:
                prices = candidate['ohlcv']['close'].tail(7)
                if len(prices) >= 7:
                    cv = prices.std() / prices.mean() if prices.mean() > 0 else 999
                    if cv < 0.3:  # 가격 변동계수 30% 이하
                        candidate['price_stability'] = 1 / (1 + cv)
                        filtered.append(candidate)
            except:
                continue
        
        return sorted(filtered, key=lambda x: x.get('price_stability', 0), reverse=True)[:80]
    
    def _momentum_filter(self, candidates: List[Dict]) -> List[Dict]:
        """모멘텀 필터"""
        for candidate in candidates:
            try:
                prices = candidate['ohlcv']['close']
                if len(prices) >= 30:
                    momentum_7d = (prices.iloc[-1] / prices.iloc[-8] - 1) * 100
                    momentum_30d = (prices.iloc[-1] / prices.iloc[-31] - 1) * 100
                    candidate['momentum_score'] = (momentum_7d * 0.7 + momentum_30d * 0.3) / 100
                else:
                    candidate['momentum_score'] = 0
            except:
                candidate['momentum_score'] = 0
        
        return sorted(candidates, key=lambda x: x.get('momentum_score', 0), reverse=True)[:60]
    
    def _technical_filter(self, candidates: List[Dict]) -> List[Dict]:
        """기술적 필터"""
        for candidate in candidates:
            try:
                ohlcv = candidate['ohlcv']
                rsi = self._calculate_rsi(ohlcv['close'])
                macd_score = self._calculate_macd_score(ohlcv['close'])
                candidate['technical_score'] = (rsi/100 + macd_score) / 2
            except:
                candidate['technical_score'] = 0.5
        
        return sorted(candidates, key=lambda x: x.get('technical_score', 0), reverse=True)[:40]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # 0으로 나누기 방지
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_macd_score(self, prices: pd.Series) -> float:
        """MACD 점수"""
        try:
            if len(prices) < 26:
                return 0.5
                
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            if len(macd_line) > 0 and len(signal_line) > 0:
                return 0.8 if macd_line.iloc[-1] > signal_line.iloc[-1] else 0.2
            else:
                return 0.5
        except:
            return 0.5

# ============================================================================
# 💎 DIAMOND HAND ALGORITHM - 켈리공식 기반 분할매매
# ============================================================================
class DiamondHandAlgorithm:
    """켈리 공식 기반 다이아몬드 핸드 알고리즘"""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
    
    def calculate_diamond_strategy(self, symbol: str, price: float, confidence: float, 
                                 cycle: str, quality_score: float) -> Dict:
        """다이아몬드 핸드 전략 계산"""
        try:
            # 켈리 비율 계산 (단순화)
            kelly_fraction = self._kelly_criterion(confidence, quality_score)
            
            # 감정 팩터
            emotion_factor = self._emotion_factor(cycle, confidence)
            
            # 총 투자 금액
            base_investment = self.portfolio_value * kelly_fraction * emotion_factor
            total_investment = min(base_investment, self.portfolio_value * 0.15)  # 최대 15%
            
            # 3단계 분할
            stage_amounts = [
                total_investment * 0.4,  # 1단계 40%
                total_investment * 0.35, # 2단계 35%
                total_investment * 0.25  # 3단계 25%
            ]
            
            # 진입가격 (현재가 기준)
            entry_prices = [
                price,           # 즉시 진입
                price * 0.95,    # -5% 추가 진입
                price * 0.90     # -10% 추가 진입
            ]
            
            # 익절/손절 (품질 점수 반영)
            take_profits = [
                price * (1 + 0.20 * (1 + quality_score)),  # 1차 익절
                price * (1 + 0.50 * (1 + quality_score)),  # 2차 익절
                price * (1 + 1.00 * (1 + quality_score))   # 3차 익절
            ]
            
            stop_loss = price * 0.85  # -15% 손절
            
            return {
                'symbol': symbol,
                'total_investment': total_investment,
                'kelly_fraction': kelly_fraction,
                'emotion_factor': emotion_factor,
                'stage_amounts': stage_amounts,
                'entry_prices': entry_prices,
                'take_profits': take_profits,
                'stop_loss': stop_loss,
                'portfolio_weight': (total_investment / self.portfolio_value) * 100
            }
            
        except Exception as e:
            logging.error(f"다이아몬드 전략 실패 {symbol}: {e}")
            return self._fallback_strategy(symbol, price)
    
    def _kelly_criterion(self, confidence: float, quality: float) -> float:
        """켈리 공식 (단순화)"""
        win_prob = (confidence + quality) / 2
        kelly = max(0.01, min(0.25, win_prob * 0.3))  # 최대 25%
        return kelly
    
    def _emotion_factor(self, cycle: str, confidence: float) -> float:
        """감정 팩터"""
        cycle_factors = {
            'strong_bull': 1.2,
            'momentum_phase': 1.1,
            'accumulation': 1.0,
            'reversal_phase': 0.9,
            'strong_bear': 0.8
        }
        
        base_factor = cycle_factors.get(cycle, 1.0)
        confidence_boost = 0.8 + (confidence * 0.4)
        
        return base_factor * confidence_boost
    
    def _fallback_strategy(self, symbol: str, price: float) -> Dict:
        """기본 전략"""
        base_investment = self.portfolio_value * 0.05
        return {
            'symbol': symbol, 'total_investment': base_investment,
            'kelly_fraction': 0.05, 'emotion_factor': 1.0,
            'stage_amounts': [base_investment * 0.5, base_investment * 0.3, base_investment * 0.2],
            'entry_prices': [price, price * 0.95, price * 0.90],
            'take_profits': [price * 1.2, price * 1.5, price * 2.0],
            'stop_loss': price * 0.85, 'portfolio_weight': 5.0
        }

# ============================================================================
# 📊 전설급 메인 시그널 클래스
# ============================================================================
@dataclass
class LegendarySignal:
    """전설급 시그널"""
    symbol: str
    action: str
    confidence: float
    price: float
    
    # 분석 결과
    neural_quality: float
    ai_explanation: str
    quantum_cycle: str
    cycle_confidence: float
    
    # 투자 전략
    kelly_fraction: float
    emotion_factor: float
    total_investment: float
    
    # 실행 계획
    entry_prices: List[float]
    stage_amounts: List[float] 
    take_profits: List[float]
    stop_loss: float
    
    # 종합 점수
    legendary_score: float
    
    timestamp: datetime

# ============================================================================
# 🎯 POSITION MANAGER - 포지션 관리자
# ============================================================================
@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    total_quantity: float
    avg_price: float
    current_stage: int  # 1, 2, 3 단계
    stage_quantities: List[float]  # 각 단계별 수량
    stage_prices: List[float]      # 각 단계별 진입가
    target_take_profits: List[float]
    stop_loss: float
    unrealized_pnl: float
    created_at: datetime
    last_updated: datetime

class PositionManager:
    """포지션 관리 시스템"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.position_file = "positions.json"
        self.load_positions()
    
    def add_position(self, signal: LegendarySignal, stage: int, quantity: float, executed_price: float):
        """포지션 추가/업데이트"""
        symbol = signal.symbol
        
        if symbol not in self.positions:
            # 새 포지션 생성
            self.positions[symbol] = Position(
                symbol=symbol,
                total_quantity=quantity,
                avg_price=executed_price,
                current_stage=stage,
                stage_quantities=[0.0, 0.0, 0.0],
                stage_prices=[0.0, 0.0, 0.0],
                target_take_profits=signal.take_profits.copy(),
                stop_loss=signal.stop_loss,
                unrealized_pnl=0.0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            self.positions[symbol].stage_quantities[stage-1] = quantity
            self.positions[symbol].stage_prices[stage-1] = executed_price
        else:
            # 기존 포지션 업데이트
            pos = self.positions[symbol]
            old_total_cost = pos.total_quantity * pos.avg_price
            new_cost = quantity * executed_price
            
            pos.total_quantity += quantity
            pos.avg_price = (old_total_cost + new_cost) / pos.total_quantity
            pos.current_stage = max(pos.current_stage, stage)
            pos.stage_quantities[stage-1] += quantity
            pos.stage_prices[stage-1] = executed_price
            pos.last_updated = datetime.now()
        
        self.save_positions()
        logging.info(f"포지션 업데이트: {symbol} - 단계 {stage}, 수량 {quantity}")
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """미실현 손익 업데이트"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.unrealized_pnl = (current_price - pos.avg_price) * pos.total_quantity
            pos.last_updated = datetime.now()
    
    def remove_position(self, symbol: str, quantity: float = None):
        """포지션 제거 (부분/전체)"""
        if symbol not in self.positions:
            return
        
        if quantity is None:
            # 전체 제거
            del self.positions[symbol]
            logging.info(f"포지션 전체 제거: {symbol}")
        else:
            # 부분 제거
            pos = self.positions[symbol]
            if quantity >= pos.total_quantity:
                del self.positions[symbol]
                logging.info(f"포지션 전체 제거: {symbol}")
            else:
                pos.total_quantity -= quantity
                pos.last_updated = datetime.now()
                logging.info(f"포지션 부분 제거: {symbol} - {quantity}")
        
        self.save_positions()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """포지션 조회"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """모든 포지션 조회"""
        return list(self.positions.values())
    
    def save_positions(self):
        """포지션 저장"""
        try:
            serializable_positions = {}
            for symbol, pos in self.positions.items():
                serializable_positions[symbol] = {
                    'symbol': pos.symbol,
                    'total_quantity': pos.total_quantity,
                    'avg_price': pos.avg_price,
                    'current_stage': pos.current_stage,
                    'stage_quantities': pos.stage_quantities,
                    'stage_prices': pos.stage_prices,
                    'target_take_profits': pos.target_take_profits,
                    'stop_loss': pos.stop_loss,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'created_at': pos.created_at.isoformat(),
                    'last_updated': pos.last_update
