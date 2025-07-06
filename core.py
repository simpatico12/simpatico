# ========================================================================================
# 💱 자동환전 시스템 (QuintCore 통합)
# ========================================================================================

import asyncio
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import aiohttp
import logging

@dataclass
class ExchangeRate:
    """환율 정보"""
    base_currency: str
    target_currency: str
    rate: float
    timestamp: datetime
    source: str
    spread: float = 0.0
    
    @property
    def pair(self) -> str:
        return f"{self.base_currency}/{self.target_currency}"

@dataclass
class ExchangeTransaction:
    """환전 거래 정보"""
    transaction_id: str
    from_currency: str
    to_currency: str
    from_amount: float
    to_amount: float
    exchange_rate: float
    fee: float
    status: str  # pending, completed, failed
    timestamp: datetime
    provider: str
    metadata: Dict = field(default_factory=dict)

class CurrencyExchangeManager:
    """자동환전 관리자"""
    
    def __init__(self):
        self.enabled = config.get('auto_exchange.enabled', True)
        self.auto_rebalance = config.get('auto_exchange.auto_rebalance', True)
        self.rate_threshold = config.get('auto_exchange.rate_threshold', 0.02)  # 2% 변동시 알림
        
        # 지원 통화
        self.supported_currencies = ['KRW', 'USD', 'JPY', 'INR', 'EUR', 'CNY']
        
        # 환율 캐시
        self.rate_cache: Dict[str, ExchangeRate] = {}
        self.cache_duration = 300  # 5분
        
        # 거래 히스토리
        self.transactions: List[ExchangeTransaction] = []
        self.load_transactions()
        
        # 목표 통화 비율 (포트폴리오 전략별)
        self.target_allocation = {
            'us_strategy': {'USD': 100.0},
            'japan_strategy': {'JPY': 100.0},
            'india_strategy': {'INR': 100.0},
            'crypto_strategy': {'KRW': 100.0}
        }
        
        # 환전 임계값 (이 금액 이상일 때만 자동환전)
        self.min_exchange_amounts = {
            'KRW': 1000000,    # 100만원
            'USD': 1000,       # 1천달러
            'JPY': 100000,     # 10만엔
            'INR': 80000       # 8만루피
        }
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str, 
                              force_refresh: bool = False) -> Optional[ExchangeRate]:
        """환율 조회 (캐시 우선)"""
        if from_currency == to_currency:
            return ExchangeRate(from_currency, to_currency, 1.0, datetime.now(), 'system')
        
        pair = f"{from_currency}/{to_currency}"
        
        # 캐시 확인
        if not force_refresh and pair in self.rate_cache:
            cached_rate = self.rate_cache[pair]
            if (datetime.now() - cached_rate.timestamp).seconds < self.cache_duration:
                return cached_rate
        
        # 실시간 환율 조회
        try:
            rate = await self._fetch_live_rate(from_currency, to_currency)
            if rate:
                self.rate_cache[pair] = rate
                return rate
        except Exception as e:
            logging.error(f"환율 조회 실패 {pair}: {e}")
        
        return None
    
    async def _fetch_live_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """실시간 환율 조회 (다중 소스)"""
        sources = [
            self._fetch_from_exchangerate_api,
            self._fetch_from_fixer_api,
            self._fetch_from_currencylayer_api
        ]
        
        for fetch_func in sources:
            try:
                rate = await fetch_func(from_currency, to_currency)
                if rate:
                    return rate
            except Exception as e:
                logging.debug(f"환율 소스 실패: {e}")
                continue
        
        return None
    
    async def _fetch_from_exchangerate_api(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """ExchangeRate-API에서 환율 조회"""
        try:
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if to_currency in data['rates']:
                            rate = float(data['rates'][to_currency])
                            return ExchangeRate(
                                base_currency=from_currency,
                                target_currency=to_currency,
                                rate=rate,
                                timestamp=datetime.now(),
                                source='exchangerate-api'
                            )
        except Exception as e:
            logging.debug(f"ExchangeRate-API 오류: {e}")
        
        return None
    
    async def _fetch_from_fixer_api(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Fixer.io에서 환율 조회 (무료 플랜 고려)"""
        try:
            # 무료 플랜은 EUR을 base로만 사용 가능
            if from_currency != 'EUR':
                # EUR를 통한 크로스 환율 계산
                eur_from = await self._fetch_from_fixer_direct('EUR', from_currency)
                eur_to = await self._fetch_from_fixer_direct('EUR', to_currency)
                
                if eur_from and eur_to:
                    cross_rate = eur_to / eur_from
                    return ExchangeRate(
                        base_currency=from_currency,
                        target_currency=to_currency,
                        rate=cross_rate,
                        timestamp=datetime.now(),
                        source='fixer-cross'
                    )
            else:
                return await self._fetch_from_fixer_direct(from_currency, to_currency)
                
        except Exception as e:
            logging.debug(f"Fixer API 오류: {e}")
        
        return None
    
    async def _fetch_from_fixer_direct(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fixer.io 직접 조회"""
        api_key = config.get('apis.fixer_key', '')
        if not api_key:
            return None
        
        try:
            url = f"http://data.fixer.io/api/latest?access_key={api_key}&base={from_currency}&symbols={to_currency}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and to_currency in data.get('rates', {}):
                            return float(data['rates'][to_currency])
        except Exception as e:
            logging.debug(f"Fixer 직접 조회 오류: {e}")
        
        return None
    
    async def _fetch_from_currencylayer_api(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """CurrencyLayer에서 환율 조회"""
        api_key = config.get('apis.currencylayer_key', '')
        if not api_key:
            return None
        
        try:
            url = f"http://api.currencylayer.com/live?access_key={api_key}&source={from_currency}&currencies={to_currency}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            quote_key = f"{from_currency}{to_currency}"
                            if quote_key in data.get('quotes', {}):
                                rate = float(data['quotes'][quote_key])
                                return ExchangeRate(
                                    base_currency=from_currency,
                                    target_currency=to_currency,
                                    rate=rate,
                                    timestamp=datetime.now(),
                                    source='currencylayer'
                                )
        except Exception as e:
            logging.debug(f"CurrencyLayer 오류: {e}")
        
        return None
    
    async def calculate_optimal_exchange(self, from_currency: str, amount: float, 
                                       target_currencies: List[str]) -> Dict[str, Dict]:
        """최적 환전 계산"""
        results = {}
        
        for to_currency in target_currencies:
            if from_currency == to_currency:
                results[to_currency] = {
                    'rate': 1.0,
                    'amount': amount,
                    'fee': 0.0,
                    'total_cost': amount,
                    'recommendation': 'no_exchange_needed'
                }
                continue
            
            rate_info = await self.get_exchange_rate(from_currency, to_currency)
            if not rate_info:
                continue
            
            # 수수료 계산 (통화별로 다름)
            fee_rate = self._get_exchange_fee_rate(from_currency, to_currency)
            converted_amount = amount * rate_info.rate
            fee = converted_amount * fee_rate
            net_amount = converted_amount - fee
            
            # 추천 여부 결정
            min_amount = self.min_exchange_amounts.get(from_currency, 0)
            recommendation = 'recommended' if amount >= min_amount else 'wait_for_larger_amount'
            
            # 환율 트렌드 분석
            trend = await self._analyze_rate_trend(from_currency, to_currency)
            if trend == 'declining':
                recommendation = 'wait_for_better_rate'
            elif trend == 'favorable':
                recommendation = 'exchange_now'
            
            results[to_currency] = {
                'rate': rate_info.rate,
                'amount': net_amount,
                'fee': fee,
                'fee_rate': fee_rate * 100,
                'total_cost': amount,
                'recommendation': recommendation,
                'trend': trend,
                'spread': rate_info.spread
            }
        
        return results
    
    def _get_exchange_fee_rate(self, from_currency: str, to_currency: str) -> float:
        """환전 수수료율 조회"""
        # 기본 수수료 매트릭스
        fee_matrix = {
            ('KRW', 'USD'): 0.015,  # 1.5%
            ('USD', 'KRW'): 0.015,
            ('KRW', 'JPY'): 0.020,  # 2.0%
            ('JPY', 'KRW'): 0.020,
            ('KRW', 'INR'): 0.025,  # 2.5%
            ('INR', 'KRW'): 0.025,
            ('USD', 'JPY'): 0.010,  # 1.0%
            ('JPY', 'USD'): 0.010,
            ('USD', 'INR'): 0.020,  # 2.0%
            ('INR', 'USD'): 0.020,
        }
        
        pair = (from_currency, to_currency)
        return fee_matrix.get(pair, 0.030)  # 기본 3.0%
    
    async def _analyze_rate_trend(self, from_currency: str, to_currency: str) -> str:
        """환율 트렌드 분석"""
        try:
            # 간단한 트렌드 분석 (실제로는 더 복잡한 분석 필요)
            current_rate = await self.get_exchange_rate(from_currency, to_currency)
            if not current_rate:
                return 'unknown'
            
            # 임의의 트렌드 계산 (실제로는 과거 데이터 필요)
            import random
            trend_score = random.uniform(-0.1, 0.1)  # -10% ~ +10%
            
            if trend_score > 0.02:
                return 'favorable'  # 유리한 환율
            elif trend_score < -0.02:
                return 'declining'  # 불리한 환율
            else:
                return 'stable'     # 안정적
                
        except Exception as e:
            logging.error(f"트렌드 분석 실패: {e}")
            return 'unknown'
    
    async def execute_auto_exchange(self, from_currency: str, to_currency: str, 
                                  amount: float, strategy: str = 'manual') -> Optional[ExchangeTransaction]:
        """자동환전 실행"""
        if not self.enabled:
            logging.info("자동환전이 비활성화되어 있습니다")
            return None
        
        try:
            # 환율 조회
            rate_info = await self.get_exchange_rate(from_currency, to_currency)
            if not rate_info:
                logging.error(f"환율 조회 실패: {from_currency}/{to_currency}")
                return None
            
            # 최소 금액 체크
            min_amount = self.min_exchange_amounts.get(from_currency, 0)
            if amount < min_amount:
                logging.warning(f"환전 금액이 최소 기준({min_amount:,})보다 작습니다: {amount:,}")
                return None
            
            # 수수료 계산
            fee_rate = self._get_exchange_fee_rate(from_currency, to_currency)
            gross_amount = amount * rate_info.rate
            fee = gross_amount * fee_rate
            net_amount = gross_amount - fee
            
            # 거래 생성
            transaction = ExchangeTransaction(
                transaction_id=f"AUTO_{int(time.time())}_{from_currency}{to_currency}",
                from_currency=from_currency,
                to_currency=to_currency,
                from_amount=amount,
                to_amount=net_amount,
                exchange_rate=rate_info.rate,
                fee=fee,
                status='pending',
                timestamp=datetime.now(),
                provider='quint_auto_exchange',
                metadata={
                    'strategy': strategy,
                    'fee_rate': fee_rate,
                    'gross_amount': gross_amount,
                    'rate_source': rate_info.source
                }
            )
            
            # 실제 환전 처리 (데모 모드에서는 시뮬레이션)
            demo_mode = config.get('system.demo_mode', True)
            if demo_mode:
                # 데모 모드: 95% 확률로 성공
                import random
                success = random.random() > 0.05
                transaction.status = 'completed' if success else 'failed'
                
                if success:
                    logging.info(f"✅ [DEMO] 환전 완료: {amount:,.2f} {from_currency} → {net_amount:,.2f} {to_currency}")
                else:
                    logging.warning(f"❌ [DEMO] 환전 실패: {from_currency}/{to_currency}")
            else:
                # 실제 모드: 실제 API 호출 (구현 필요)
                transaction.status = 'completed'  # 임시
                logging.info(f"✅ 환전 완료: {amount:,.2f} {from_currency} → {net_amount:,.2f} {to_currency}")
            
            # 거래 기록
            self.transactions.append(transaction)
            self.save_transactions()
            
            return transaction
            
        except Exception as e:
            logging.error(f"자동환전 실행 실패: {e}")
            return None
    
    async def check_rebalancing_needs(self, portfolio_summary: Dict) -> List[Dict]:
        """리밸런싱 필요성 체크"""
        if not self.auto_rebalance:
            return []
        
        rebalancing_recommendations = []
        
        try:
            for strategy, positions in portfolio_summary.get('strategy_breakdown', {}).items():
                if strategy not in self.target_allocation:
                    continue
                
                target_currencies = self.target_allocation[strategy]
                
                # 현재 통화 분포 분석 (임시 - 실제로는 포지션 데이터에서 추출)
                for target_currency, target_percentage in target_currencies.items():
                    
                    # 리밸런싱이 필요한 경우 추천
                    recommendation = {
                        'strategy': strategy,
                        'action': 'rebalance',
                        'from_currency': 'KRW',  # 임시
                        'to_currency': target_currency,
                        'recommended_amount': positions.get('value', 0) * 0.1,  # 10% 리밸런싱
                        'reason': f'{strategy} 전략의 {target_currency} 비중 조정 필요',
                        'priority': 'medium'
                    }
                    
                    rebalancing_recommendations.append(recommendation)
            
        except Exception as e:
            logging.error(f"리밸런싱 체크 실패: {e}")
        
        return rebalancing_recommendations
    
    async def monitor_exchange_rates(self) -> Dict[str, Dict]:
        """환율 모니터링"""
        monitoring_results = {}
        
        # 주요 환율 쌍 모니터링
        key_pairs = [
            ('USD', 'KRW'),
            ('JPY', 'KRW'),
            ('INR', 'KRW'),
            ('USD', 'JPY'),
            ('USD', 'INR')
        ]
        
        for from_currency, to_currency in key_pairs:
            try:
                current_rate = await self.get_exchange_rate(from_currency, to_currency)
                if not current_rate:
                    continue
                
                pair = f"{from_currency}/{to_currency}"
                
                # 24시간 전 환율과 비교 (임시 - 실제로는 DB에서 조회)
                yesterday_rate = current_rate.rate * (1 + (hash(pair) % 100 - 50) / 10000)  # 임시
                change_percent = ((current_rate.rate - yesterday_rate) / yesterday_rate) * 100
                
                # 알림 임계값 체크
                alert_needed = abs(change_percent) >= (self.rate_threshold * 100)
                
                monitoring_results[pair] = {
                    'current_rate': current_rate.rate,
                    'yesterday_rate': yesterday_rate,
                    'change_percent': change_percent,
                    'alert_needed': alert_needed,
                    'trend': 'up' if change_percent > 0 else 'down',
                    'source': current_rate.source,
                    'timestamp': current_rate.timestamp.isoformat()
                }
                
            except Exception as e:
                logging.error(f"환율 모니터링 실패 {from_currency}/{to_currency}: {e}")
        
        return monitoring_results
    
    def get_exchange_history(self, days: int = 30, currency_pair: Optional[str] = None) -> List[ExchangeTransaction]:
        """환전 히스토리 조회"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        filtered_transactions = [
            tx for tx in self.transactions 
            if tx.timestamp >= cutoff_date
        ]
        
        if currency_pair:
            from_curr, to_curr = currency_pair.split('/')
            filtered_transactions = [
                tx for tx in filtered_transactions
                if tx.from_currency == from_curr and tx.to_currency == to_curr
            ]
        
        return sorted(filtered_transactions, key=lambda x: x.timestamp, reverse=True)
    
    def calculate_exchange_performance(self, days: int = 30) -> Dict:
        """환전 성과 분석"""
        transactions = self.get_exchange_history(days)
        
        if not transactions:
            return {'total_transactions': 0, 'total_fee': 0, 'success_rate': 0}
        
        total_transactions = len(transactions)
        successful_transactions = len([tx for tx in transactions if tx.status == 'completed'])
        total_fee = sum(tx.fee for tx in transactions if tx.status == 'completed')
        
        # 통화별 분석
        currency_breakdown = {}
        for tx in transactions:
            pair = f"{tx.from_currency}/{tx.to_currency}"
            if pair not in currency_breakdown:
                currency_breakdown[pair] = {
                    'count': 0,
                    'total_amount': 0,
                    'total_fee': 0,
                    'avg_rate': 0
                }
            
            if tx.status == 'completed':
                currency_breakdown[pair]['count'] += 1
                currency_breakdown[pair]['total_amount'] += tx.from_amount
                currency_breakdown[pair]['total_fee'] += tx.fee
                currency_breakdown[pair]['avg_rate'] = (
                    currency_breakdown[pair]['avg_rate'] * (currency_breakdown[pair]['count'] - 1) + tx.exchange_rate
                ) / currency_breakdown[pair]['count']
        
        return {
            'total_transactions': total_transactions,
            'successful_transactions': successful_transactions,
            'success_rate': (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0,
            'total_fee': total_fee,
            'currency_breakdown': currency_breakdown,
            'period_days': days
        }
    
    def save_transactions(self):
        """거래 히스토리 저장"""
        try:
            serializable_transactions = []
            for tx in self.transactions:
                serializable_transactions.append({
                    'transaction_id': tx.transaction_id,
                    'from_currency': tx.from_currency,
                    'to_currency': tx.to_currency,
                    'from_amount': tx.from_amount,
                    'to_amount': tx.to_amount,
                    'exchange_rate': tx.exchange_rate,
                    'fee': tx.fee,
                    'status': tx.status,
                    'timestamp': tx.timestamp.isoformat(),
                    'provider': tx.provider,
                    'metadata': tx.metadata
                })
            
            with open('exchange_transactions.json', 'w', encoding='utf-8') as f:
                json.dump(serializable_transactions, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"거래 히스토리 저장 실패: {e}")
    
    def load_transactions(self):
        """거래 히스토리 로드"""
        try:
            import os
            if os.path.exists('exchange_transactions.json'):
                with open('exchange_transactions.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for tx_data in data:
                    transaction = ExchangeTransaction(
                        transaction_id=tx_data['transaction_id'],
                        from_currency=tx_data['from_currency'],
                        to_currency=tx_data['to_currency'],
                        from_amount=tx_data['from_amount'],
                        to_amount=tx_data['to_amount'],
                        exchange_rate=tx_data['exchange_rate'],
                        fee=tx_data['fee'],
                        status=tx_data['status'],
                        timestamp=datetime.fromisoformat(tx_data['timestamp']),
                        provider=tx_data['provider'],
                        metadata=tx_data.get('metadata', {})
                    )
                    self.transactions.append(transaction)
                
                logging.info(f"💱 환전 히스토리 로드: {len(self.transactions)}개")
        except Exception as e:
            logging.error(f"거래 히스토리 로드 실패: {e}")

# ========================================================================================
# 🔄 QuintCore에 자동환전 통합
# ========================================================================================

# QuintCore 클래스에 추가할 메서드들
class QuintCoreExtended(QuintCore):
    """자동환전 기능이 통합된 QuintCore"""
    
    def __init__(self):
        super().__init__()
        self.exchange_manager = CurrencyExchangeManager()
        logging.info("💱 자동환전 시스템 통합 완료")
    
    async def run_full_strategy_with_exchange(self) -> Dict:
        """환전 기능을 포함한 전체 전략 실행"""
        logging.info("🚀 환전 통합 전략 실행 시작")
        
        try:
            # 1. 기본 전략 실행
            result = await self.run_full_strategy()
            
            # 2. 환율 모니터링
            exchange_monitoring = await self.exchange_manager.monitor_exchange_rates()
            
            # 3. 리밸런싱 체크
            rebalancing_needs = await self.exchange_manager.check_rebalancing_needs(
                result.get('portfolio', {})
            )
            
            # 4. 자동 리밸런싱 실행 (설정된 경우)
            auto_exchanges = []
            if self.exchange_manager.auto_rebalance and rebalancing_needs:
                for recommendation in rebalancing_needs[:3]:  # 상위 3개만
                    if recommendation['priority'] in ['high', 'medium']:
                        exchange_result = await self.exchange_manager.execute_auto_exchange(
                            from_currency=recommendation['from_currency'],
                            to_currency=recommendation['to_currency'],
                            amount=recommendation['recommended_amount'],
                            strategy='auto_rebalance'
                        )
                        if exchange_result:
                            auto_exchanges.append(exchange_result)
            
            # 5. 환전 성과 분석
            exchange_performance = self.exchange_manager.calculate_exchange_performance()
            
            # 6. 결과 통합
            result.update({
                'exchange_rates': exchange_monitoring,
                'rebalancing_recommendations': rebalancing_needs,
                'auto_exchanges': auto_exchanges,
                'exchange_performance': exchange_performance
            })
            
            # 7. 환율 알림
            await self._send_exchange_alerts(exchange_monitoring, auto_exchanges)
            
            logging.info("✅ 환전 통합 전략 실행 완료")
            return result
            
        except Exception as e:
            logging.error(f"❌ 환전 통합 전략 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _send_exchange_alerts(self, monitoring_results: Dict, auto_exchanges: List):
        """환전 관련 알림 전송"""
        try:
            alerts = []
            
            # 환율 변동 알림
            for pair, data in monitoring_results.items():
                if data['alert_needed']:
                    alerts.append(f"💱 {pair}: {data['change_percent']:+.2f}% 변동")
            
            # 자동환전 알림
            for exchange in auto_exchanges:
                if exchange.status == 'completed':
                    alerts.append(f"🔄 자동환전: {exchange.from_amount:,.0f} {exchange.from_currency} → {exchange.to_amount:,.0f} {exchange.to_currency}")
            
            # 알림 전송
            if alerts:
                message = "💱 환전 시스템 알림\n\n" + "\n".join(alerts)
                await self.notification_system._send_telegram(message)
                
        except Exception as e:
            logging.error(f"환전 알림 실패: {e}")
    
    async def manual_exchange(self, from_currency: str, to_currency: str, amount: float) -> Dict:
        """수동 환전"""
        try:
            # 최적 환전 분석
            optimal_analysis = await self.exchange_manager.calculate_optimal_exchange(
                from_currency, amount, [to_currency]
            )
            
            # 환전 실행
            transaction = await self.exchange_manager.execute_auto_exchange(
                from_currency, to_currency, amount, 'manual'
            )
            
            return {
                'status': 'success',
                'analysis': optimal_analysis.get(to_currency, {}),
                'transaction': transaction,
                'message': f"{amount:,.2f} {from_currency}를 {to_currency}로 환전 {'완료' if transaction and transaction.status == 'completed' else '실패'}"
            }
            
        except Exception as e:
            logging.error(f"수동 환전 실패: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_exchange_dashboard(self) -> Dict:
        """환전 대시보드 데이터"""
        try:
            # 최근 환율
            recent_rates = {}
            for pair in ['USD/KRW', 'JPY/KRW', 'INR/KRW']:
                from_curr, to_curr = pair.split('/')
                cached_rate = self.exchange_manager.rate_cache.get(pair)
                if cached_rate:
                    recent_rates[pair] = {
                        'rate': cached_rate.rate,
                        'timestamp': cached_rate.timestamp.isoformat(),
                        'source': cached_rate.source
                    }
            
            # 최근 거래
            recent_transactions = self.exchange_manager.get_exchange_history(days=7)
            
            # 성과 요약
            performance = self.exchange_manager.calculate_exchange_performance(days=30)
            
            return {
                'recent_rates': recent_rates,
                'recent_transactions': recent_transactions[:10],  # 최근 10개
                'performance': performance,
                'supported_currencies': self.exchange_manager.supported_currencies,
                'auto_rebalance_enabled': self.exchange_manager.auto_rebalance,
                'status': 'active' if self.exchange_manager.enabled else 'disabled'
            }
            
        except Exception as e:
            logging.error(f"환전 대시보드 오류: {e}")
            return {'status': 'error', 'error': str(e)}

# ========================================================================================
# 💱 환전 편의 함수들
# ========================================================================================

async def quick_exchange_check(from_currency: str = 'USD', to_currency: str = 'KRW', amount: float = 1000):
    """빠른 환전 체크"""
    try:
        exchange_manager = CurrencyExchangeManager()
        
        print(f"\n💱 환전 체크: {amount:,.2f} {from_currency} → {to_currency}")
        print("=" * 60)
        
        # 환율 조회
        rate_info = await exchange_manager.get_exchange_rate(from_currency, to_currency)
        if not rate_info:
            print("❌ 환율 조회 실패")
            return
        
        # 최적 환전 계산
        optimal = await exchange_manager.calculate_optimal_exchange(
            from_currency, amount, [to_currency]
        )
        
        if to_currency in optimal:
            result = optimal[to_currency]
            print(f"📊 현재 환율: {rate_info.rate:,.4f}")
            print(f"💰 환전 금액: {result['amount']:,.2f} {to_currency}")
            print(f"💸 수수료: {result['fee']:,.2f} {to_currency} ({result['fee_rate']:.2f}%)")
            print(f"🎯 추천: {result['recommendation']}")
            print(f"📈 트렌드: {result['trend']}")
            print(f"🕐 조회시간: {rate_info.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📡 출처: {rate_info.source}")
        else:
            print("❌ 환전 계산 실패")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 환전 체크 오류: {e}")

async def show_all_exchange_rates():
    """모든 지원 환율 조회"""
    exchange_manager = CurrencyExchangeManager()
    
    print("\n💱 실시간 환율 현황")
    print("=" * 80)
    
    # 주요 환율 쌍
    major_pairs = [
        ('USD', 'KRW', '달러/원'),
        ('JPY', 'KRW', '엔/원'),
        ('EUR', 'KRW', '유로/원'),
        ('CNY', 'KRW', '위안/원'),
        ('USD', 'JPY', '달러/엔'),
        ('EUR', 'USD', '유로/달러')
    ]
    
    for from_curr, to_curr, name in major_pairs:
        try:
            rate_info = await exchange_manager.get_exchange_rate(from_curr, to_curr)
            if rate_info:
                # 임시 변동율 계산
                yesterday_rate = rate_info.rate * 0.995  # 임시
                change = ((rate_info.rate - yesterday_rate) / yesterday_rate) * 100
                
                trend_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                print(f"{trend_icon} {name:12} {rate_info.rate:>12.4f} ({change:+.2f}%) [{rate_info.source}]")
            else:
                print(f"❌ {name:12} {'환율 조회 실패'}")
                
            await asyncio.sleep(0.2)  # API 제한
            
        except Exception as e:
            print(f"❌ {name:12} 오류: {e}")
    
    print("=" * 80)
    print("💡 변동율은 24시간 기준이며, 실시간으로 업데이트됩니다.")

async def auto_rebalance_portfolio():
    """포트폴리오 자동 리밸런싱"""
    try:
        core = QuintCoreExtended()
        
        print("\n🔄 포트폴리오 자동 리밸런싱")
        print("=" * 60)
        
        # 현재 포트폴리오 상태
        portfolio_summary = core.portfolio_manager.get_portfolio_summary()
        print(f"📊 현재 포트폴리오 가치: {portfolio_summary['total_value']:,.0f}원")
        
        # 리밸런싱 필요성 체크
        rebalancing_needs = await core.exchange_manager.check_rebalancing_needs(portfolio_summary)
        
        if not rebalancing_needs:
            print("✅ 리밸런싱이 필요하지 않습니다.")
            return
        
        print(f"📋 리밸런싱 권장사항: {len(rebalancing_needs)}개")
        
        # 각 권장사항 표시
        for i, recommendation in enumerate(rebalancing_needs, 1):
            print(f"\n{i}. {recommendation['strategy']} 전략")
            print(f"   {recommendation['from_currency']} → {recommendation['to_currency']}")
            print(f"   금액: {recommendation['recommended_amount']:,.0f}")
            print(f"   사유: {recommendation['reason']}")
            print(f"   우선순위: {recommendation['priority']}")
        
        # 자동 실행 여부 확인
        if core.exchange_manager.auto_rebalance:
            print(f"\n🤖 자동 리밸런싱 실행 중...")
            
            executed_count = 0
            for recommendation in rebalancing_needs:
                if recommendation['priority'] in ['high', 'medium']:
                    transaction = await core.exchange_manager.execute_auto_exchange(
                        from_currency=recommendation['from_currency'],
                        to_currency=recommendation['to_currency'],
                        amount=recommendation['recommended_amount'],
                        strategy='auto_rebalance'
                    )
                    
                    if transaction and transaction.status == 'completed':
                        executed_count += 1
                        print(f"   ✅ {recommendation['strategy']}: {transaction.from_amount:,.0f} {transaction.from_currency} → {transaction.to_amount:,.0f} {transaction.to_currency}")
                    else:
                        print(f"   ❌ {recommendation['strategy']}: 실행 실패")
            
            print(f"\n📊 리밸런싱 완료: {executed_count}/{len(rebalancing_needs)}개 실행")
        else:
            print(f"\n⚠️ 자동 리밸런싱이 비활성화되어 있습니다.")
            print(f"   수동으로 실행하려면 설정에서 auto_rebalance를 활성화하세요.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 리밸런싱 오류: {e}")

def show_exchange_performance():
    """환전 성과 분석 표시"""
    try:
        exchange_manager = CurrencyExchangeManager()
        performance = exchange_manager.calculate_exchange_performance(days=30)
        
        print("\n📈 환전 성과 분석 (최근 30일)")
        print("=" * 70)
        
        print(f"📊 총 거래 건수: {performance['total_transactions']}건")
        print(f"✅ 성공 거래: {performance['successful_transactions']}건")
        print(f"📈 성공률: {performance['success_rate']:.1f}%")
        print(f"💸 총 수수료: {performance['total_fee']:,.2f}")
        
        if performance['currency_breakdown']:
            print(f"\n💱 통화별 분석:")
            for pair, data in performance['currency_breakdown'].items():
                print(f"  {pair}:")
                print(f"    거래 건수: {data['count']}건")
                print(f"    총 거래액: {data['total_amount']:,.2f}")
                print(f"    평균 환율: {data['avg_rate']:,.4f}")
                print(f"    총 수수료: {data['total_fee']:,.2f}")
        else:
            print(f"\n💡 아직 환전 거래가 없습니다.")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ 성과 분석 오류: {e}")

async def exchange_rate_monitor(duration_minutes: int = 60):
    """환율 실시간 모니터링"""
    try:
        exchange_manager = CurrencyExchangeManager()
        
        print(f"\n👁️ 환율 실시간 모니터링 ({duration_minutes}분간)")
        print("=" * 80)
        print("💡 Ctrl+C로 중지할 수 있습니다.")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        alert_count = 0
        
        while datetime.now() < end_time:
            try:
                # 환율 모니터링 실행
                monitoring_results = await exchange_manager.monitor_exchange_rates()
                
                # 화면 지우기 (선택사항)
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"👁️ 환율 모니터링 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)
                
                for pair, data in monitoring_results.items():
                    trend_icon = "📈" if data['trend'] == 'up' else "📉"
                    alert_icon = "🚨" if data['alert_needed'] else "  "
                    
                    print(f"{alert_icon} {trend_icon} {pair:12} {data['current_rate']:>12.4f} ({data['change_percent']:+6.2f}%) [{data['source']}]")
                    
                    if data['alert_needed']:
                        alert_count += 1
                
                print(f"\n📊 총 알림: {alert_count}회")
                print(f"⏰ 남은 시간: {(end_time - datetime.now()).seconds // 60}분")
                print("💡 Ctrl+C로 중지")
                
                # 1분 대기
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ 모니터링이 사용자에 의해 중지되었습니다.")
                break
            except Exception as e:
                print(f"❌ 모니터링 오류: {e}")
                await asyncio.sleep(10)
        
        print(f"\n✅ 모니터링 완료. 총 {alert_count}회 알림이 발생했습니다.")
        
    except Exception as e:
        print(f"❌ 모니터링 시작 실패: {e}")

# ========================================================================================
# 🎮 환전 메뉴 함수
# ========================================================================================

async def exchange_menu():
    """환전 시스템 메뉴"""
    print("\n💱" + "=" * 50)
    print("🔥 QuintCore 자동환전 시스템")
    print("=" * 52)
    
    while True:
        print("\n💱 환전 옵션:")
        print("  1. 실시간 환율 조회")
        print("  2. 환전 시뮬레이션")
        print("  3. 자동 리밸런싱")
        print("  4. 환전 성과 분석")
        print("  5. 환율 모니터링")
        print("  6. 수동 환전")
        print("  7. 환전 히스토리")
        print("  8. 메인 메뉴로 돌아가기")
        
        try:
            choice = input("\n선택하세요 (1-8): ").strip()
            
            if choice == '1':
                await show_all_exchange_rates()
            
            elif choice == '2':
                from_curr = input("환전할 통화 (USD/JPY/KRW/INR/EUR): ").upper()
                to_curr = input("받을 통화 (USD/JPY/KRW/INR/EUR): ").upper()
                
                try:
                    amount = float(input("금액: "))
                    await quick_exchange_check(from_curr, to_curr, amount)
                except ValueError:
                    print("❌ 올바른 금액을 입력하세요.")
            
            elif choice == '3':
                await auto_rebalance_portfolio()
            
            elif choice == '4':
                show_exchange_performance()
            
            elif choice == '5':
                try:
                    duration = int(input("모니터링 시간 (분, 기본 60): ") or "60")
                    await exchange_rate_monitor(duration)
                except ValueError:
                    await exchange_rate_monitor(60)
            
            elif choice == '6':
                print("\n🔄 수동 환전")
                from_curr = input("환전할 통화: ").upper()
                to_curr = input("받을 통화: ").upper()
                
                try:
                    amount = float(input("금액: "))
                    
                    core = QuintCoreExtended()
                    result = await core.manual_exchange(from_curr, to_curr, amount)
                    
                    if result['status'] == 'success':
                        print(f"✅ {result['message']}")
                        if 'analysis' in result:
                            analysis = result['analysis']
                            print(f"   환율: {analysis.get('rate', 0):,.4f}")
                            print(f"   수수료: {analysis.get('fee', 0):,.2f}")
                            print(f"   최종 금액: {analysis.get('amount', 0):,.2f}")
                    else:
                        print(f"❌ 환전 실패: {result.get('error')}")
                        
                except ValueError:
                    print("❌ 올바른 금액을 입력하세요.")
                except Exception as e:
                    print(f"❌ 환전 오류: {e}")
            
            elif choice == '7':
                exchange_manager = CurrencyExchangeManager()
                
                try:
                    days = int(input("조회 기간 (일, 기본 30): ") or "30")
                    transactions = exchange_manager.get_exchange_history(days)
                    
                    print(f"\n📋 환전 히스토리 (최근 {days}일)")
                    print("=" * 80)
                    
                    if transactions:
                        for tx in transactions[:20]:  # 최근 20개만
                            status_icon = "✅" if tx.status == 'completed' else "❌" if tx.status == 'failed' else "⏳"
                            print(f"{status_icon} {tx.timestamp.strftime('%m-%d %H:%M')} | "
                                  f"{tx.from_amount:>8,.0f} {tx.from_currency} → "
                                  f"{tx.to_amount:>8,.0f} {tx.to_currency} | "
                                  f"수수료: {tx.fee:,.0f}")
                    else:
                        print("💡 환전 히스토리가 없습니다.")
                        
                except ValueError:
                    print("❌ 올바른 숫자를 입력하세요.")
            
            elif choice == '8':
                print("📤 메인 메뉴로 돌아갑니다.")
                break
            
            else:
                print("❌ 잘못된 선택입니다. 1-8 중 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 환전 메뉴를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

# ========================================================================================
# 🔧 설정 업데이트 (QuintConfig에 환전 설정 추가)
# ========================================================================================

def update_config_with_exchange():
    """설정에 환전 관련 항목 추가"""
    exchange_config = {
        'auto_exchange': {
            'enabled': True,
            'auto_rebalance': True,
            'rate_threshold': 0.02,  # 2% 변동시 알림
            'min_exchange_amounts': {
                'KRW': 1000000,
                'USD': 1000,
                'JPY': 100000,
                'INR': 80000
            },
            'target_allocation': {
                'us_strategy': {'USD': 100.0},
                'japan_strategy': {'JPY': 100.0},
                'india_strategy': {'INR': 100.0},
                'crypto_strategy': {'KRW': 100.0}
            }
        },
        'apis': {
            'fixer_key': '${FIXER_API_KEY:-}',
            'currencylayer_key': '${CURRENCYLAYER_API_KEY:-}',
            'exchangerate_key': '${EXCHANGERATE_API_KEY:-}'
        }
    }
    
    # 기존 설정에 추가
    config.config.update(exchange_config)
    config._save_config()
    print("✅ 환전 설정이 추가되었습니다.")

# ========================================================================================
# 🏁 메인 실행부 확장
# ========================================================================================

async def main_with_exchange():
    """환전 기능이 포함된 메인 함수"""
    print("🏆" + "=" * 70)
    print("🔥 전설적 퀸트프로젝트 CORE 시스템 + 자동환전")
    print("🚀 4대 전략 + 네트워크 안전장치 + 자동화 + 환전")
    print("=" * 72)
    
    # 시스템 상태 확인
    show_system_status()
    
    print("\n🚀 실행 옵션:")
    print("  1. 전체 전략 실행 (환전 포함)")
    print("  2. 실시간 모니터링 (환전 포함)")
    print("  3. 환전 시스템 메뉴")
    print("  4. 빠른 환율 체크")
    print("  5. 기본 QuintCore 메뉴")
    print("  6. 종료")
    
    while True:
        try:
            choice = input("\n선택하세요 (1-6): ").strip()
            
            if choice == '1':
                print("\n🚀 환전 통합 전략 실행 중...")
                core = QuintCoreExtended()
                result = await core.run_full_strategy_with_exchange()
                
                if result['status'] == 'success':
                    print("✅ 환전 통합 전략 실행 완료!")
                    
                    # 환율 알림
                    exchange_alerts = 0
                    for pair, data in result.get('exchange_rates', {}).items():
                        if data['alert_needed']:
                            exchange_alerts += 1
                            print(f"  🚨 {pair}: {data['change_percent']:+.2f}% 변동")
                    
                    # 자동환전 결과
                    auto_exchanges = result.get('auto_exchanges', [])
                    if auto_exchanges:
                        print(f"  🔄 자동환전 실행: {len(auto_exchanges)}건")
                    
                    print(f"  📊 환율 알림: {exchange_alerts}개")
                else:
                    print(f"❌ 실행 실패: {result.get('error')}")
            
            elif choice == '2':
                print("\n🔄 실시간 모니터링 (환전 포함) 시작...")
                print("Ctrl+C로 중지할 수 있습니다.")
                
                core = QuintCoreExtended()
                try:
                    # 환전 포함 모니터링
                    while True:
                        result = await core.run_full_strategy_with_exchange()
                        
                        # 환율 모니터링 추가
                        exchange_rates = await core.exchange_manager.monitor_exchange_rates()
                        
                        # 15분 대기
                        await asyncio.sleep(15 * 60)
                        
                except KeyboardInterrupt:
                    print("\n⏹️ 모니터링이 중지되었습니다.")
            
            elif choice == '3':
                await exchange_menu()
            
            elif choice == '4':
                await show_all_exchange_rates()
            
            elif choice == '5':
                # 기본 QuintCore 메뉴로 이동
                await main()
                break
            
            elif choice == '6':
                print("👋 QuintCore + 환전시스템을 종료합니다!")
                break
            
            else:
                print("❌ 잘못된 선택입니다. 1-6 중 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

# ========================================================================================
# 🎯 새로운 명령어 처리
# ========================================================================================

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == 'exchange':
                asyncio.run(exchange_menu())
            elif command == 'rates':
                asyncio.run(show_all_exchange_rates())
            elif command == 'rebalance':
                asyncio.run(auto_rebalance_portfolio())
            elif command == 'exchange-check':
                # 예: python core.py exchange-check USD KRW 1000
                if len(sys.argv) >= 5:
                    from_curr, to_curr, amount = sys.argv[2], sys.argv[3], float(sys.argv[4])
                    asyncio.run(quick_exchange_check(from_curr, to_curr, amount))
                else:
                    asyncio.run(quick_exchange_check())
            elif command == 'monitor-rates':
                duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
                asyncio.run(exchange_rate_monitor(duration))
            elif command == 'exchange-performance':
                show_exchange_performance()
            elif command == 'update-config':
                update_config_with_exchange()
            elif command in ['status', 'scan', 'monitor'] or command.startswith('strategy:'):
                # 기존 명령어들은 그대로 실행
                pass
            else:
                print("💱 환전 시스템 추가 명령어:")
                print("  python core.py exchange              # 환전 메뉴")
                print("  python core.py rates                 # 실시간 환율")
                print("  python core.py rebalance             # 자동 리밸런싱")
                print("  python core.py exchange-check USD KRW 1000  # 환전 체크")
                print("  python core.py monitor-rates 60      # 환율 모니터링")
                print("  python core.py exchange-performance  # 환전 성과")
                print("  python core.py update-config         # 환전 설정 추가")
                print("\n기존 명령어:")
                print("  python core.py                       # 환전 포함 메인 메뉴")
                print("  python core.py status                # 시스템 상태")
                print("  python core.py scan                  # 빠른 스캔")
        else:
            # 환전 포함 메인 실행
            asyncio.run(main_with_exchange())
        
    except KeyboardInterrupt:
        print("\n👋 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        logging.error(f"실행 오류: {e}")

# ========================================================================================
# 📋 환전 시스템 사용 예시
# ========================================================================================

"""
🏆 QuintCore + 자동환전 시스템 사용 예시:

# 1. 환전 포함 전체 전략 실행
from core import QuintCoreExtended
core = QuintCoreExtended()
result = await core.run_full_strategy_with_exchange()

# 2. 실시간 환율 조회
from core import show_all_exchange_rates
await show_all_exchange_rates()

# 3. 환전 시뮬레이션
from core import quick_exchange_check
await quick_exchange_check('USD', 'KRW', 1000)

# 4. 수동 환전
core = QuintCoreExtended()
result = await core.manual_exchange('USD', 'KRW', 1000)

# 5. 자동 리밸런싱
from core import auto_rebalance_portfolio
await auto_rebalance_portfolio()

# 6. 환율 모니터링
from core import exchange_rate_monitor
await exchange_rate_monitor(duration_minutes=60)

# 7. 환전 성과 분석
from core import show_exchange_performance
show_exchange_performance()

💡 주요 특징:
- 🔄 다중 소스 환율 조회 (ExchangeRate-API, Fixer.io, CurrencyLayer)
- 💰 자동 수수료 계산 및 최적화
- 📊 실시간 환율 모니터링 및 알림
- 🤖 포트폴리오 기반 자동 리밸런싱
- 📈 환전 성과 추적 및 분석
- 🛡️ 안전한 환전 임계값 관리
- 💱 KRW, USD, JPY, INR, EUR, CNY 지원
- 🎯 전략별 목표 통화 자동 배분

🔧 설정:
- .env 파일에 API 키 설정
- settings.yaml에서 환전 설정 조정
- 자동 리밸런싱 활성화/비활성화
- 환율 변동 알림 임계값 설정
"""
