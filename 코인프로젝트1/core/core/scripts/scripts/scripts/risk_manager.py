# risk_manager.py
"""
고급 리스크 관리 시스템 - 퀸트프로젝트 수준
동적 리스크 조정, 포트폴리오 최적화, 실시간 모니터링
"""

import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from config import get_config
from notifier import notifier
from logger import logger
from db import db_manager
from exceptions import PositionLimitError


@dataclass
class RiskMetrics:
    """리스크 지표"""
    total_exposure: float
    max_position_size: float
    correlation_risk: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass
class Position:
    """포지션 정보"""
    asset: str
    asset_type: str
    quantity: float
    avg_price: float
    current_price: float
    value: float
    pnl: float
    pnl_percent: float
    weight: float  # 포트폴리오 비중


class AdvancedRiskManager:
    """고급 리스크 관리자"""
    
    def __init__(self):
        self.cfg = get_config()
        self.risk_limits = self._load_risk_limits()
        self.positions_cache = {}
        self.last_check = datetime.now()
        
    def _load_risk_limits(self) -> Dict:
        """리스크 한도 로드"""
        return {
            'max_portfolio_var': 0.05,  # 5% VaR
            'max_single_position': 0.3,  # 단일 종목 30%
            'max_sector_exposure': 0.5,  # 섹터별 50%
            'max_correlation': 0.7,      # 상관계수 0.7
            'min_sharpe_ratio': 0.5,     # 최소 샤프비율
            'max_drawdown': 0.2,         # 최대 낙폭 20%
            'min_liquidity_ratio': 0.1   # 최소 현금 10%
        }
    
    async def check_pre_trade_risk(self, asset: str, asset_type: str, 
                                  order_type: str, amount: float,
                                  total_value: float, cash: float) -> Tuple[bool, str]:
        """
        거래 전 리스크 체크 (종합)
        
        Returns:
            (승인여부, 이유)
        """
        checks = [
            self._check_cash_ratio(asset_type, cash, total_value),
            self._check_position_limit(asset, asset_type, amount, total_value),
            self._check_correlation_risk(asset, asset_type),
            self._check_volatility_adjusted_size(asset, amount),
            self._check_market_conditions(asset_type),
            await self._check_liquidity(asset, asset_type, amount)
        ]
        
        for passed, reason in checks:
            if not passed:
                await self._notify_risk_rejection(asset, reason)
                return False, reason
        
        return True, "모든 리스크 체크 통과"
    
    def _check_cash_ratio(self, asset_type: str, cash: float, 
                         total_value: float) -> Tuple[bool, str]:
        """현금 비율 체크"""
        if total_value <= 0:
            return False, "자산가치 오류"
        
        params = self.cfg['trading'][asset_type]
        min_cash_ratio = params.get('min_cash_ratio', 0.1)
        trading_pct = params.get('percentage', 20) / 100
        
        cash_ratio = cash / total_value
        
        # 동적 현금 비율 조정 (시장 변동성에 따라)
        volatility = self._get_market_volatility()
        adjusted_min_cash = min_cash_ratio * (1 + volatility)
        
        if cash_ratio < adjusted_min_cash:
            return False, f"현금비율 부족 ({cash_ratio:.1%} < {adjusted_min_cash:.1%})"
        
        if cash_ratio < trading_pct:
            return False, f"거래비율 초과 ({cash_ratio:.1%} < {trading_pct:.1%})"
        
        return True, "현금비율 적정"
    
    def _check_position_limit(self, asset: str, asset_type: str,
                            amount: float, total_value: float) -> Tuple[bool, str]:
        """포지션 한도 체크"""
        # 현재 포지션 조회
        current_positions = self._get_current_positions()
        
        # 예상 포지션 크기
        expected_position = amount / total_value
        
        # 단일 종목 한도
        if expected_position > self.risk_limits['max_single_position']:
            return False, f"단일종목 한도 초과 ({expected_position:.1%})"
        
        # 자산군별 한도
        asset_type_exposure = sum(
            p.weight for p in current_positions 
            if p.asset_type == asset_type
        )
        
        if asset_type_exposure + expected_position > 0.6:  # 60% 한도
            return False, f"{asset_type} 비중 초과"
        
        # 집중도 체크 (HHI)
        hhi = self._calculate_hhi(current_positions, asset, expected_position)
        if hhi > 0.3:  # HHI 0.3 초과시 과도한 집중
            return False, "포트폴리오 집중도 초과"
        
        return True, "포지션 한도 적정"
    
    def _check_correlation_risk(self, asset: str, asset_type: str) -> Tuple[bool, str]:
        """상관관계 리스크 체크"""
        current_positions = self._get_current_positions()
        
        if not current_positions:
            return True, "첫 포지션"
        
        # 상관계수 계산 (간단한 예시)
        high_corr_assets = {
            'BTC': ['ETH', 'XRP'],
            'ETH': ['BTC', 'ADA'],
            'AAPL': ['MSFT', 'GOOGL']
        }
        
        correlated = high_corr_assets.get(asset, [])
        
        for pos in current_positions:
            if pos.asset in correlated:
                if pos.weight > 0.1:  # 이미 10% 이상 보유
                    return False, f"{pos.asset}와 높은 상관관계"
        
        return True, "상관관계 적정"
    
    def _check_volatility_adjusted_size(self, asset: str, amount: float) -> Tuple[bool, str]:
        """변동성 조정 포지션 크기"""
        volatility = self._get_asset_volatility(asset)
        
        # 고변동성 자산은 포지션 축소
        if volatility > 0.5:  # 50% 이상 변동성
            max_amount = amount * 0.5  # 50% 축소
            return False, f"고변동성 자산 ({volatility:.1%})"
        
        return True, "변동성 수준 적정"
    
    def _check_market_conditions(self, asset_type: str) -> Tuple[bool, str]:
        """시장 상황 체크"""
        # VIX 지수나 시장 변동성 체크
        market_condition = self._analyze_market_condition()
        
        if market_condition == "extreme_volatility":
            return False, "극단적 시장 변동성"
        
        if market_condition == "crash" and asset_type != "cash":
            return False, "시장 폭락 상황"
        
        return True, "시장 상황 정상"
    
    async def _check_liquidity(self, asset: str, asset_type: str, 
                              amount: float) -> Tuple[bool, str]:
        """유동성 체크"""
        try:
            # 일일 거래량 대비 주문 크기
            daily_volume = await self._get_daily_volume(asset, asset_type)
            
            if daily_volume > 0:
                order_impact = amount / daily_volume
                
                if order_impact > 0.01:  # 일 거래량의 1% 초과
                    return False, "유동성 부족 (시장 충격)"
            
            return True, "유동성 충분"
            
        except Exception as e:
            logger.error(f"유동성 체크 실패: {e}")
            return True, "유동성 체크 스킵"
    
    def calculate_position_size(self, asset: str, asset_type: str,
                              confidence: int, total_value: float) -> float:
        """
        켈리 기준 기반 최적 포지션 크기 계산
        """
        base_size = self.cfg['trading'][asset_type]['percentage'] / 100
        
        # 켈리 공식 적용
        win_rate = self._get_historical_win_rate(asset)
        avg_win = self._get_average_win(asset)
        avg_loss = abs(self._get_average_loss(asset))
        
        if avg_loss > 0:
            kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_f = max(0, min(kelly_f, 0.25))  # 최대 25%로 제한
        else:
            kelly_f = base_size
        
        # 신뢰도 조정
        confidence_factor = confidence / 100
        
        # 변동성 조정
        volatility = self._get_asset_volatility(asset)
        vol_factor = 1 / (1 + volatility)
        
        # 최종 크기
        position_size = kelly_f * confidence_factor * vol_factor
        
        # 최소/최대 제한
        min_size = 0.01  # 1%
        max_size = base_size * 1.5
        
        return max(min_size, min(position_size, max_size)) * total_value
    
    async def monitor_positions(self) -> Dict[str, List[str]]:
        """
        실시간 포지션 모니터링
        
        Returns:
            {'stop_loss': [...], 'take_profit': [...], 'rebalance': [...]}
        """
        actions = {
            'stop_loss': [],
            'take_profit': [],
            'rebalance': [],
            'risk_alert': []
        }
        
        positions = self._get_current_positions()
        
        for pos in positions:
            # 손절/익절 체크
            if self._should_stop_loss(pos):
                actions['stop_loss'].append(pos.asset)
                await self._notify_stop_loss(pos)
            
            elif self._should_take_profit(pos):
                actions['take_profit'].append(pos.asset)
                await self._notify_take_profit(pos)
            
            # 리밸런싱 필요 체크
            if self._needs_rebalancing(pos):
                actions['rebalance'].append(pos.asset)
        
        # 포트폴리오 레벨 리스크
        portfolio_risks = await self._check_portfolio_risks(positions)
        if portfolio_risks:
            actions['risk_alert'] = portfolio_risks
        
        return actions
    
    def _should_stop_loss(self, position: Position) -> bool:
        """손절 조건 체크 (동적)"""
        base_threshold = self.cfg['trading'][position.asset_type].get('stop_loss', -0.05)
        
        # 보유 기간에 따른 조정
        holding_days = self._get_holding_days(position.asset)
        
        # 시간이 지날수록 손절선 완화
        time_factor = 1 + (holding_days / 30) * 0.2  # 30일마다 20% 완화
        adjusted_threshold = base_threshold * time_factor
        
        # 시장 상황 고려
        if self._analyze_market_condition() == "bear_market":
            adjusted_threshold *= 0.8  # 약세장에서는 빠른 손절
        
        return position.pnl_percent <= adjusted_threshold
    
    def _should_take_profit(self, position: Position) -> bool:
        """익절 조건 체크 (동적)"""
        base_threshold = self.cfg['trading'][position.asset_type].get('take_profit', 0.10)
        
        # 수익률에 따른 단계적 익절
        if position.pnl_percent >= base_threshold * 2:  # 20% 이상
            return True
        
        # 추세 약화시 익절
        if position.pnl_percent >= base_threshold:
            if self._is_trend_weakening(position.asset):
                return True
        
        # 포트폴리오 비중 과다시
        if position.weight > 0.4:  # 40% 초과
            if position.pnl_percent > 0:
                return True
        
        return False
    
    def _needs_rebalancing(self, position: Position) -> bool:
        """리밸런싱 필요 여부"""
        target_weight = 1 / self._get_portfolio_size()  # 균등 가중
        
        # 목표 대비 20% 이상 벗어나면 리밸런싱
        deviation = abs(position.weight - target_weight) / target_weight
        
        return deviation > 0.2
    
    async def _check_portfolio_risks(self, positions: List[Position]) -> List[str]:
        """포트폴리오 레벨 리스크 체크"""
        alerts = []
        
        # 1. VaR 계산
        var_95 = self._calculate_var(positions)
        if var_95 > self.risk_limits['max_portfolio_var']:
            alerts.append(f"VaR 한도 초과: {var_95:.1%}")
        
        # 2. 최대 낙폭
        max_dd = self._calculate_max_drawdown()
        if max_dd > self.risk_limits['max_drawdown']:
            alerts.append(f"최대낙폭 한도 초과: {max_dd:.1%}")
        
        # 3. 샤프비율
        sharpe = self._calculate_sharpe_ratio()
        if sharpe < self.risk_limits['min_sharpe_ratio']:
            alerts.append(f"샤프비율 미달: {sharpe:.2f}")
        
        # 4. 집중도
        concentration = self._calculate_concentration(positions)
        if concentration > 0.5:
            alerts.append(f"포트폴리오 집중도 높음: {concentration:.1%}")
        
        return alerts
    
    def _calculate_var(self, positions: List[Position], confidence: float = 0.95) -> float:
        """Value at Risk 계산"""
        if not positions:
            return 0
        
        # 포지션별 수익률
        returns = [p.pnl_percent for p in positions]
        
        if len(returns) < 2:
            return 0
        
        # 정규분포 가정하에 VaR
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for 95% confidence
        z_score = 1.645
        
        var = -(mean_return - z_score * std_return)
        
        return max(0, var)
    
    def _calculate_sharpe_ratio(self) -> float:
        """샤프비율 계산"""
        # 최근 30일 수익률
        trades = db_manager.get_recent_trades(days=30)
        
        if len(trades) < 5:
            return 0
        
        returns = [t.profit_rate for t in trades if t.profit_rate]
        
        if not returns:
            return 0
        
        # 무위험 수익률 (연 2%)
        risk_free_rate = 0.02 / 365
        
        excess_returns = [r - risk_free_rate for r in returns]
        
        if np.std(excess_returns) == 0:
            return 0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
        
        return sharpe
    
    def _calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        # 최근 90일 포트폴리오 가치
        values = db_manager.get_portfolio_values(days=90)
        
        if len(values) < 2:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_concentration(self, positions: List[Position]) -> float:
        """포트폴리오 집중도 (HHI)"""
        if not positions:
            return 0
        
        weights = [p.weight for p in positions]
        hhi = sum(w**2 for w in weights)
        
        return hhi
    
    def _calculate_hhi(self, positions: List[Position], 
                      new_asset: str, new_weight: float) -> float:
        """신규 포지션 포함 HHI 계산"""
        weights = {p.asset: p.weight for p in positions}
        
        if new_asset in weights:
            weights[new_asset] += new_weight
        else:
            weights[new_asset] = new_weight
        
        # 정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return sum(w**2 for w in weights.values())
    
    def _get_current_positions(self) -> List[Position]:
        """현재 포지션 조회"""
        # 캐시 확인
        if (datetime.now() - self.last_check).seconds < 60:
            return list(self.positions_cache.values())
        
        # DB에서 조회 (구현 필요)
        positions = []
        
        # 예시 데이터
        # positions = db_manager.get_active_positions()
        
        self.positions_cache = {p.asset: p for p in positions}
        self.last_check = datetime.now()
        
        return positions
    
    def _get_market_volatility(self) -> float:
        """시장 변동성 조회"""
        # VIX 지수나 최근 변동성 계산
        return 0.15  # 15% 예시
    
    def _get_asset_volatility(self, asset: str) -> float:
        """자산별 변동성"""
        # 최근 30일 일일 수익률 표준편차
        return 0.20  # 20% 예시
    
    def _analyze_market_condition(self) -> str:
        """시장 상황 분석"""
        # 실제 구현시 여러 지표 종합
        return "normal"  # normal, volatile, extreme_volatility, crash, bear_market
    
    def _get_historical_win_rate(self, asset: str) -> float:
        """과거 승률"""
        trades = db_manager.get_asset_performance(asset, days=90)
        return trades.get('win_rate', 50) / 100
    
    def _get_average_win(self, asset: str) -> float:
        """평균 수익률"""
        trades = db_manager.get_asset_performance(asset, days=90)
        return trades.get('avg_profit', 0.05)
    
    def _get_average_loss(self, asset: str) -> float:
        """평균 손실률"""
        trades = db_manager.get_asset_performance(asset, days=90)
        return trades.get('avg_loss', -0.03)
    
    async def _get_daily_volume(self, asset: str, asset_type: str) -> float:
        """일일 거래량 조회"""
        # 실제 API 호출
        return 1000000000  # 예시
    
    def _get_holding_days(self, asset: str) -> int:
        """보유 일수"""
        # DB에서 조회
        return 10  # 예시
    
    def _is_trend_weakening(self, asset: str) -> bool:
        """추세 약화 감지"""
        # RSI, MACD 등 기술적 지표 활용
        return False  # 예시
    
    def _get_portfolio_size(self) -> int:
        """포트폴리오 종목 수"""
        return len(self._get_current_positions()) or 1
    
    async def _notify_risk_rejection(self, asset: str, reason: str):
        """리스크 거부 알림"""
        await notifier.send_message(
            f"🚫 리스크 체크 실패\n"
            f"종목: {asset}\n"
            f"사유: {reason}"
        )
    
    async def _notify_stop_loss(self, position: Position):
        """손절 알림"""
        await notifier.send_message(
            f"🛑 손절 신호\n"
            f"종목: {position.asset}\n"
            f"손실: {position.pnl_percent:.1%}\n"
            f"현재가: {position.current_price:,.0f}"
        )
    
    async def _notify_take_profit(self, position: Position):
        """익절 알림"""
        await notifier.send_message(
            f"💰 익절 신호\n"
            f"종목: {position.asset}\n"
            f"수익: {position.pnl_percent:.1%}\n"
            f"현재가: {position.current_price:,.0f}"
        )
    
    async def generate_risk_report(self) -> str:
        """리스크 리포트 생성"""
        positions = self._get_current_positions()
        
        if not positions:
            return "포지션 없음"
        
        # 리스크 지표 계산
        metrics = RiskMetrics(
            total_exposure=sum(p.value for p in positions),
            max_position_size=max(p.weight for p in positions),
            correlation_risk=self._calculate_correlation_risk(positions),
            var_95=self._calculate_var(positions),
            expected_shortfall=self._calculate_expected_shortfall(positions),
            sharpe_ratio=self._calculate_sharpe_ratio(),
            max_drawdown=self._calculate_max_drawdown()
        )
        
        report = f"""
📊 리스크 리포트
================
총 노출: {metrics.total_exposure:,.0f}원
최대 포지션: {metrics.max_position_size:.1%}
상관관계 리스크: {metrics.correlation_risk:.2f}
VaR (95%): {metrics.var_95:.1%}
Expected Shortfall: {metrics.expected_shortfall:.1%}
샤프비율: {metrics.sharpe_ratio:.2f}
최대낙폭: {metrics.max_drawdown:.1%}

⚠️ 리스크 경고:
"""
        
        # 경고 사항 추가
        if metrics.var_95 > 0.05:
            report += "- VaR 한도 초과\n"
        if metrics.max_position_size > 0.3:
            report += "- 단일 종목 집중도 높음\n"
        if metrics.sharpe_ratio < 0.5:
            report += "- 위험 대비 수익률 낮음\n"
        
        return report
    
    def _calculate_correlation_risk(self, positions: List[Position]) -> float:
        """상관관계 리스크 계산"""
        # 실제로는 상관계수 매트릭스 활용
        return 0.5  # 예시
    
    def _calculate_expected_shortfall(self, positions: List[Position]) -> float:
        """Expected Shortfall (CVaR) 계산"""
        var_95 = self._calculate_var(positions)
        # 실제로는 VaR 이하 손실들의 평균
        return var_95 * 1.2  # 예시


# 전역 인스턴스
risk_manager = AdvancedRiskManager()

# 기존 인터페이스 호환
def check_asset_ratio(asset: str, asset_type: str, asset_value: float,
                     total_asset_value: float, cash_balance: float) -> bool:
    """기존 인터페이스 호환"""
    result, _ = asyncio.run(
        risk_manager.check_pre_trade_risk(
            asset, asset_type, 'buy', asset_value, 
            total_asset_value, cash_balance
        )
    )
    return result

def enforce_stop_loss(current_price: float, avg_price: float, threshold: float) -> bool:
    """기존 인터페이스 호환"""
    pnl = (current_price - avg_price) / avg_price
    return pnl <= threshold

def enforce_take_profit(current_price: float, avg_price: float, threshold: float) -> bool:
    """기존 인터페이스 호환"""
    pnl = (current_price - avg_price) / avg_price
    return pnl >= threshold


# 리스크 모니터링 스케줄러
async def scheduled_risk_monitoring():
    """정기 리스크 모니터링"""
    while True:
        try:
            # 포지션 모니터링
            actions = await risk_manager.monitor_positions()
            
            # 액션 실행
            if actions['stop_loss']:
                logger.warning(f"손절 필요: {actions['stop_loss']}")
            
            if actions['take_profit']:
                logger.info(f"익절 필요: {actions['take_profit']}")
            
            if actions['risk_alert']:
                await notifier.send_message(
                    "⚠️ 포트폴리오 리스크 경고\n" + 
                    "\n".join(actions['risk_alert'])
                )
            
            # 30분마다 실행
            await asyncio.sleep(1800)
            
        except Exception as e:
            logger.error(f"리스크 모니터링 오류: {e}")
            await asyncio.sleep(60)