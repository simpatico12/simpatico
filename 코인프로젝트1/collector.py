# collector.py
"""
뉴스 수집 및 감성 분석 모듈 - 퀸트프로젝트 수준
병렬 처리와 캐시 최적화로 성능 극대화
"""
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
from collections import defaultdict

from utils import fetch_all_news, evaluate_news, is_holiday_or_weekend
from notifier import notifier
from logger import logger
from config import get_config


class NewsCollector:
    """뉴스 수집 및 감성 분석 클래스"""
    
    def __init__(self):
        self.cfg = get_config()
        self.cache: Dict[Tuple[str, str], Dict] = {}
        self.cache_ttl = self.cfg.get('cache_ttl', 3600)  # 1시간
        self.max_workers = self.cfg.get('max_workers', 5)
        self.batch_size = self.cfg.get('batch_size', 10)
        
    def _is_cache_valid(self, key: Tuple[str, str]) -> bool:
        """캐시 유효성 검사"""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key].get('timestamp')
        if not cached_time:
            return False
            
        return (datetime.now() - cached_time).seconds < self.cache_ttl
    
    def _get_cached_sentiment(self, asset_type: str, asset: str) -> Optional[str]:
        """캐시된 감성 데이터 조회"""
        key = (asset_type, asset)
        if self._is_cache_valid(key):
            return self.cache[key].get('sentiment')
        return None
    
    def _update_cache(self, asset_type: str, asset: str, sentiment: str, news_count: int = 0):
        """캐시 업데이트"""
        key = (asset_type, asset)
        self.cache[key] = {
            'sentiment': sentiment,
            'timestamp': datetime.now(),
            'news_count': news_count
        }
    
    async def collect_single(self, asset: str, asset_type: str) -> Dict:
        """단일 자산 뉴스 수집"""
        try:
            # 캐시 확인
            cached = self._get_cached_sentiment(asset_type, asset)
            if cached:
                logger.info(f"캐시 히트: {asset_type} {asset}")
                return {'asset': asset, 'sentiment': cached, 'cached': True}
            
            # 뉴스 수집 및 분석
            news = await asyncio.to_thread(fetch_all_news, asset)
            sentiment = await asyncio.to_thread(evaluate_news, news)
            
            # 캐시 업데이트
            self._update_cache(asset_type, asset, sentiment, len(news))
            
            return {
                'asset': asset,
                'sentiment': sentiment,
                'news_count': len(news),
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"{asset} 수집 실패: {e}")
            return {
                'asset': asset,
                'sentiment': 'ERROR',
                'error': str(e)
            }
    
    async def collect_batch(self, assets: List[str], asset_type: str) -> List[Dict]:
        """배치 단위 병렬 수집"""
        tasks = [self.collect_single(asset, asset_type) for asset in assets]
        return await asyncio.gather(*tasks)
    
    async def collect(self, asset_list: List[str], asset_type: str) -> None:
        """메인 수집 함수"""
        # 휴장일 체크
        if is_holiday_or_weekend() and asset_type != 'coin':
            await notifier.send_message(f'⏸️ <b>{asset_type.upper()}</b> 시장 휴장일')
            return
        
        start_time = datetime.now()
        total_assets = len(asset_list)
        
        # 시작 알림
        await notifier.send_message(
            f"📊 <b>{asset_type.upper()} 뉴스 수집 시작</b>\n"
            f"대상: {total_assets}개 종목"
        )
        
        # 배치 처리
        all_results = []
        for i in range(0, total_assets, self.batch_size):
            batch = asset_list[i:i + self.batch_size]
            results = await self.collect_batch(batch, asset_type)
            all_results.extend(results)
            
            # 진행상황 알림 (25% 단위)
            progress = (i + len(batch)) / total_assets * 100
            if progress % 25 == 0 and progress < 100:
                logger.info(f"{asset_type} 수집 진행률: {progress:.0f}%")
        
        # 결과 집계
        summary = self._summarize_results(all_results, asset_type)
        
        # 완료 알림
        elapsed = (datetime.now() - start_time).seconds
        await notifier.send_message(self._format_summary(summary, elapsed))
        
        # 주요 감성 변화 알림
        await self._notify_significant_changes(all_results, asset_type)
    
    def _summarize_results(self, results: List[Dict], asset_type: str) -> Dict:
        """결과 요약"""
        sentiment_counts = defaultdict(int)
        errors = []
        cached_count = 0
        
        for result in results:
            if 'error' in result:
                errors.append(result['asset'])
            elif result.get('cached'):
                cached_count += 1
                sentiment_counts[result['sentiment']] += 1
            else:
                sentiment_counts[result['sentiment']] += 1
        
        return {
            'asset_type': asset_type,
            'total': len(results),
            'sentiments': dict(sentiment_counts),
            'errors': errors,
            'cached': cached_count
        }
    
    def _format_summary(self, summary: Dict, elapsed: int) -> str:
        """요약 메시지 포맷팅"""
        sentiments = summary['sentiments']
        
        return f"""
✅ <b>{summary['asset_type'].upper()} 뉴스 수집 완료</b>
━━━━━━━━━━━━━━━
📈 긍정: {sentiments.get('positive', 0)}개
➖ 중립: {sentiments.get('neutral', 0)}개  
📉 부정: {sentiments.get('negative', 0)}개
💾 캐시: {summary['cached']}개
⚠️ 오류: {len(summary['errors'])}개
⏱️ 소요: {elapsed}초
━━━━━━━━━━━━━━━"""
    
    async def _notify_significant_changes(self, results: List[Dict], asset_type: str):
        """주요 감성 변화 개별 알림"""
        # 부정적 감성 종목만 별도 알림
        negative_assets = [
            r for r in results 
            if r.get('sentiment') == 'negative' and not r.get('cached')
        ]
        
        if negative_assets:
            message = f"🚨 <b>부정적 뉴스 감지</b>\n"
            for asset in negative_assets[:5]:  # 상위 5개만
                message += f"• {asset['asset']}\n"
            
            if len(negative_assets) > 5:
                message += f"외 {len(negative_assets) - 5}개..."
            
            await notifier.send_message(message)
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 조회"""
        valid_count = sum(1 for k in self.cache if self._is_cache_valid(k))
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_count,
            'hit_rate': valid_count / len(self.cache) * 100 if self.cache else 0
        }
    
    def clear_expired_cache(self):
        """만료된 캐시 정리"""
        expired_keys = [k for k in self.cache if not self._is_cache_valid(k)]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"캐시 정리: {len(expired_keys)}개 항목 삭제")


# 싱글톤 인스턴스
collector = NewsCollector()

# 기존 인터페이스 호환
NEWS_CACHE = collector.cache

def collect(asset_list: List[str], asset_type: str) -> None:
    """기존 동기 함수 인터페이스"""
    asyncio.run(collector.collect(asset_list, asset_type))


# 스케줄러용 함수
async def scheduled_collect(asset_configs: Dict[str, List[str]]):
    """여러 자산군 순차 수집"""
    for asset_type, assets in asset_configs.items():
        await collector.collect(assets, asset_type)
        await asyncio.sleep(60)  # 자산군 간 1분 대기
    
    # 수집 완료 후 캐시 정리
    collector.clear_expired_cache()