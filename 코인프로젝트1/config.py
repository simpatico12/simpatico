"""
설정 관리 모듈
- 환경변수와 YAML 통합
- 설정값 검증
- 타입 안전성
- 자동 리로드
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
import hashlib

from logger import logger, error, info, warning

# .env 파일 로드
load_dotenv()

class ConfigError(Exception):
    """설정 관련 에러"""
    pass

class ConfigValidator:
    """설정값 검증 클래스"""
    
    @staticmethod
    def validate_api_keys(config: dict) -> None:
        """API 키 검증"""
        required_keys = {
            'upbit': ['access_key', 'secret_key'],
            'openai': ['key'],
        }
        
        for service, keys in required_keys.items():
            if service not in config.get('api', {}):
                raise ConfigError(f"{service} API 설정이 없습니다")
                
            for key in keys:
                value = config['api'][service].get(key)
                if not value or value.startswith('your_'):
                    raise ConfigError(f"{service} {key}가 설정되지 않았습니다")
    
    @staticmethod
    def validate_telegram(config: dict) -> None:
        """텔레그램 설정 검증"""
        telegram = config.get('telegram', {})
        if not telegram.get('token') or not telegram.get('chat_id'):
            warning("텔레그램 설정이 없습니다. 알림이 비활성화됩니다.")
    
    @staticmethod
    def validate_trading_params(config: dict) -> None:
        """거래 파라미터 검증"""
        for market in config.get('assets', {}).keys():
            trading = config.get('strategy', {}).get(market, {})
            
            # 퍼센트 값 검증
            percentage = trading.get('trading_percentage', 0)
            if not 0 < percentage <= 100:
                raise ConfigError(f"{market} trading_percentage가 잘못되었습니다: {percentage}")
            
            # 손절/익절 검증
            stop_loss = trading.get('stop_loss', 0)
            take_profit = trading.get('take_profit', 0)
            
            if stop_loss >= 0:
                raise ConfigError(f"{market} stop_loss는 음수여야 합니다: {stop_loss}")
            if take_profit <= 0:
                raise ConfigError(f"{market} take_profit은 양수여야 합니다: {take_profit}")

class Config:
    """싱글톤 설정 관리자"""
    
    _instance = None
    _config = None
    _config_hash = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.config_file = Path('config.yaml')
        self.env_file = Path('.env')
        self.backup_dir = Path('config_backups')
        self.backup_dir.mkdir(exist_ok=True)
        
        if self._config is None:
            self.reload()
    
    def reload(self) -> Dict[str, Any]:
        """설정 리로드"""
        try:
            info("설정 파일 로딩 중...")
            
            # 기존 설정 백업
            if self._config:
                self._backup_config()
            
            # YAML 파일 로드
            if not self.config_file.exists():
                raise ConfigError("config.yaml 파일이 없습니다")
                
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 환경변수 통합
            config = self._merge_env_vars(config)
            
            # 동적 설정 추가
            config = self._add_dynamic_configs(config)
            
            # 설정 검증
            self._validate_config(config)
            
            # 해시 업데이트 (변경 감지용)
            self._config_hash = self._calculate_hash(config)
            
            self._config = config
            info("설정 로딩 완료")
            
            return config
            
        except Exception as e:
            error(f"설정 로딩 실패: {e}")
            if self._config:
                warning("이전 설정을 사용합니다")
                return self._config
            raise ConfigError(f"설정 로딩 실패: {e}")
    
    def _merge_env_vars(self, config: dict) -> dict:
        """환경변수 통합"""
        # API 설정
        config['api'] = config.get('api', {})
        
        # Upbit
        config['api']['upbit'] = {
            'access_key': os.getenv('UPBIT_ACCESS_KEY'),
            'secret_key': os.getenv('UPBIT_SECRET_KEY')
        }
        
        # IBKR Japan
        if os.getenv('IBKR_JAPAN_CLIENT_ID'):
            config['api']['ibkr_japan'] = {
                'client_id': os.getenv('IBKR_JAPAN_CLIENT_ID'),
                'secret': os.getenv('IBKR_JAPAN_SECRET')
            }
        
        # IBKR US
        if os.getenv('IBKR_US_CLIENT_ID'):
            config['api']['ibkr_us'] = {
                'client_id': os.getenv('IBKR_US_CLIENT_ID'),
                'secret': os.getenv('IBKR_US_SECRET')
            }
        
        # OpenAI
        config['api']['openai'] = {
            'key': os.getenv('OPENAI_API_KEY')
        }
        
        # Telegram
        config['telegram'] = {
            'token': os.getenv('TELEGRAM_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
        # 자산별 거래 파라미터
        for market in config.get('assets', {}).keys():
            if 'strategy' not in config:
                config['strategy'] = {}
            if market not in config['strategy']:
                config['strategy'][market] = {}
                
            strategy = config['strategy'][market]
            
            # 환경변수에서 값 가져오기
            env_vars = {
                'trading_percentage': f'TRADING_PERCENTAGE_{market.upper()}',
                'min_cash_ratio': f'MIN_CASH_RATIO_{market.upper()}',
                'stop_loss': f'STOP_LOSS_{market.upper()}',
                'take_profit': f'TAKE_PROFIT_{market.upper()}'
            }
            
            for key, env_var in env_vars.items():
                value = os.getenv(env_var)
                if value is not None:
                    try:
                        strategy[key] = float(value)
                    except ValueError:
                        warning(f"{env_var} 값이 올바르지 않습니다: {value}")
        
        # 시스템 설정
        config['system'] = config.get('system', {})
        config['system']['dry_run'] = os.getenv('DRY_RUN', 'false').lower() == 'true'
        config['system']['environment'] = os.getenv('ENVIRONMENT', 'development')
        
        # 리스크 관리
        config['risk_management'] = config.get('risk_management', {})
        if os.getenv('DAILY_LOSS_LIMIT'):
            config['risk_management']['daily_loss_limit'] = float(os.getenv('DAILY_LOSS_LIMIT'))
        if os.getenv('EMERGENCY_STOP_ENABLED'):
            config['risk_management']['emergency_stop'] = os.getenv('EMERGENCY_STOP_ENABLED').lower() == 'true'
        
        return config
    
    def _add_dynamic_configs(self, config: dict) -> dict:
        """동적 설정 추가"""
        # 런타임 정보
        config['runtime'] = {
            'start_time': datetime.now().isoformat(),
            'config_version': self._calculate_hash(config)[:8],
            'pid': os.getpid()
        }
        
        # 경로 설정
        config['paths'] = {
            'logs': 'logs',
            'data': 'data',
            'backtest': 'backtest_results',
            'database': config.get('database', {}).get('path', './data/trading.db')
        }
        
        # 기본값 설정
        defaults = {
            'max_retry_count': int(os.getenv('MAX_RETRY_COUNT', '3')),
            'retry_delay': 1,
            'request_timeout': 30,
            'rate_limit_delay': 0.1
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _validate_config(self, config: dict) -> None:
        """설정 검증"""
        validator = ConfigValidator()
        
        # 필수 섹션 확인
        required_sections = ['api', 'assets', 'schedule', 'strategy']
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"필수 섹션이 없습니다: {section}")
        
        # API 키 검증
        validator.validate_api_keys(config)
        
        # 텔레그램 검증
        validator.validate_telegram(config)
        
        # 거래 파라미터 검증
        validator.validate_trading_params(config)
        
        # 자산 검증
        for market, assets in config.get('assets', {}).items():
            if not assets:
                warning(f"{market} 자산 목록이 비어있습니다")
    
    def _calculate_hash(self, config: dict) -> str:
        """설정 해시 계산"""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _backup_config(self) -> None:
        """설정 백업"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f'config_backup_{timestamp}.yaml'
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
                
            # 오래된 백업 삭제 (최근 10개만 유지)
            backups = sorted(self.backup_dir.glob('config_backup_*.yaml'))
            for old_backup in backups[:-10]:
                old_backup.unlink()
                
        except Exception as e:
            warning(f"설정 백업 실패: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 가져오기 (점 표기법 지원)"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
    
    def get_market_config(self, market: str) -> Dict[str, Any]:
        """특정 시장 설정 가져오기"""
        return {
            'assets': self.get(f'assets.{market}', []),
            'strategy': self.get(f'strategy.{market}', {}),
            'schedule': self.get(f'schedule.{market}', [])
        }
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.get('system.environment') == 'production'
    
    def is_dry_run(self) -> bool:
        """모의거래 모드 여부"""
        return self.get('system.dry_run', False)
    
    def has_changed(self) -> bool:
        """설정 파일 변경 여부 확인"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                current_hash = self._calculate_hash(yaml.safe_load(f))
            return current_hash != self._config_hash
        except:
            return False
    
    def save_runtime_state(self, state: dict) -> None:
        """런타임 상태 저장"""
        state_file = Path('runtime_state.json')
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            error(f"런타임 상태 저장 실패: {e}")
    
    def __getitem__(self, key: str) -> Any:
        """딕셔너리 스타일 접근"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """in 연산자 지원"""
        return key in self._config

# 싱글톤 인스턴스
config = Config()

# 편의 함수
def get_config() -> Dict[str, Any]:
    """전체 설정 가져오기"""
    return config._config

def reload_config() -> Dict[str, Any]:
    """설정 리로드"""
    return config.reload()

def get_api_config(service: str) -> Dict[str, str]:
    """특정 API 설정 가져오기"""
    return config.get(f'api.{service}', {})

def get_telegram_config() -> Dict[str, str]:
    """텔레그램 설정 가져오기"""
    return config.get('telegram', {})

def is_market_open(market: str) -> bool:
    """시장 오픈 여부 확인"""
    from datetime import datetime
    import pytz
    
    # 간단한 시장 시간 체크 (추후 확장 가능)
    now = datetime.now(pytz.UTC)
    
    if market == 'coin':
        return True  # 24/7
    elif market == 'us':
        # 미국 시장: 9:30 AM - 4:00 PM EST
        et = pytz.timezone('US/Eastern')
        et_now = now.astimezone(et)
        return 9 <= et_now.hour < 16 and et_now.weekday() < 5
    elif market == 'japan':
        # 일본 시장: 9:00 AM - 3:00 PM JST
        jst = pytz.timezone('Asia/Tokyo')
        jst_now = now.astimezone(jst)
        return 9 <= jst_now.hour < 15 and jst_now.weekday() < 5
    
    return False

# 테스트 코드
if __name__ == '__main__':
    import pprint
    
    try:
        # 설정 로드
        cfg = get_config()
        
        print("=== 전체 설정 ===")
        pprint.pprint(cfg)
        
        print("\n=== API 설정 ===")
        print(f"Upbit: {get_api_config('upbit')}")
        print(f"OpenAI: {get_api_config('openai')}")
        
        print("\n=== 시장별 설정 ===")
        for market in ['coin', 'japan', 'us']:
            print(f"\n{market}:")
            pprint.pprint(config.get_market_config(market))
        
        print("\n=== 시스템 정보 ===")
        print(f"환경: {config.get('system.environment')}")
        print(f"모의거래: {config.is_dry_run()}")
        print(f"프로덕션: {config.is_production()}")
        
        print("\n=== 시장 오픈 상태 ===")
        for market in ['coin', 'japan', 'us']:
            print(f"{market}: {'오픈' if is_market_open(market) else '마감'}")
            
    except ConfigError as e:
        print(f"설정 에러: {e}")
    except Exception as e:
        print(f"예상치 못한 에러: {e}")
        import traceback
        traceback.print_exc()