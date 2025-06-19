"""
Core Package Initialization
===========================

이 패키지는 프로젝트의 핵심 기능들을 초기화하고 관리합니다.
초보자도 쉽게 이해하고 유지보수할 수 있도록 구성되었습니다.

Author: Your Name
Created: 2025-06-18
Version: 1.0.0
"""

# =============================================================================
# 패키지 정보
# =============================================================================
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "핵심 패키지 - 프로젝트의 기본 기능 제공"

# =============================================================================
# 로깅 설정 (디버깅과 오류 추적을 위해)
# =============================================================================
import logging
import sys
from pathlib import Path

# 로그 디렉토리 생성
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 핸들러 설정 (콘솔과 파일 동시 출력)
if not logger.handlers:
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        LOG_DIR / "core.log", 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.info("Core 패키지가 초기화되었습니다.")

# =============================================================================
# 환경 설정 관리
# =============================================================================
import os
from typing import Dict, Any, Optional

class Config:
    """
    환경 설정을 관리하는 클래스
    
    사용법:
        config = Config()
        debug_mode = config.get('DEBUG', False)
    """
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._load_default_config()
        self._load_env_config()
    
    def _load_default_config(self):
        """기본 설정 로드"""
        self._config.update({
            'DEBUG': False,
            'LOG_LEVEL': 'INFO',
            'MAX_WORKERS': os.cpu_count() or 4,
            'TIMEOUT': 30,
            'ENCODING': 'utf-8',
        })
    
    def _load_env_config(self):
        """환경 변수에서 설정 로드"""
        env_mapping = {
            'DEBUG': ('DEBUG', bool),
            'LOG_LEVEL': ('LOG_LEVEL', str),
            'MAX_WORKERS': ('MAX_WORKERS', int),
            'TIMEOUT': ('TIMEOUT', int),
        }
        
        for key, (env_key, type_func) in env_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    if type_func == bool:
                        self._config[key] = env_value.lower() in ('true', '1', 'yes')
                    else:
                        self._config[key] = type_func(env_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"환경 변수 {env_key} 변환 실패: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """설정 값 설정하기"""
        self._config[key] = value
        logger.debug(f"설정 변경: {key} = {value}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """설정 딕셔너리로 일괄 업데이트"""
        self._config.update(config_dict)
        logger.debug(f"설정 일괄 업데이트: {config_dict}")

# 전역 설정 인스턴스
config = Config()

# =============================================================================
# 유틸리티 함수들
# =============================================================================

def get_project_root() -> Path:
    """
    프로젝트의 루트 디렉토리 경로를 반환합니다.
    
    Returns:
        Path: 프로젝트 루트 디렉토리 경로
    """
    current_file = Path(__file__)
    # core/__init__.py에서 두 단계 위로 올라가면 프로젝트 루트
    return current_file.parent.parent

def ensure_directory(path: Path) -> Path:
    """
    디렉토리가 존재하지 않으면 생성합니다.
    
    Args:
        path: 생성할 디렉토리 경로
    
    Returns:
        Path: 생성된 디렉토리 경로
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    안전하게 모듈을 import합니다. 실패시 None을 반환합니다.
    
    Args:
        module_name: import할 모듈명
        package: 패키지명 (상대 import시 사용)
    
    Returns:
        모듈 객체 또는 None
    """
    try:
        if package:
            from importlib import import_module
            return import_module(module_name, package)
        else:
            return __import__(module_name)
    except ImportError as e:
        logger.warning(f"모듈 {module_name} import 실패: {e}")
        return None

# =============================================================================
# 예외 처리 클래스들
# =============================================================================

class CoreError(Exception):
    """Core 패키지의 기본 예외 클래스"""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        logger.error(f"CoreError 발생: {message} (코드: {error_code})")

class ConfigError(CoreError):
    """설정 관련 오류"""
    pass

class ValidationError(CoreError):
    """데이터 검증 오류"""
    pass

# =============================================================================
# 패키지 내 모듈들의 공통 인터페이스
# =============================================================================

class BaseComponent:
    """
    Core 패키지 내 모든 컴포넌트의 기본 클래스
    
    모든 핵심 컴포넌트는 이 클래스를 상속받아야 합니다.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.config = config
        self._initialized = False
    
    def initialize(self) -> None:
        """컴포넌트 초기화"""
        if self._initialized:
            self.logger.warning(f"{self.name}이 이미 초기화되었습니다.")
            return
        
        self._do_initialize()
        self._initialized = True
        self.logger.info(f"{self.name} 컴포넌트가 초기화되었습니다.")
    
    def _do_initialize(self) -> None:
        """실제 초기화 작업 (하위 클래스에서 오버라이드)"""
        pass
    
    def cleanup(self) -> None:
        """리소스 정리"""
        self.logger.info(f"{self.name} 컴포넌트 정리 중...")
        self._do_cleanup()
        self._initialized = False
    
    def _do_cleanup(self) -> None:
        """실제 정리 작업 (하위 클래스에서 오버라이드)"""
        pass
    
    def is_initialized(self) -> bool:
        """초기화 상태 확인"""
        return self._initialized

# =============================================================================
# 패키지 초기화 및 정리 함수들
# =============================================================================

_components: Dict[str, BaseComponent] = {}

def register_component(component: BaseComponent) -> None:
    """
    컴포넌트를 등록합니다.
    
    Args:
        component: 등록할 컴포넌트
    """
    _components[component.name] = component
    logger.debug(f"컴포넌트 등록: {component.name}")

def get_component(name: str) -> Optional[BaseComponent]:
    """
    등록된 컴포넌트를 가져옵니다.
    
    Args:
        name: 컴포넌트 이름
    
    Returns:
        컴포넌트 객체 또는 None
    """
    return _components.get(name)

def initialize_all_components() -> None:
    """모든 등록된 컴포넌트를 초기화합니다."""
    logger.info("모든 컴포넌트 초기화 시작...")
    
    for name, component in _components.items():
        try:
            component.initialize()
        except Exception as e:
            logger.error(f"컴포넌트 {name} 초기화 실패: {e}")
            raise CoreError(f"컴포넌트 초기화 실패: {name}", "INIT_FAILED")
    
    logger.info("모든 컴포넌트 초기화 완료")

def cleanup_all_components() -> None:
    """모든 등록된 컴포넌트를 정리합니다."""
    logger.info("모든 컴포넌트 정리 시작...")
    
    for name, component in _components.items():
        try:
            component.cleanup()
        except Exception as e:
            logger.error(f"컴포넌트 {name} 정리 실패: {e}")
    
    logger.info("모든 컴포넌트 정리 완료")

# =============================================================================
# 종료 시 자동 정리 설정
# =============================================================================
import atexit

def _cleanup_on_exit():
    """프로그램 종료 시 자동으로 정리 작업 수행"""
    logger.info("프로그램 종료 - 정리 작업 수행")
    cleanup_all_components()

atexit.register(_cleanup_on_exit)

# =============================================================================
# 패키지 공개 API
# =============================================================================

# 다른 모듈에서 import할 수 있는 주요 객체들
__all__ = [
    # 버전 정보
    '__version__',
    '__author__',
    '__description__',
    
    # 핵심 클래스들
    'Config',
    'BaseComponent',
    'CoreError',
    'ConfigError',
    'ValidationError',
    
    # 전역 객체들
    'config',
    'logger',
    
    # 유틸리티 함수들
    'get_project_root',
    'ensure_directory',
    'safe_import',
    
    # 컴포넌트 관리 함수들
    'register_component',
    'get_component',
    'initialize_all_components',
    'cleanup_all_components',
]

# =============================================================================
# 패키지 초기화 완료 로그
# =============================================================================
logger.info(f"Core 패키지 초기화 완료 (버전: {__version__})")
logger.debug(f"설정: {config._config}")
logger.debug(f"프로젝트 루트: {get_project_root()}")

# 개발 모드에서 추가 정보 출력
if config.get('DEBUG'):
    logger.debug("=== DEBUG 모드 활성화 ===")
    logger.debug(f"Python 버전: {sys.version}")
    logger.debug(f"현재 작업 디렉토리: {os.getcwd()}")
    logger.debug(f"로그 디렉토리: {LOG_DIR}")