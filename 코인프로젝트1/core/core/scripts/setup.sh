#!/usr/bin/env bash
# scripts/setup.sh
# 퀀트 트레이딩 시스템 설치 스크립트 - 퀸트프로젝트 수준

set -euo pipefail  # 에러시 즉시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 스피너 애니메이션
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        for i in `seq 0 9`; do
            printf " ${spinstr:$i:1}" 
            sleep $delay
            printf "\b\b"
        done
    done
    printf "  \b\b"
}

# OS 확인
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        log_error "지원하지 않는 OS: $OSTYPE"
        exit 1
    fi
    log_info "OS 감지: $OS"
}

# Python 버전 확인
check_python() {
    log_info "Python 버전 확인 중..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되어 있지 않습니다"
        log_info "Python 3.8 이상을 설치해주세요"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_VERSION="3.8"
    
    if [[ $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc) -eq 1 ]]; then
        log_error "Python $REQUIRED_VERSION 이상이 필요합니다 (현재: $PYTHON_VERSION)"
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION 확인"
}

# 가상환경 생성 및 활성화
setup_venv() {
    log_info "가상환경 설정 중..."
    
    # 기존 가상환경 제거 옵션
    if [[ -d ".venv" ]]; then
        log_warning "기존 가상환경이 존재합니다"
        read -p "제거하고 새로 생성하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf .venv
            log_info "기존 가상환경 제거 완료"
        else
            log_info "기존 가상환경 유지"
            return
        fi
    fi
    
    # 가상환경 생성
    python3 -m venv .venv &
    spinner $!
    
    # 가상환경 활성화
    if [[ "$OS" == "windows" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    
    log_success "가상환경 생성 및 활성화 완료"
}

# 의존성 설치
install_dependencies() {
    log_info "의존성 패키지 설치 중..."
    
    # pip 업그레이드
    pip install --upgrade pip &> /dev/null &
    spinner $!
    
    # requirements.txt 확인
    if [[ ! -f "requirements.txt" ]]; then
        log_error "requirements.txt 파일이 없습니다"
        create_requirements
    fi
    
    # 패키지 설치
    pip install -r requirements.txt --no-cache-dir &> install.log &
    spinner $!
    
    if [[ $? -eq 0 ]]; then
        log_success "모든 패키지 설치 완료"
        rm -f install.log
    else
        log_error "패키지 설치 실패. install.log 확인"
        exit 1
    fi
}

# requirements.txt 생성
create_requirements() {
    log_info "requirements.txt 생성 중..."
    
    cat > requirements.txt << EOF
# Core
python-dotenv==1.0.0
pyyaml==6.0.1

# Trading
pyupbit==0.2.33
pandas==2.0.3
numpy==1.24.3

# Notification
python-telegram-bot==20.7

# Database
sqlalchemy==2.0.23

# Analysis
matplotlib==3.7.2
seaborn==0.12.2

# Scheduling
apscheduler==3.10.4

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
EOF
    
    log_success "requirements.txt 생성 완료"
}

# 설정 파일 생성
create_config() {
    log_info "설정 파일 생성 중..."
    
    # config 디렉토리 생성
    mkdir -p config
    
    # config.yaml 템플릿
    if [[ ! -f "config/config.yaml" ]]; then
        cat > config/config.yaml << EOF
# 퀀트 트레이딩 시스템 설정
version: "1.0.0"
environment: "development"

# API 설정
api:
  access_key: "YOUR_ACCESS_KEY"
  secret_key: "YOUR_SECRET_KEY"

# 텔레그램 설정
telegram:
  token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
  retry_count: 3
  retry_delay: 2

# 트레이딩 설정
trading:
  stock:
    percentage: 30
    max_positions: 5
  coin:
    percentage: 20
    max_positions: 3

# 스케줄 설정
schedule:
  coin_collect: "*/30 * * * *"  # 30분마다
  stock_collect: "0 9-16 * * 1-5"  # 평일 9-16시
  
# 백테스트 설정
backtest:
  fee_rate: 0.0025
  slippage: 0.001
  initial_capital: 10000000

# 기타 설정
cache_ttl: 3600
max_workers: 5
batch_size: 10
EOF
        log_success "config.yaml 템플릿 생성 완료"
    else
        log_info "기존 config.yaml 유지"
    fi
    
    # .env 파일 생성
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# 환경 변수
DATABASE_URL=sqlite:///quant.db
LOG_LEVEL=INFO
DRY_RUN=false
EOF
        log_success ".env 파일 생성 완료"
    fi
}

# 디렉토리 구조 생성
create_directories() {
    log_info "디렉토리 구조 생성 중..."
    
    directories=(
        "logs"
        "data"
        "backtest_results"
        "scripts"
        "tests"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "디렉토리 구조 생성 완료"
}

# 데이터베이스 초기화
init_database() {
    log_info "데이터베이스 초기화 중..."
    
    python3 << 'PYCODE'
from db import Base, engine
Base.metadata.create_all(bind=engine)
print("✅ DB initialized")
PYCODE
}

# 시스템 테스트
run_tests() {
    log_info "시스템 테스트 실행 중..."
    
    # 간단한 import 테스트
    python3 << 'PYCODE'
try:
    import pyupbit
    import telegram
    import pandas
    import sqlalchemy
    from config import get_config
    from notifier import notifier
    from db import db_manager
    
    print("✅ 모든 모듈 import 성공")
    
    # 설정 테스트
    cfg = get_config()
    if cfg:
        print("✅ 설정 파일 로드 성공")
    
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    exit(1)
PYCODE
}

# 서비스 설정 (Linux)
setup_service() {
    if [[ "$OS" != "linux" ]]; then
        return
    fi
    
    log_info "시스템 서비스 설정..."
    
    read -p "systemd 서비스로 등록하시겠습니까? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo tee /etc/systemd/system/quant-trading.service > /dev/null << EOF
[Unit]
Description=Quant Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/.venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo systemctl daemon-reload
        log_success "서비스 등록 완료: sudo systemctl start quant-trading"
    fi
}

# 완료 메시지
print_completion() {
    echo
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✨ 퀀트 트레이딩 시스템 설치 완료! ✨${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "${YELLOW}다음 단계:${NC}"
    echo "1. config/config.yaml 파일에 API 키 입력"
    echo "2. 가상환경 활성화: source .venv/bin/activate"
    echo "3. 시스템 실행: python main.py"
    echo
    echo -e "${BLUE}추가 명령어:${NC}"
    echo "- 테스트 실행: pytest"
    echo "- 코드 포맷팅: black ."
    echo "- 백테스트: python -m backtest"
    echo
}

# 메인 실행
main() {
    echo -e "${BLUE}🚀 퀀트 트레이딩 시스템 설치 시작${NC}"
    echo
    
    check_os
    check_python
    setup_venv
    install_dependencies
    create_config
    create_directories
    init_database
    run_tests
    setup_service
    print_completion
}

# 스크립트 실행
main "$@"