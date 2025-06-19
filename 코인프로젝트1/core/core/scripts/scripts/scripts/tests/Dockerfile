################################
# EC2용 Docker 설정 (수정사항 포함)
################################

################################
# 1. Dockerfile (EC2 최적화)
################################
FROM python:3.11-slim as builder

ARG PYTHON_VERSION=3.11
ARG TIMEZONE=Asia/Seoul

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# EC2는 apt 미러 서버가 느릴 수 있으므로 한국 미러 사용
RUN sed -i 's/deb.debian.org/ftp.kr.debian.org/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

FROM python:3.11-slim

LABEL maintainer="your-email@example.com" \
      version="1.0.0" \
      description="Quant Trading System for EC2"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/quant/.local/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    TZ=Asia/Seoul

RUN groupadd -r quant && \
    useradd -r -g quant -d /home/quant -s /sbin/nologin -c "Quant user" quant && \
    mkdir -p /home/quant && \
    chown -R quant:quant /home/quant

RUN sed -i 's/deb.debian.org/ftp.kr.debian.org/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

WORKDIR /app

COPY --from=builder --chown=quant:quant /root/.local /home/quant/.local
COPY --chown=quant:quant . .

RUN mkdir -p logs data backtest_results && \
    chown -R quant:quant /app && \
    chmod -R 755 /app && \
    chmod +x scripts/docker-entrypoint.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python scripts/health_check.py || exit 1

USER quant
EXPOSE 8501

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]

################################
# 2. docker-compose.yml (EC2 최적화)
################################
version: '3.8'

services:
  quant-trading:
    build:
      context: .
      dockerfile: Dockerfile
    image: quant-trading:latest
    container_name: quant-trading-system
    restart: unless-stopped
    
    environment:
      - TRADING_ENV=production
      - DATABASE_URL=sqlite:///data/quant.db
      - LOG_LEVEL=INFO
    
    volumes:
      # EC2 EBS 볼륨 마운트 가능
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    
    # EC2 인스턴스 리소스에 맞게 조정
    deploy:
      resources:
        limits:
          cpus: '1'  # t2.micro는 1 vCPU
          memory: 900M  # t2.micro는 1GB RAM
        reservations:
          cpus: '0.5'
          memory: 512M
    
    # 로깅 드라이버 설정 (CloudWatch 연동 가능)
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    image: quant-trading:latest
    container_name: quant-dashboard
    restart: unless-stopped
    command: ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    
    ports:
      - "8501:8501"
    
    environment:
      - DATABASE_URL=sqlite:///data/quant.db
    
    volumes:
      - ./data:/app/data:ro
    
    depends_on:
      - quant-trading
    
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 400M

################################
# 3. scripts/ec2-setup.sh (EC2 초기 설정)
################################
#!/bin/bash
# EC2 인스턴스 초기 설정 스크립트

set -e

echo "🚀 EC2 퀀트 트레이딩 시스템 설정 시작..."

# 1. 시스템 업데이트
echo "📦 시스템 패키지 업데이트..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Docker 설치
echo "🐳 Docker 설치..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# 3. Docker Compose 설치
echo "🐳 Docker Compose 설치..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# 4. 스왑 메모리 설정 (t2.micro용)
echo "💾 스왑 메모리 설정..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# 5. 방화벽 설정
echo "🔒 방화벽 설정..."
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 8501/tcp  # Streamlit
sudo ufw --force enable

# 6. 시간대 설정
echo "🕐 시간대 설정..."
sudo timedatectl set-timezone Asia/Seoul

# 7. 로그 로테이션 설정
echo "📝 로그 로테이션 설정..."
sudo tee /etc/logrotate.d/quant-trading > /dev/null << EOF
/home/ubuntu/quant-trading/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
EOF

echo "✅ EC2 설정 완료! 재로그인 필요 (docker 그룹 적용)"

################################
# 4. scripts/ec2-deploy.sh (배포 스크립트)
################################
#!/bin/bash
# EC2 배포 자동화 스크립트

set -e

# 설정
REMOTE_USER="ubuntu"
REMOTE_HOST="your-ec2-public-ip"
KEY_PATH="~/.ssh/your-key.pem"
PROJECT_DIR="/home/ubuntu/quant-trading"

echo "🚀 EC2 배포 시작..."

# 1. 코드 동기화
echo "📦 코드 업로드..."
rsync -avz -e "ssh -i $KEY_PATH" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='logs/*' \
    --exclude='data/*' \
    ./ $REMOTE_USER@$REMOTE_HOST:$PROJECT_DIR/

# 2. 원격 명령 실행
echo "🔧 원격 배포 실행..."
ssh -i $KEY_PATH $REMOTE_USER@$REMOTE_HOST << 'ENDSSH'
cd /home/ubuntu/quant-trading

# 기존 컨테이너 중지
docker-compose down

# 이미지 빌드
docker-compose build

# 컨테이너 시작
docker-compose up -d

# 헬스체크
sleep 10
docker-compose ps
docker-compose logs --tail=50 quant-trading

echo "✅ 배포 완료!"
ENDSSH

################################
# 5. systemd 서비스 파일 (선택사항)
################################
# /etc/systemd/system/quant-trading.service
[Unit]
Description=Quant Trading Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/quant-trading
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target

################################
# 6. CloudWatch 로깅 설정 (선택사항)
################################
# docker-compose.yml에 추가
services:
  quant-trading:
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/quant-trading
        awslogs-region: ap-northeast-2
        awslogs-stream-prefix: quant

################################
# 7. EC2 User Data (인스턴스 생성시)
################################
#!/bin/bash
# EC2 인스턴스 시작시 자동 실행

# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Docker Compose 설치
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Git 설치
apt-get update
apt-get install -y git

# 프로젝트 클론
su - ubuntu -c "git clone https://github.com/your-repo/quant-trading.git /home/ubuntu/quant-trading"

# 시작
su - ubuntu -c "cd /home/ubuntu/quant-trading && docker-compose up -d"