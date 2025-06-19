#!/usr/bin/env bash
# scripts/backup.sh
# 데이터 백업 스크립트

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# DB 백업
cp quant.db "$BACKUP_DIR/"

# 로그 백업
cp -r logs/ "$BACKUP_DIR/"

# 설정 백업
cp config/config.yaml "$BACKUP_DIR/"

# 압축
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "✅ 백업 완료: $BACKUP_DIR.tar.gz"

---

#!/usr/bin/env bash
# scripts/update.sh
# 시스템 업데이트 스크립트

# Git pull
git pull origin main

# 가상환경 활성화
source .venv/bin/activate

# 패키지 업데이트
pip install -r requirements.txt --upgrade

# DB 마이그레이션 (필요시)
python -c "from db import Base, engine; Base.metadata.create_all(bind=engine)"

echo "✅ 업데이트 완료"

---

#!/usr/bin/env bash
# scripts/health_check.sh
# 시스템 상태 체크

source .venv/bin/activate

python3 << 'EOF'
import asyncio
from notifier import notifier
from db import db_manager
import pyupbit

async def check():
    print("🔍 시스템 체크 중...")
    
    # DB 체크
    try:
        summary = db_manager.get_daily_summary()
        print("✅ DB 정상")
    except:
        print("❌ DB 오류")
    
    # 텔레그램 체크
    try:
        await notifier.send_message("🏥 Health Check")
        print("✅ 텔레그램 정상")
    except:
        print("❌ 텔레그램 오류")
    
    # API 체크
    try:
        btc = pyupbit.get_current_price("KRW-BTC")
        print(f"✅ API 정상 (BTC: {btc:,.0f}원)")
    except:
        print("❌ API 오류")

asyncio.run(check())
EOF

---

#!/usr/bin/env bash
# scripts/logs.sh
# 로그 확인 도구

# 최근 에러 로그
echo "📋 최근 에러 로그:"
grep -i error logs/*.log | tail -20

# 오늘 거래 로그
echo -e "\n📊 오늘 거래:"
grep -i "execute_trade" logs/$(date +%Y%m%d)*.log

# 로그 크기 확인
echo -e "\n💾 로그 크기:"
du -sh logs/*