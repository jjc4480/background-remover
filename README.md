# AWS Lambda 배경 제거 서비스 (BiRefNet)

AWS Lambda와 BiRefNet 모델을 사용한 고품질 배경 제거 서비스입니다. S3에 업로드된 이미지를 자동으로 처리하여 최첨단 AI 기술로 배경을 제거합니다.

## 🚀 주요 기능

- **고품질 배경 제거**: BiRefNet 모델 사용으로 특히 머리카락과 복잡한 엣지 처리 우수
- **자동 처리**: S3 이미지 업로드 시 자동 트리거
- **다양한 포맷 지원**: JPG, JPEG, PNG, WebP 지원
- **확장 가능한 아키텍처**: Docker 컨테이너 기반 서버리스 Lambda 함수
- **엣지 개선**: 자연스러운 결과를 위한 고급 후처리
- **스마트 경로 관리**: 원본 유지, 처리된 버전은 `_bg_removed.png` 접미사 추가
- **모듈화된 구조**: Lambda와 로컬 실행 환경을 위한 공통 배경 제거 로직

## 📋 사전 요구사항

- 적절한 권한을 가진 AWS 계정
- AWS CLI 설정 완료
- Docker 설치
- Python 3.11+

## 🏗️ 아키텍처

```
S3 버킷 → S3 이벤트 트리거 → Lambda 함수 (BiRefNet) → 처리된 이미지 S3 저장
```

## 📁 프로젝트 구조

```
lambda-background-remover/
├── src/
│   ├── lambda_function.py      # Lambda 핸들러
│   ├── background_remover.py   # 공통 배경 제거 로직
│   └── local_runner.py         # 로컬 실행 스크립트
├── scripts/
│   ├── deploy.sh               # 배포 스크립트
│   └── create-lambda.sh        # Lambda 생성 스크립트
├── config/
│   ├── lambda_config.json      # Lambda 설정
│   └── s3-trigger-config.json  # S3 트리거 설정
├── Dockerfile                   # Docker 컨테이너 정의
├── requirements.txt             # Python 의존성
├── .env.example                 # 환경 변수 템플릿
└── README.md                    # 이 파일
```

## 🔧 설치

1. **저장소 클론**
```bash
git clone <repository-url>
cd lambda-background-remover
```

2. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일을 편집하여 AWS 자격 증명 및 설정 입력
```

3. **의존성 설치 (로컬 테스트용)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **로컬 테스트 실행**
```bash
# 단일 이미지 처리
python src/local_runner.py test_image.jpg

# 디렉토리 내 모든 이미지 처리
python src/local_runner.py image_folder/
```

## 🚀 배포

### 빠른 배포

```bash
# 스크립트 실행 권한 부여
chmod +x scripts/*.sh

# ECR 배포 및 Lambda 업데이트
./scripts/deploy.sh
```

배포 스크립트는 자동으로:
- ECR 리포지토리 생성 (없는 경우)
- BiRefNet 모델을 포함한 Docker 이미지 빌드
- ECR 인증
- 이미지 태그 및 푸시

### Lambda 함수 생성 (최초 1회)

```bash
./scripts/create-lambda.sh
```

## ⚙️ 설정

### Lambda 설정
- **메모리**: 10240 MB (10 GB) - BiRefNet 모델 필수
- **타임아웃**: 900초 (15분)
- **임시 스토리지**: 10 GB
- **모델**: BiRefNet-general (973MB)
- **아키텍처**: x86_64

### S3 설정
- **경로 패턴**: `static/competition/applicant/[entry_no]/`
- **출력 형식**: `{원본파일명}_bg_removed.png`
- **지원 포맷**: jpg, jpeg, png, webp

## 🧪 테스트

### 로컬 테스트
```bash
# 테스트 이미지 준비
mkdir bg_remove_test_images
# 테스트 이미지를 bg_remove_test_images/ 폴더에 복사

# 단일 이미지 테스트
python src/local_runner.py bg_remove_test_images/test.jpg

# 디렉토리 내 모든 이미지 테스트
python src/local_runner.py bg_remove_test_images/

# 다른 모델 사용
python src/local_runner.py test.jpg output.png u2net
```

### Lambda 배포 테스트
```bash
# S3에 테스트 이미지 업로드
aws s3 cp test_image.jpg s3://your-bucket/static/competition/applicant/12345/test.jpg

# CloudWatch 로그 확인
aws logs tail /aws/lambda/background-remover --follow
```

## 💰 비용 예상

### BiRefNet 모델 (현재 설정)
- **요청당 비용**: 약 ₩6.65 ($0.005)
- **월 1,000회**: 약 ₩6,650 ($5.00)
- **월 10,000회**: 약 ₩66,500 ($50.00)

*참고: BiRefNet은 u2net보다 약 20배 비싸지만, 특히 머리카락과 복잡한 엣지 처리에서 훨씬 우수한 품질을 제공합니다.*

## 📊 성능

- **처리 시간**: 이미지당 약 30초
- **메모리 사용량**: 약 9.8 GB
- **모델 크기**: 973 MB (BiRefNet)
- **품질**: ⭐⭐⭐⭐⭐ (최고 수준)

## 🔍 모델 비교

| 모델 | 품질 | 속도 | 메모리 | 비용 |
|------|------|------|--------|------|
| u2net | ⭐⭐⭐ | 5초 | 3GB | ₩0.33/요청 |
| BiRefNet | ⭐⭐⭐⭐⭐ | 30초 | 10GB | ₩6.65/요청 |

## 🛠️ 문제 해결

### Lambda 메모리 부족
메모리 오류 발생 시 Lambda를 10GB로 설정:
```bash
aws lambda update-function-configuration \
  --function-name background-remover \
  --memory-size 10240
```

### 모델 다운로드 문제
BiRefNet 모델(973MB)은 첫 콜드 스타트 시 다운로드됩니다. 확인 사항:
- 임시 스토리지를 10GB로 설정
- Lambda 타임아웃을 최소 900초로 설정

### 처리되지 않는 이미지
- CloudWatch 로그 확인
- IAM 역할 권한 확인 (S3 GetObject, PutObject)
- S3 트리거 설정 확인

## 📝 환경 변수

```bash
# 필수
AWS_ACCOUNT_ID=your_account_id
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-bucket-name

# Lambda 환경 변수
NUMBA_CACHE_DIR=/tmp
NUMBA_DISABLE_JIT=1
U2NET_HOME=/tmp
HOME=/tmp
```

## 🤝 기여하기

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경 사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 제공됩니다.

## 🙏 감사의 말

- [rembg](https://github.com/danielgatis/rembg) - 핵심 배경 제거 라이브러리
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - 최첨단 배경 제거 모델
- AWS Lambda - 서버리스 인프라

## 📞 지원

문제, 질문, 제안 사항이 있으시면 GitHub 저장소에서 Issue를 열어주세요.

---

**참고**: 이것은 고품질 배경 제거를 위한 프로덕션 준비 구현입니다. 비용 최적화를 위해 일반 요청에는 u2net을, 프리미엄 요청에는 BiRefNet을 사용하는 계층화된 시스템을 구현하는 것을 고려하세요.