# 모델 비교 및 사용 가이드

## 사용 가능한 모델

### 1. Isolation Forest (권장)

**장점:**
- 매우 빠른 학습 (1초 이내)
- 빠른 예측
- TensorFlow/PyTorch 불필요
- 안정적이고 Mac에서 문제 없음
- 해석 가능

**단점:**
- 복잡한 패턴 학습 능력 제한적

**사용 방법:**
```bash
python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log --output models/isolation_forest
```

**대시보드에서:**
- 모델 경로: `models/isolation_forest`

---

### 2. PyTorch AutoEncoder (Mac 권장)

**장점:**
- 빠른 학습 (약 1초)
- Mac에서 안정적
- 복잡한 패턴 학습 가능
- TensorFlow 문제 없음

**단점:**
- PyTorch 설치 필요

**사용 방법:**
```bash
python scripts/train_pytorch_autoencoder.py --data data/raw_logs/nginx_access.log --output models/pytorch_autoencoder --epochs 20
```

**대시보드에서:**
- 모델 경로: `models/pytorch_autoencoder`

---

### 3. TensorFlow AutoEncoder (Mac에서 문제 있음)

**장점:**
- 복잡한 패턴 학습 가능

**단점:**
- Mac에서 mutex lock 문제
- 학습이 멈춤
- Python 3.13 호환성 문제

**현재 상태:** Mac에서 사용 불가

---

## 성능 비교

| 모델 | 학습 시간 | 예측 속도 | Mac 호환성 | 정확도 |
|------|----------|----------|-----------|--------|
| Isolation Forest | 1초 | 매우 빠름 | 완벽 | 좋음 |
| PyTorch AutoEncoder | 1초 | 빠름 | 완벽 | 매우 좋음 |
| TensorFlow AutoEncoder | 멈춤 | - | 문제 | - |

## 권장 사항

### Mac 사용자
1. **Isolation Forest** (가장 빠르고 안정적)
2. **PyTorch AutoEncoder** (더 정확한 결과 원할 때)

### Linux/Windows 사용자
1. **TensorFlow AutoEncoder** (가능하면)
2. **PyTorch AutoEncoder** (대안)
3. **Isolation Forest** (빠른 테스트용)

## 현재 학습된 모델

```bash
ls -lh models/
```

- `isolation_forest`: Isolation Forest 모델 (161KB)
- `pytorch_autoencoder_pytorch.pth`: PyTorch AutoEncoder 모델
- `pytorch_autoencoder_pytorch_metadata.pkl`: PyTorch 모델 메타데이터

## 빠른 시작

```bash
# 1. Isolation Forest (가장 빠름)
python scripts/train_isolation_forest.py

# 2. PyTorch AutoEncoder (더 정확)
python scripts/train_pytorch_autoencoder.py --epochs 20
```
