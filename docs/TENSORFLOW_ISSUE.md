# TensorFlow 학습 문제 해결 가이드

## 문제 상황

AutoEncoder 모델 학습 시 TensorFlow가 mutex lock 오류로 멈추는 문제가 발생합니다.

## 원인

- macOS에서 TensorFlow의 멀티스레딩 문제
- Python 3.13과 TensorFlow 호환성 문제
- 데이터가 적을 때 TensorFlow 초기화가 느림

## 해결 방안

### 방안 1: Isolation Forest 사용 (권장)

**장점:**
- 즉시 작동 (1초 이내 학습)
- TensorFlow 불필요
- 안정적이고 빠름
- 이미 학습 완료됨

**사용 방법:**
```bash
# 이미 학습 완료됨
# 대시보드에서 models/isolation_forest 사용
```

### 방안 2: TensorFlow 재설치

```bash
# 가상환경에서
pip uninstall tensorflow tensorflow-macos
pip install tensorflow-macos tensorflow-metal  # Apple Silicon용
```

### 방안 3: 더 가벼운 모델 사용

현재 프로젝트에는 다음 모델들이 있습니다:
- **Isolation Forest**: 작동 중 (빠르고 안정적)
- **AutoEncoder**: TensorFlow 문제로 학습 어려움
- **LSTM AutoEncoder**: 미구현

## 현재 상태

- Isolation Forest 모델: 학습 완료 (`models/isolation_forest`)
- 데이터: 5000개 로그 생성 완료
- AutoEncoder: TensorFlow 문제로 학습 불가

## 권장 사항

**프로덕션 환경에서는 Isolation Forest를 사용하는 것을 권장합니다:**
- 빠른 학습 및 예측
- 안정적
- 해석 가능
- TensorFlow 의존성 없음

AutoEncoder는 더 복잡한 패턴을 학습할 수 있지만, 현재 환경에서는 학습이 어렵습니다.
