#!/usr/bin/env bash
# Nsight Systems quick test for PyTorch (Git Bash / Linux 호환)
# - Windows: trace=cuda,nvtx,cublas
# - Linux  : trace=cuda,nvtx,cublas,osrt
# - 산출물(.nsys-rep / .qdrep) 자동 감지
# - 최신 nsys stats 리포트 이름 사용 + 실패 시 기본 리포트로 폴백
set -euo pipefail

OUTPUT_BASE="nsys_test"
INPUT_FILE=""

# 0) 사전 체크
if ! command -v nsys &>/dev/null; then
  echo "[ERROR] nsys not found in PATH. Add Nsight Systems '.../target-windows-x64' to PATH."
  exit 1
fi
if ! command -v python &>/dev/null; then
  echo "[ERROR] python not found in PATH."
  exit 1
fi
if [[ ! -f "test_nsys_pytorch.py" ]]; then
  echo "[ERROR] test_nsys_pytorch.py not found in current directory."
  exit 1
fi

# 1) 플랫폼별 trace 리스트
UNAME_S="$(uname -s || true)"
TRACE_LIST="cuda,nvtx,cublas"        # Windows 권장
if echo "$UNAME_S" | grep -Eqi 'linux'; then
  TRACE_LIST="cuda,nvtx,cublas,osrt" # Linux면 OS runtime 포함 가능
fi

echo "[INFO] nsys version:"
nsys --version || true
echo "[INFO] Using trace list: $TRACE_LIST"

# 2) 이전 산출물 정리
rm -f "${OUTPUT_BASE}.nsys-rep" "${OUTPUT_BASE}.qdrep" "${OUTPUT_BASE}.sqlite" "nsys_test_report.txt" || true

# 3) 프로파일 실행
echo "[INFO] Running Nsight Systems profile..."
# 관리자 권한 경고를 피하려고 CPU 샘플링/컨텍스트 스위치는 끕니다.
# CUDA event trace는 오버헤드/가짜 의존성 줄이기 위해 비활성화.
nsys profile \
  --output "${OUTPUT_BASE}" \
  --trace "${TRACE_LIST}" \
  --cuda-event-trace=false \
  --sample=none \
  python test_nsys_pytorch.py

# 4) 산출물 확장자 자동 감지 (.nsys-rep 우선, 없으면 .qdrep)
if [[ -f "${OUTPUT_BASE}.nsys-rep" ]]; then
  INPUT_FILE="${OUTPUT_BASE}.nsys-rep"
elif [[ -f "${OUTPUT_BASE}.qdrep" ]]; then
  INPUT_FILE="${OUTPUT_BASE}.qdrep"
else
  echo "[ERROR] profiling output not found (.nsys-rep/.qdrep)."
  exit 2
fi
echo "[INFO] Profiling output: ${INPUT_FILE}"

# 5) 리포트 생성 (신버전 리포트명 → 실패 시 기본 리포트 폴백)
echo "[INFO] Generating stats report..."
if nsys stats \
    --report cuda_api_sum \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_trace \
    --format column \
    "${INPUT_FILE}" > nsys_test_report.txt 2> _nsys_stats_err.log; then
  echo "[OK] Wrote nsys_test_report.txt (explicit reports)"
else
  echo "[WARN] Explicit reports failed. Falling back to default stats output."
  nsys stats "${INPUT_FILE}" > nsys_test_report.txt
  echo "[OK] Wrote nsys_test_report.txt (default)"
fi
rm -f _nsys_stats_err.log || true

# 6) 결과 안내
echo ""
echo "[DONE]"
echo " - ${INPUT_FILE}            (timeline)"
echo " - nsys_test_report.txt     (stats summary)"
echo ""
echo "Open in GUI:"
echo "  nsys-ui ${INPUT_FILE}"
