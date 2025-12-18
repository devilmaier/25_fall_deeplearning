#!/bin/bash

# run_st_loader.sh

set -e

# (옵션) 가상환경 쓰면 여기서 활성화
# source venv/bin/activate

# BLAS / MKL 스레드 수 제한해서 CPU 폭주 방지
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 파이썬 버퍼링 끄고 실행 (로그 바로바로 보이게)
python -u train_st_model.py
