#!/bin/bash

module load amdclang

cd ${REPO_DIR}/atomics_openmp

make arraysum2
export HSA_XNACK=0
./arraysum2

make clean

