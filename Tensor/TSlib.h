#pragma once
#include <iostream>
#include <array>
#include <tuple>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>
#include <functional>
#include <type_traits>
#include <algorithm>
#include <math.h>
#include <cstdarg>
#include <numeric>
#include <filesystem>
#include <fstream>

#include "TensorEnums.h"
#include "TSliceWrapper.h"
#include "TensorCompareOperators.h"

#include "Tensor.h"
#ifdef _TS_CUDA
#include "TensorCuda.cuh"
#endif
#include "TensorArithmetic.h"
#include "TensorExceptions.h"
#ifdef _TS_CUDA
#include "TensorOperatorKernels.cuh"
#endif

#include "TensorTools.h"


#include "Tensor.ipp"
#ifdef _TS_CUDA
#include "TensorCuda.ipp"
#endif
#include "TensorArithmeticOperators.ipp"
#include "TensorMath.ipp"
#include "TensorSlice.ipp"
#include "TensorTools.ipp"

#ifdef _TS_OPENCV_MODULE
#include "TensorOpenCV.h"
#include "TensorOpenCV.ipp"
#endif

#ifdef _TS_NOISE_MODULE
#include "TensorPerlin.h"
#include "TensorPerlin.ipp"
#endif
