//
// Copyright (c) 2023-2024 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef BASIC_TEST_CONVERSIONS_H
#define BASIC_TEST_CONVERSIONS_H

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "harness/errorHelpers.h"
#include "harness/rounding_mode.h"

#include <stdio.h>
#if defined( __APPLE__ )
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include <CL/cl_half.h>

#include "harness/mt19937.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#define kVectorSizeCount 6
#define kMaxVectorSize 16
#define kPageSize 4096

#define BUFFER_SIZE (1024 * 1024 * 8)
#define EMBEDDED_REDUCTION_FACTOR 16

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
/* Rounding modes and saturation for use with qcom 64 bit to float conversion
 * library */
#define CONVERSIONS_QCOM 1
#else
#define CONVERSIONS_QCOM 0
#endif

typedef enum
{
    kUnsaturated = 0,
    kSaturated,

    kSaturationModeCount
} SaturationMode;

extern const char *gTypeNames[ kTypeCount ];
extern const char *gRoundingModeNames[ kRoundingModeCount ];        // { "", "_rte", "_rtp", "_rtn", "_rtz" }
extern const char *gSaturationNames[ kSaturationModeCount ];        // { "", "_sat" }
extern const char *gVectorSizeNames[kVectorSizeCount];              // { "", "2", "4", "8", "16" }
extern const size_t gTypeSizes[kTypeCount];

//Functions for clamping floating point numbers into the representable range for the type
typedef float (*clampf)( float );
typedef double (*clampd)( double );

extern clampf gClampFloat[ kTypeCount ][kRoundingModeCount];
extern clampd gClampDouble[ kTypeCount ][kRoundingModeCount];

typedef void (*InitDataFunc)( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
extern InitDataFunc gInitFunctions[ kTypeCount ];

typedef int (*CheckResults)( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
extern CheckResults gCheckResults[ kTypeCount ];

#define kCallStyleCount (kVectorSizeCount + 1 /* for implicit scalar */)

extern cl_context gContext;
extern int gHasDouble;
extern int gTestDouble;
extern int gHasHalfs;
extern int gTestHalfs;
extern int gWimpyReductionFactor;
extern int gSkipTesting;
extern int gMinVectorSize;
extern int gMaxVectorSize;
extern int gForceFTZ;
extern int gIsRTZ;
extern int gForceHalfFTZ;
extern int gIsHalfRTZ;
extern cl_half_rounding_mode gDefaultHalfRoundingMode;

struct buffers
{
    void *in;
    void *ref;
    void *allowZ;
    void *out[kCallStyleCount];
    cl_mem inBuffer;
    cl_mem outBuffers[kCallStyleCount];
    cl_command_queue queue;
};
extern std::vector<struct buffers> buffers_vec;

extern bool gTestAll;
extern std::recursive_mutex gLock;

extern const char *sizeNames[];
extern const int vectorSizes[];

struct ConversionsTest;

namespace conv_test {

cl_program MakeProgram(Type outType, Type inType, SaturationMode sat,
                       RoundingMode round, int vectorSize,
                       cl_kernel *outKernel);

cl_int InitData(cl_uint job_id, cl_uint thread_id, void *p);
cl_int PrepareReference(cl_uint job_id, cl_uint thread_id, void *p);
}

struct CalcRefValsBase
{
    CalcRefValsBase(struct buffers &buffers_): buffers(buffers_) {};
    virtual ~CalcRefValsBase() = default;
    virtual int check_result(void *, uint32_t, int) { return 0; }

    clKernelWrapper kernel; // the kernel for this vector size
    clProgramWrapper program; // the program for this vector size
    struct buffers &buffers;
    cl_event event;
};

template <typename InType, typename OutType, bool InFP, bool OutFP>
struct CalcRefValsPat : CalcRefValsBase
{
    CalcRefValsPat(struct buffers &buffers): CalcRefValsBase(buffers) {}
    int check_result(void *, uint32_t, int) override;
};

int RunTest(cl_device_id device, cl_context context, cl_command_queue queue,
            int num_elements, void *arg);

struct ConversionsTest
{
    template <typename InType, typename OutType, bool InFP, bool OutFP>
    test_status DoTest(struct buffers &buffers);

    Type outType;
    Type inType;
    SaturationMode sat;
    RoundingMode round;
    std::string name;
};

#endif /* BASIC_TEST_CONVERSIONS_H */

