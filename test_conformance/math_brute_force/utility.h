//
// Copyright (c) 2017-2024 The Khronos Group Inc.
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
#ifndef UTILITY_H
#define UTILITY_H

#include "harness/compat.h"
#include "harness/rounding_mode.h"
#include "harness/fpcontrol.h"
#include "harness/mathHelpers.h"
#include "harness/testHarness.h"
#include "harness/ThreadPool.h"
#include "harness/conversions.h"
#include "harness/parseParameters.h"
#include "CL/cl_half.h"

#define BUFFER_SIZE (1024 * 1024 * 2)
#define EMBEDDED_REDUCTION_FACTOR (64)

#if defined(__GNUC__)
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

struct Func;

extern int gWimpyReductionFactor;

#define VECTOR_SIZE_COUNT 6
extern const char *sizeNames[VECTOR_SIZE_COUNT];
extern const int sizeValues[VECTOR_SIZE_COUNT];

#include <mutex>
#include <queue>
#include <cstdlib>
#include <alloc.h>

struct MathResources
{
    void *in = nullptr;
    void *in2 = nullptr;
    void *in3 = nullptr;
    void *out_ref = nullptr;
    void *out[VECTOR_SIZE_COUNT] = {};
    void *out_ref2 = nullptr;
    void *out2[VECTOR_SIZE_COUNT] = {};
    cl_mem inBuffer = nullptr;
    cl_mem inBuffer2 = nullptr;
    cl_mem inBuffer3 = nullptr;
    cl_mem outBuffer[VECTOR_SIZE_COUNT] = {};
    cl_mem outBuffer2[VECTOR_SIZE_COUNT] = {};
    cl_command_queue queue = nullptr;
    MTdataHolder d;
};

extern thread_local MathResources *gThreadResources;
extern MathResources *gActiveThreadPoolResources;

#define gQueue                                                                 \
    (gThreadResources ? gThreadResources->queue                                \
                      : gActiveThreadPoolResources->queue)
#define gIn                                                                    \
    (gThreadResources ? gThreadResources->in : gActiveThreadPoolResources->in)
#define gIn2                                                                   \
    (gThreadResources ? gThreadResources->in2 : gActiveThreadPoolResources->in2)
#define gIn3                                                                   \
    (gThreadResources ? gThreadResources->in3 : gActiveThreadPoolResources->in3)
#define gOut_Ref                                                               \
    (gThreadResources ? gThreadResources->out_ref                              \
                      : gActiveThreadPoolResources->out_ref)
#define gOut_Ref2                                                              \
    (gThreadResources ? gThreadResources->out_ref2                             \
                      : gActiveThreadPoolResources->out_ref2)
#define gOut                                                                   \
    (gThreadResources ? gThreadResources->out : gActiveThreadPoolResources->out)
#define gOut2                                                                  \
    (gThreadResources ? gThreadResources->out2                                 \
                      : gActiveThreadPoolResources->out2)
#define gInBuffer                                                              \
    (gThreadResources ? gThreadResources->inBuffer                             \
                      : gActiveThreadPoolResources->inBuffer)
#define gInBuffer2                                                             \
    (gThreadResources ? gThreadResources->inBuffer2                            \
                      : gActiveThreadPoolResources->inBuffer2)
#define gInBuffer3                                                             \
    (gThreadResources ? gThreadResources->inBuffer3                            \
                      : gActiveThreadPoolResources->inBuffer3)
#define gOutBuffer                                                             \
    (gThreadResources ? gThreadResources->outBuffer                            \
                      : gActiveThreadPoolResources->outBuffer)
#define gOutBuffer2                                                            \
    (gThreadResources ? gThreadResources->outBuffer2                           \
                      : gActiveThreadPoolResources->outBuffer2)

#define gMTdata                                                                \
    (gThreadResources ? gThreadResources->d : gActiveThreadPoolResources->d)

cl_int MathThreadPool_Do(TPFuncPtr func_ptr, cl_uint count, void *userInfo,
                         MathResources *res);
#define ThreadPool_Do(func, count, arg)                                        \
    MathThreadPool_Do(func, count, arg, gThreadResources)

class MathResourcesPool {
public:
    MathResources Acquire(cl_context context, cl_device_id device)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        MathResources res;
        if (pool_.empty())
        {
            cl_uint min_alignment = 0;
            clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                            sizeof(cl_uint), (void *)&min_alignment, NULL);
            min_alignment >>= 3; // convert bits to bytes

            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type),
                            &device_type, NULL);

            res.in = align_malloc(BUFFER_SIZE, min_alignment);
            res.in2 = align_malloc(BUFFER_SIZE, min_alignment);
            res.in3 = align_malloc(BUFFER_SIZE, min_alignment);
            res.out_ref = align_malloc(BUFFER_SIZE, min_alignment);
            res.out_ref2 = align_malloc(BUFFER_SIZE, min_alignment);
            for (int j = 0; j < VECTOR_SIZE_COUNT; ++j)
            {
                res.out[j] = align_malloc(BUFFER_SIZE, min_alignment);
                res.out2[j] = align_malloc(BUFFER_SIZE, min_alignment);
            }

            cl_int error;
            cl_mem_flags device_flags = CL_MEM_READ_ONLY;
            if (device_type == CL_DEVICE_TYPE_CPU)
                device_flags |= CL_MEM_USE_HOST_PTR;
            else
                device_flags |= CL_MEM_COPY_HOST_PTR;

            res.inBuffer = clCreateBuffer(context, device_flags, BUFFER_SIZE,
                                          res.in, &error);
            res.inBuffer2 = clCreateBuffer(context, device_flags, BUFFER_SIZE,
                                           res.in2, &error);
            res.inBuffer3 = clCreateBuffer(context, device_flags, BUFFER_SIZE,
                                           res.in3, &error);

            device_flags = CL_MEM_READ_WRITE;
            if (device_type == CL_DEVICE_TYPE_CPU)
                device_flags |= CL_MEM_USE_HOST_PTR;
            else
                device_flags |= CL_MEM_COPY_HOST_PTR;

            for (int j = 0; j < VECTOR_SIZE_COUNT; ++j)
            {
                res.outBuffer[j] = clCreateBuffer(
                    context, device_flags, BUFFER_SIZE, res.out[j], &error);
                res.outBuffer2[j] = clCreateBuffer(
                    context, device_flags, BUFFER_SIZE, res.out2[j], &error);
            }

            res.queue = clCreateCommandQueue(context, device, 0, &error);
        }
        else
        {
            res = std::move(pool_.front());
            pool_.pop();
        }

        res.d = MTdataHolder(gRandomSeed);
        return res;
    }

    void Release(MathResources &res)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push(std::move(res));
    }

    void ReleaseAll()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pool_.empty())
        {
            MathResources res = std::move(pool_.front());
            pool_.pop();
            clReleaseMemObject(res.inBuffer);
            clReleaseMemObject(res.inBuffer2);
            clReleaseMemObject(res.inBuffer3);
            for (int j = 0; j < VECTOR_SIZE_COUNT; ++j)
            {
                clReleaseMemObject(res.outBuffer[j]);
                clReleaseMemObject(res.outBuffer2[j]);
            }
            clReleaseCommandQueue(res.queue);
            align_free(res.in);
            align_free(res.in2);
            align_free(res.in3);
            align_free(res.out_ref);
            align_free(res.out_ref2);
            for (int j = 0; j < VECTOR_SIZE_COUNT; ++j)
            {
                align_free(res.out[j]);
                align_free(res.out2[j]);
            }
        }
    }

private:
    std::mutex mutex_;
    std::queue<MathResources> pool_;
};

extern cl_device_id gDevice;
extern cl_context gContext;
extern int gSkipCorrectnessTesting;
extern int gForceFTZ;
extern int gFastRelaxedDerived;
extern int gHostFill;
extern int gIsInRTZMode;
extern int gHasHalf;
extern int gHasDouble;
extern int gTestFloat;
extern int gInfNanSupport;
extern int gIsEmbedded;
extern int gVerboseBruteForce;
extern uint32_t gMaxVectorSizeIndex;
extern uint32_t gMinVectorSizeIndex;
extern cl_device_fp_config gFloatCapabilities;
extern cl_device_fp_config gHalfCapabilities;
extern cl_device_fp_config gDoubleCapabilities;
extern RoundingMode gFloatToHalfRoundingMode;

extern cl_half_rounding_mode gHalfRoundingMode;

extern bool gTestAll;

#define HFF(num) cl_half_from_float(num, gHalfRoundingMode)
#define HFD(num) cl_half_from_double(num, gHalfRoundingMode)
#define HTF(num) cl_half_to_float(num)

#define LOWER_IS_BETTER 0
#define HIGHER_IS_BETTER 1

#include "harness/errorHelpers.h"

#if defined(_MSC_VER)
// Deal with missing scalbn on windows
#define scalbnf(_a, _i) ldexpf(_a, _i)
#define scalbn(_a, _i) ldexp(_a, _i)
#define scalbnl(_a, _i) ldexpl(_a, _i)
#endif

float Abs_Error(float test, double reference);
float Ulp_Error(float test, double reference);
float Bruteforce_Ulp_Error_Double(double test, long double reference);

extern MathResourcesPool gResourcesPool;
struct ResourceGuard
{
    MathResources res;
    ResourceGuard()
    {
        res = gResourcesPool.Acquire(gContext, gDevice);
        gThreadResources = &res;
    }
    ~ResourceGuard()
    {
        gResourcesPool.Release(res);
        gThreadResources = nullptr;
    }
};

// The spec is fairly clear that we may enforce a hard cutoff to prevent
// premature flushing to zero.
// However, to avoid conflict for 1.0, we are letting results at TYPE_MIN +
// ulp_limit to be flushed to zero.
inline int IsFloatResultSubnormal(double x, float ulps)
{
    x = fabs(x) - MAKE_HEX_DOUBLE(0x1.0p-149, 0x1, -149) * (double)ulps;
    return x < MAKE_HEX_DOUBLE(0x1.0p-126, 0x1, -126);
}

inline int IsHalfResultSubnormal(float x, float ulps)
{
    x = fabs(x) - MAKE_HEX_FLOAT(0x1.0p-24, 0x1, -24) * ulps;
    return x < MAKE_HEX_FLOAT(0x1.0p-14, 0x1, -14);
}

inline int IsFloatResultSubnormalAbsError(double x, float abs_err)
{
    x = x - abs_err;
    return x < MAKE_HEX_DOUBLE(0x1.0p-126, 0x1, -126);
}

inline int IsDoubleResultSubnormal(long double x, float ulps)
{
    x = fabsl(x) - MAKE_HEX_LONG(0x1.0p-1074, 0x1, -1074) * (long double)ulps;
    return x < MAKE_HEX_LONG(0x1.0p-1022, 0x1, -1022);
}

inline int IsFloatInfinity(double x)
{
    union {
        cl_float d;
        cl_uint u;
    } u;
    u.d = (cl_float)x;
    return ((u.u & 0x7fffffffU) == 0x7F800000U);
}

inline int IsFloatMaxFloat(double x)
{
    union {
        cl_float d;
        cl_uint u;
    } u;
    u.d = (cl_float)x;
    return ((u.u & 0x7fffffffU) == 0x7F7FFFFFU);
}

inline int IsFloatNaN(double x)
{
    union {
        cl_float d;
        cl_uint u;
    } u;
    u.d = (cl_float)x;
    return ((u.u & 0x7fffffffU) > 0x7F800000U);
}

inline bool IsHalfInfinity(const cl_half v)
{
    // Extract FP16 exponent and mantissa
    uint16_t h_exp = (((cl_half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = ((cl_half)v) & 0x3FF;

    // Inf test
    return (h_exp == 0x1F && h_mant == 0);
}

cl_uint RoundUpToNextPowerOfTwo(cl_uint x);

// Windows (since long double got deprecated) sets the x87 to 53-bit precision
// (that's x87 default state).  This causes problems with the tests that
// convert long and ulong to float and double or otherwise deal with values
// that need more precision than 53-bit. So, set the x87 to 64-bit precision.
inline void Force64BitFPUPrecision(void)
{
#if __MINGW32__
    // The usual method is to use _controlfp as follows:
    //     #include <float.h>
    //     _controlfp(_PC_64, _MCW_PC);
    //
    // _controlfp is available on MinGW32 but not on MinGW64. Instead of having
    // divergent code just use inline assembly which works for both.
    unsigned short int orig_cw = 0;
    unsigned short int new_cw = 0;
    __asm__ __volatile__("fstcw %0" : "=m"(orig_cw));
    new_cw = orig_cw | 0x0300; // set precision to 64-bit
    __asm__ __volatile__("fldcw  %0" ::"m"(new_cw));
#elif defined(_WIN32)                                                          \
    && (defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER))
    // Unfortunately, usual method (`_controlfp( _PC_64, _MCW_PC );') does *not*
    // work on win.x64: > On the x64 architecture, changing the floating point
    // precision is not supported. (Taken from
    // http://msdn.microsoft.com/en-us/library/e9b52ceh%28v=vs.100%29.aspx)
    int cw;
    __asm { fnstcw cw }
    ; // Get current value of FPU control word.
    cw = cw & 0xfffffcff
        | (3 << 8); // Set Precision Control to Double Extended Precision.
    __asm { fldcw cw }
    ; // Set new value of FPU control word.
#else
    /* Implement for other platforms if needed */
#endif
}

void memset_pattern4(void *dest, const void *src_pattern, size_t bytes);

union int32f_t {
    int32_t i;
    float f;
};

union int64d_t {
    int64_t l;
    double d;
};

void MulD(double *rhi, double *rlo, double u, double v);
void AddD(double *rhi, double *rlo, double a, double b);
void MulDD(double *rhi, double *rlo, double xh, double xl, double yh,
           double yl);
void AddDD(double *rhi, double *rlo, double xh, double xl, double yh,
           double yl);
void DivideDD(double *chi, double *clo, double a, double b);
int compareFloats(float x, float y);
int compareDoubles(double x, double y);

void logFunctionInfo(const char *fname, unsigned int float_size,
                     unsigned int isFastRelaxed);

float getAllowedUlpError(const Func *f, Type t, const bool relaxed);

inline uint64_t getTestStep(size_t typeSize, size_t bufferSize)
{
    return bufferSize / typeSize;
}

#endif /* UTILITY_H */
