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
#include "harness/mathHelpers.h"
#include "harness/testHarness.h"
#include "harness/compat.h"
#include "harness/ThreadPool.h"
#include "harness/parseParameters.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/sysctl.h>
#endif
#if defined(__linux__)
#include <sys/param.h>
#include <libgen.h>
#endif

#if defined(__MINGW32__)
#include <sys/param.h>
#endif

#include <sstream>
#include <stdarg.h>
#if !defined(_WIN32)
#include <libgen.h>
#include <sys/mman.h>
#endif
#include <time.h>

#include <algorithm>

#include <vector>
#include <type_traits>
#include <cmath>
#include <mutex>

#include "basic_test_conversions.h"
#include "conversions_data_info.h"

#if defined(_M_IX86) || defined(_M_X64)
#include <mmintrin.h>
#include <emmintrin.h>
#else
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#endif

const char *gTypeNames[kTypeCount] = { "uchar",  "char",  "ushort", "short",
                                       "uint",   "int",   "half",   "float",
                                       "double", "ulong", "long" };

const char *gRoundingModeNames[kRoundingModeCount] = { "", "_rte", "_rtp",
                                                       "_rtn", "_rtz" };

const char *gSaturationNames[2] = { "", "_sat" };

const size_t gTypeSizes[kTypeCount] = {
    sizeof(cl_uchar),  sizeof(cl_char),  sizeof(cl_ushort), sizeof(cl_short),
    sizeof(cl_uint),   sizeof(cl_int),   sizeof(cl_half),   sizeof(cl_float),
    sizeof(cl_double), sizeof(cl_ulong), sizeof(cl_long),
};

const char *sizeNames[] = { "", "", "2", "3", "4", "8", "16" };
const int vectorSizes[] = { 1, 1, 2, 3, 4, 8, 16 };
int gMinVectorSize = 0;
int gMaxVectorSize = sizeof(vectorSizes) / sizeof(vectorSizes[0]);

cl_context gContext = NULL;
int gWimpyReductionFactor = 128;
int gSkipTesting = 0;
int gForceFTZ = 0;
int gIsRTZ = 0;
int gForceHalfFTZ = 0;
int gIsHalfRTZ = 0;
int gHasDouble = 0;
int gTestDouble = 1;
int gHasHalfs = 0;
int gTestHalfs = 1;
bool gTestAll = false;

std::recursive_mutex gLock;
static std::mutex gThreadPoolLock;

cl_half_rounding_mode gDefaultHalfRoundingMode = CL_HALF_RTE;

std::vector<struct buffers> buffers_vec;
static test_status acquire_buffers(cl_device_id device, struct buffers &buffers)
{
    std::lock_guard<std::recursive_mutex> guard(gLock);
    if (buffers_vec.empty())
    {
        struct buffers tmp_buffers;
        cl_int error;
        // Allocate buffers
        // FIXME: use clProtectedArray for guarded allocations?
        tmp_buffers.in = malloc(BUFFER_SIZE + 2 * kPageSize);
        tmp_buffers.allowZ = malloc(BUFFER_SIZE + 2 * kPageSize);
        tmp_buffers.ref = malloc(BUFFER_SIZE + 2 * kPageSize);
        for (uint32_t i = 0; i < kCallStyleCount; i++)
        {
            tmp_buffers.out[i] = malloc(BUFFER_SIZE + 2 * kPageSize);
            if (NULL == tmp_buffers.out[i]) return TEST_FAIL;
        }

        // setup input buffers
        tmp_buffers.inBuffer =
            clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                           BUFFER_SIZE, NULL, &error);
        if (tmp_buffers.inBuffer == NULL || error)
        {
            vlog_error("clCreateBuffer failed for input (%d)\n", error);
            return TEST_FAIL;
        }

        // setup output buffers
        for (uint32_t i = 0; i < kCallStyleCount; i++)
        {
            tmp_buffers.outBuffers[i] = clCreateBuffer(
                gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                BUFFER_SIZE, NULL, &error);
            if (tmp_buffers.outBuffers[i] == NULL || error)
            {
                vlog_error("clCreateArray failed for output (%d)\n", error);
                return TEST_FAIL;
            }
        }
        tmp_buffers.queue = clCreateCommandQueue(gContext, device, 0, &error);
        if (tmp_buffers.queue == NULL || error)
        {
            vlog_error("clCreateCommandQueue failed (%d)\n", error);
            return TEST_FAIL;
        }
        buffers_vec.push_back(tmp_buffers);
    }
    buffers = buffers_vec.back();
    buffers_vec.pop_back();
    return TEST_PASS;
}
static void release_buffers(struct buffers buffers)
{
    std::lock_guard<std::recursive_mutex> guard(gLock);
    buffers_vec.push_back(buffers);
}

// Windows (since long double got deprecated) sets the x87 to 53-bit precision
// (that's x87 default state).  This causes problems with the tests that
// convert long and ulong to float and double or otherwise deal with values
// that need more precision than 53-bit. So, set the x87 to 64-bit precision.
static inline void Force64BitFPUPrecision(void)
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
#else
    /* Implement for other platforms if needed */
#endif
}

template <typename Ty> static bool is_nan(void *in_ptr, uint32_t idx)
{
    if constexpr (std::is_same_v<Ty, cl_half>)
    {
        cl_half val = static_cast<cl_half *>(in_ptr)[idx];
        return (val & 0x7fff) > 0x7c00;
    }
    else if constexpr (std::is_same_v<Ty, float>)
    {
        cl_uint val = reinterpret_cast<cl_uint *>(in_ptr)[idx];
        return (val & 0x7fffffffU) > 0x7f800000U;
    }
    else if constexpr (std::is_same_v<Ty, double>)
    {
        cl_ulong val = reinterpret_cast<cl_ulong *>(in_ptr)[idx];
        return (val & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL;
    }
    return false;
}

template <typename InType, typename OutType, bool InFP, bool OutFP>
int CalcRefValsPat<InType, OutType, InFP, OutFP>::check_result(void *test,
                                                               uint32_t count,
                                                               int vectorSize)
{
    const cl_uchar *a = (const cl_uchar *)buffers.allowZ;

    if constexpr (is_half<OutType, OutFP>())
    {
        const cl_half *t = (const cl_half *)test;
        const cl_half *c = (const cl_half *)buffers.ref;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !(is_nan<OutType>(test, i) && is_nan<OutType>(buffers.ref, i))
                && !(a[i] != (cl_uchar)0 && t[i] == (c[i] & 0x8000)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, HTF(c[i]), HTF(t[i]));
                return i + 1;
            }
    }
    else if constexpr (std::is_integral<OutType>::value)
    { // char/uchar/short/ushort/half/int/uint/long/ulong
        const OutType *t = (const OutType *)test;
        const OutType *c = (const OutType *)buffers.ref;
        for (uint32_t i = 0; i < count; i++)
        {
            if constexpr (InFP)
                if (is_nan<InType>(buffers.in, i)) continue;

            if (t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (OutType)0))
            {
                size_t s = sizeof(OutType) * 2;
                std::stringstream sstr;
                sstr << "\nError for vector size %d found at 0x%8.8x:  *0x%"
                     << s << "." << s << "x vs 0x%" << s << "." << s << "x\n";
                vlog(sstr.str().c_str(), vectorSize, i, c[i], t[i]);
                return i + 1;
            }
        }
    }
    else if constexpr (std::is_same<OutType, cl_float>::value)
    {
        // cast to integral - from original test
        const cl_uint *t = (const cl_uint *)test;
        const cl_uint *c = (const cl_uint *)buffers.ref;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !(is_nan<OutType>(test, i) && is_nan<OutType>(buffers.ref, i))
                && !(a[i] != (cl_uchar)0 && t[i] == (c[i] & 0x80000000U)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((OutType *)buffers.ref)[i],
                    ((OutType *)test)[i]);
                return i + 1;
            }
    }
    else
    {
        const cl_ulong *t = (const cl_ulong *)test;
        const cl_ulong *c = (const cl_ulong *)buffers.ref;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !(is_nan<OutType>(test, i) && is_nan<OutType>(buffers.ref, i))
                && !(a[i] != (cl_uchar)0
                     && t[i] == (c[i] & 0x8000000000000000ULL)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((OutType *)buffers.ref)[i],
                    ((OutType *)test)[i]);
                return i + 1;
            }
    }

    return 0;
}


static cl_uint RoundUpToNextPowerOfTwo(cl_uint x)
{
    if (0 == (x & (x - 1))) return x;

    while (x & (x - 1)) x &= x - 1;

    return x + x;
}

template <typename T, bool IsFP> struct TypeTag
{
    using type = T;
    static constexpr bool is_fp = IsFP;
};

template <typename F> test_status dispatch_type(Type t, F &&f)
{
    switch (t)
    {
        case kuchar: return f(TypeTag<cl_uchar, false>{});
        case kchar: return f(TypeTag<cl_char, false>{});
        case kushort: return f(TypeTag<cl_ushort, false>{});
        case kshort: return f(TypeTag<cl_short, false>{});
        case kuint: return f(TypeTag<cl_uint, false>{});
        case kint: return f(TypeTag<cl_int, false>{});
        case khalf: return f(TypeTag<cl_half, true>{});
        case kfloat: return f(TypeTag<cl_float, true>{});
        case kdouble: return f(TypeTag<cl_double, true>{});
        case kulong: return f(TypeTag<cl_ulong, false>{});
        case klong: return f(TypeTag<cl_long, false>{});
        default: return TEST_SKIP;
    }
}

int RunTest(cl_device_id device, cl_context context, cl_command_queue queue,
            int num_elements, void *arg)
{
    ConversionsTest *test = (ConversionsTest *)arg;
    return dispatch_type(test->inType, [&](auto in_tag) {
        using InType = typename decltype(in_tag)::type;
        constexpr bool InFP = decltype(in_tag)::is_fp;
        return dispatch_type(test->outType, [&](auto out_tag) {
            using OutType = typename decltype(out_tag)::type;
            constexpr bool OutFP = decltype(out_tag)::is_fp;

            struct buffers buffers;
            if (acquire_buffers(device, buffers) != TEST_PASS)
            {
                vlog_error("\t\tFAILED -- Could not acquire buffers.\n");
                return TEST_FAIL;
            }
            test_status status =
                test->DoTest<InType, OutType, InFP, OutFP>(buffers);
            release_buffers(buffers);
            return status;
        });
    });
}

template <typename InType, typename OutType, bool InFP, bool OutFP>
test_status ConversionsTest::DoTest(struct buffers &buffers)
{
#ifdef __APPLE__
    cl_ulong wall_start = mach_absolute_time();
#endif
    if ((!gTestDouble && (outType == Type::kdouble || inType == Type::kdouble))
        || (!gTestHalfs && (outType == Type::khalf || inType == Type::khalf))
        || (!gHasLong
            && (outType == Type::klong || outType == Type::kulong
                || inType == Type::klong || inType == Type::kulong)))
    {
        return TEST_SKIPPED_ITSELF;
    }

    cl_uint threads = GetThreadCount();

    DataInitInfo info = { 0, 0, outType, inType, sat, round, threads, buffers };
    DataInfoSpec<InType, OutType, InFP, OutFP> init_info(info);
    std::vector<std::unique_ptr<CalcRefValsBase>> calcInfo;
    int vectorSize;
    int error = 0;
    uint64_t i;
    // Skip implicit test (index 0) because implicit can't saturate/round
    int minVectorSize =
        (gMinVectorSize == 0 && (sat || round != kDefaultRoundingMode))
        ? 1
        : gMinVectorSize;

    size_t blockCount =
        BUFFER_SIZE / std::max(gTypeSizes[inType], gTypeSizes[outType]);
    size_t step = blockCount;

    for (i = 0; i < threads; i++)
    {
        init_info.mdv.emplace_back(MTdataHolder(gRandomSeed));
    }

    calcInfo.resize(gMaxVectorSize);
    for (vectorSize = minVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        calcInfo[vectorSize].reset(
            new CalcRefValsPat<InType, OutType, InFP, OutFP>(buffers));
        calcInfo[vectorSize]->program =
            conv_test::MakeProgram(outType, inType, sat, round, vectorSize,
                                   &calcInfo[vectorSize]->kernel);
        if (NULL == calcInfo[vectorSize]->program)
        {
            return TEST_FAIL;
        }
        if (NULL == calcInfo[vectorSize]->kernel)
        {
            vlog_error("\t\tFAILED -- Failed to create kernel.\n");
            return TEST_FAIL;
        }
    }

    if (gSkipTesting) return TEST_SKIPPED_ITSELF;

    // Patch up rounding mode if default is RTZ
    // We leave the part above in default rounding mode so that the right kernel
    // is compiled.
    if (std::is_same<OutType, cl_float>::value)
    {
        if (round == kDefaultRoundingMode && gIsRTZ)
            init_info.round = round = kRoundTowardZero;
    }
    else if (std::is_same<OutType, cl_half>::value && OutFP)
    {
        if (round == kDefaultRoundingMode && gIsHalfRTZ)
            init_info.round = round = kRoundTowardZero;
    }

    uint64_t nbInputs = (1ULL << 25);
    if constexpr (sizeof(InType) <= 2)
    {
        nbInputs = (1ULL << (sizeof(InType) * 8));
    }
    else
    {
        if (gTestAll)
        {
            nbInputs = (1ULL << 32);
        }
        else
        {
            if (gWimpyMode)
            {
                nbInputs /= gWimpyReductionFactor;
            }
            if (gIsEmbedded)
            {
                nbInputs /= EMBEDDED_REDUCTION_FACTOR;
            }
        }
    }

    vlog("Testing... ");
    fflush(stdout);
    for (i = 0; i < (uint64_t)nbInputs; i += step)
    {

        if (0 == (i & ((nbInputs >> 3) - 1)))
        {
            vlog(".");
            fflush(stdout);
        }

        //      Call this in a multithreaded manner
        cl_uint chunks = RoundUpToNextPowerOfTwo(threads) * 2;
        init_info.start = i;
        init_info.size = blockCount / chunks;
        if (init_info.size < 16384)
        {
            chunks = RoundUpToNextPowerOfTwo(threads);
            init_info.size = blockCount / chunks;
            if (init_info.size < 16384)
            {
                init_info.size = blockCount;
                chunks = 1;
            }
        }

        {
            std::lock_guard<std::mutex> lock(gThreadPoolLock);
            ThreadPool_Do(conv_test::InitData, chunks, &init_info);
        }

        // Copy the inputs to the device
        if ((error = clEnqueueWriteBuffer(
                 buffers.queue, buffers.inBuffer, CL_FALSE, 0,
                 blockCount * gTypeSizes[inType], buffers.in, 0, NULL, NULL)))
        {
            vlog_error("ERROR: clEnqueueWriteBuffer failed. (%d)\n", error);
            return TEST_FAIL;
        }

        for (vectorSize = minVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            const cl_uint pattern = 0xffffdead;
            if ((error = clEnqueueFillBuffer(
                     buffers.queue, buffers.outBuffers[vectorSize], &pattern,
                     sizeof(pattern), 0, blockCount * gTypeSizes[outType], 0,
                     NULL, NULL)))
            {
                vlog_error("ERROR: clEnqueueFillBuffer failed. (%d)\n", error);
                return TEST_FAIL;
            }

            error = clSetKernelArg(calcInfo[vectorSize]->kernel, 0,
                                   sizeof(buffers.inBuffer), &buffers.inBuffer);
            error |= clSetKernelArg(calcInfo[vectorSize]->kernel, 1,
                                    sizeof(buffers.outBuffers[vectorSize]),
                                    &buffers.outBuffers[vectorSize]);
            if (error)
            {
                vlog_error("FAILED -- could not set kernel args (%d)\n", error);
                return TEST_FAIL;
            }

            size_t workItemCount = (blockCount + vectorSizes[vectorSize] - 1)
                / (vectorSizes[vectorSize]);
            if ((error = clEnqueueNDRangeKernel(
                     buffers.queue, calcInfo[vectorSize]->kernel, 1, NULL,
                     &workItemCount, NULL, 0, NULL, NULL)))
            {
                vlog_error("FAILED -- could not execute kernel (%d)\n", error);
                return TEST_FAIL;
            }

            if ((error = clEnqueueReadBuffer(
                     buffers.queue, buffers.outBuffers[vectorSize], CL_FALSE, 0,
                     blockCount * gTypeSizes[outType], buffers.out[vectorSize],
                     0, NULL, &calcInfo[vectorSize]->event)))
            {
                vlog_error("ERROR: WriteInputBufferComplete calback failed "
                           "with status: %d\n",
                           error);
                return TEST_FAIL;
            }
        }

        // Make sure the work is actually running, so we don't deadlock
        if ((error = clFlush(buffers.queue)))
        {
            vlog_error("clFlush failed with error %d\n", error);
            return TEST_FAIL;
        }

        {
            std::lock_guard<std::mutex> lock(gThreadPoolLock);
            ThreadPool_Do(conv_test::PrepareReference, chunks, &init_info);
        }

        for (vectorSize = minVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            auto &info = calcInfo[vectorSize];
            if ((error = clWaitForEvents(1, &info->event)))
            {
                vlog_error("ERROR: clWaitForEvents failed. (%d)\n", error);
                return TEST_FAIL;
            }
            if ((error = clReleaseEvent(info->event)))
            {
                vlog_error("ERROR: clReleaseEvent failed. (%d)\n", error);
                return TEST_FAIL;
            }
            // verify results
            if (memcmp(info->buffers.out[vectorSize], info->buffers.ref,
                       blockCount * gTypeSizes[outType]))
                error = info->check_result(info->buffers.out[vectorSize],
                                           blockCount, vectorSizes[vectorSize]);

            if (error)
            {
                switch (inType)
                {
                    case kuchar:
                    case kchar:
                        vlog("Input value: 0x%2.2x ",
                             ((unsigned char *)buffers.in)[error - 1]);
                        break;
                    case kushort:
                    case kshort:
                        vlog("Input value: 0x%4.4x ",
                             ((unsigned short *)buffers.in)[error - 1]);
                        break;
                    case kuint:
                    case kint:
                        vlog("Input value: 0x%8.8x ",
                             ((unsigned int *)buffers.in)[error - 1]);
                        break;
                    case khalf:
                        vlog("Input value: %a ",
                             HTF(((cl_half *)buffers.in)[error - 1]));
                        break;
                    case kfloat:
                        vlog("Input value: %a ",
                             ((float *)buffers.in)[error - 1]);
                        break;
                    case kulong:
                    case klong:
                        vlog("Input value: 0x%16.16llx ",
                             ((unsigned long long *)buffers.in)[error - 1]);
                        break;
                    case kdouble:
                        vlog("Input value: %a ",
                             ((double *)buffers.in)[error - 1]);
                        break;
                    default:
                        vlog_error("Internal error at %s: %d\n", __FILE__,
                                   __LINE__);
                        abort();
                        break;
                }

                // tell the user which conversion it was.
                if (0 == vectorSize)
                    vlog(" (implicit scalar conversion from %s to %s)\n",
                         gTypeNames[inType], gTypeNames[outType]);
                else
                    vlog(" (convert_%s%s%s%s( %s%s ))\n", gTypeNames[outType],
                         sizeNames[vectorSize], gSaturationNames[sat],
                         gRoundingModeNames[round], gTypeNames[inType],
                         sizeNames[vectorSize]);

                return TEST_FAIL;
            }
        }
    }

    if ((error = clFinish(buffers.queue)))
    {
        vlog_error("clFinish failed with error %d\n", error);
        return TEST_FAIL;
    }

    log_info("done.\n");

    if (gWimpyMode)
        vlog("\tWimp pass");
    else
        vlog("\tpassed");

#ifdef __APPLE__
    // record the run time
    vlog("\t(%f s)", 1e-9 * (mach_absolute_time() - wall_start));
#endif
    vlog("\n\n");
    fflush(stdout);

    return TEST_PASS;
}

namespace conv_test {

cl_int InitData(cl_uint job_id, cl_uint thread_id, void *p)
{
    DataInitBase *info = (DataInitBase *)p;

    info->init(job_id, thread_id);

    return CL_SUCCESS;
}

cl_int PrepareReference(cl_uint job_id, cl_uint thread_id, void *p)
{
    DataInitBase *info = (DataInitBase *)p;

    cl_uint count = info->size;
    Type inType = info->inType;
    Type outType = info->outType;
    RoundingMode round = info->round;

    Force64BitFPUPrecision();

    void *s = (cl_uchar *)info->buffers.in
        + job_id * count * gTypeSizes[info->inType];
    void *a = (cl_uchar *)info->buffers.allowZ + job_id * count;
    void *d = (cl_uchar *)info->buffers.ref
        + job_id * count * gTypeSizes[info->outType];

    if (outType != inType)
    {
        // create the reference while we wait
#if CONVERSIONS_QCOM
        /* ARM VFP doesn't have hardware instruction for converting from 64-bit
         * integer to float types, hence GCC ARM uses the floating-point
         * emulation code despite which -mfloat-abi setting it is. But the
         * emulation code in libgcc.a has only one rounding mode (round to
         * nearest even in this case) and ignores the user rounding mode setting
         * in hardware. As a result setting rounding modes in hardware won't
         * give correct rounding results for type covert from 64-bit integer to
         * float using GCC for ARM compiler so for testing different rounding
         * modes, we need to use alternative reference function. ARM64 does have
         * an instruction, however we cannot guarantee the compiler will use it.
         * On all ARM architechures use emulation to calculate reference.*/
        switch (round)
        {
            /* conversions to floating-point type use the current rounding mode.
             * The only default floating-point rounding mode supported is round
             * to nearest even i.e the current rounding mode will be _rte for
             * floating-point types. */
            case kDefaultRoundingMode: info->qcom_rm = qcomRTE; break;
            case kRoundToNearestEven: info->qcom_rm = qcomRTE; break;
            case kRoundUp: info->qcom_rm = qcomRTP; break;
            case kRoundDown: info->qcom_rm = qcomRTN; break;
            case kRoundTowardZero: info->qcom_rm = qcomRTZ; break;
            default:
                vlog_error("ERROR: undefined rounding mode %d\n", round);
                break;
        }
#endif

        RoundingMode oldRound;
        if (outType == khalf)
        {
            oldRound = set_round(kRoundToNearestEven, kfloat);
            switch (round)
            {
                default:
                case kDefaultRoundingMode:
                    info->halfRoundingMode = gDefaultHalfRoundingMode;
                    break;
                case kRoundToNearestEven:
                    info->halfRoundingMode = CL_HALF_RTE;
                    break;
                case kRoundUp: info->halfRoundingMode = CL_HALF_RTP; break;
                case kRoundDown: info->halfRoundingMode = CL_HALF_RTN; break;
                case kRoundTowardZero:
                    info->halfRoundingMode = CL_HALF_RTZ;
                    break;
            }
        }
        else
            oldRound = set_round(round, outType);

        if (info->sat)
            info->conv_array_sat(d, s, count);
        else
            info->conv_array(d, s, count);

        set_round(oldRound, outType);

        // Decide if we allow a zero result in addition to the correctly rounded
        // one
        memset(a, 0, count);
        if (gForceFTZ && (inType == kfloat || outType == kfloat))
        {
            info->set_allow_zero_array((uint8_t *)a, d, s, count);
        }
        if (gForceHalfFTZ && (inType == khalf || outType == khalf))
        {
            info->set_allow_zero_array((uint8_t *)a, d, s, count);
        }
    }
    else
    {
        // Copy the input to the reference
        memcpy(d, s, info->size * gTypeSizes[inType]);
    }

    return CL_SUCCESS;
}

cl_program MakeProgram(Type outType, Type inType, SaturationMode sat,
                       RoundingMode round, int vectorSize, cl_kernel *outKernel)
{
    cl_program program;
    char testName[256];
    int error = 0;

    std::ostringstream source;
    if (outType == kdouble || inType == kdouble)
        source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    if (outType == khalf || inType == khalf)
        source << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

    // Create the program. This is a bit complicated because we are trying to
    // avoid byte and short stores.
    if (0 == vectorSize)
    {
        // Create the type names.
        char inName[32];
        char outName[32];
        strncpy(inName, gTypeNames[inType], sizeof(inName));
        inName[sizeof(inName) - 1] = '\0';
        strncpy(outName, gTypeNames[outType], sizeof(outName));
        outName[sizeof(outName) - 1] = '\0';
        sprintf(testName, "test_implicit_%s_%s", outName, inName);

        source << "__kernel void " << testName << "( __global " << inName
               << " *src, __global " << outName << " *dest )\n";
        source << "{\n";
        source << "   size_t i = get_global_id(0);\n";
        source << "   dest[i] =  src[i];\n";
        source << "}\n";

        vlog("Building implicit %s -> %s conversion test\n", gTypeNames[inType],
             gTypeNames[outType]);
        fflush(stdout);
    }
    else
    {
        int vectorSizetmp = vectorSizes[vectorSize];

        // Create the type names.
        char convertString[128];
        char inName[32];
        char outName[32];
        switch (vectorSizetmp)
        {
            case 1:
                strncpy(inName, gTypeNames[inType], sizeof(inName) - 1);
                inName[sizeof(inName) - 1] = '\0';
                strncpy(outName, gTypeNames[outType], sizeof(outName) - 1);
                outName[sizeof(outName) - 1] = '\0';
                snprintf(convertString, sizeof(convertString), "convert_%s%s%s",
                         outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s", convertString, inName);
                vlog("Building %s( %s ) test\n", convertString, inName);
                break;
            case 3:
                strncpy(inName, gTypeNames[inType], sizeof(inName) - 1);
                inName[sizeof(inName) - 1] = '\0';
                strncpy(outName, gTypeNames[outType], sizeof(outName) - 1);
                outName[sizeof(outName) - 1] = '\0';
                snprintf(convertString, sizeof(convertString),
                         "convert_%s3%s%s", outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s3", convertString, inName);
                vlog("Building %s( %s3 ) test\n", convertString, inName);
                break;
            default:
                snprintf(inName, sizeof(inName), "%s%d", gTypeNames[inType],
                         vectorSizetmp);
                snprintf(outName, sizeof(outName), "%s%d", gTypeNames[outType],
                         vectorSizetmp);
                snprintf(convertString, sizeof(convertString), "convert_%s%s%s",
                         outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s", convertString, inName);
                vlog("Building %s( %s ) test\n", convertString, inName);
                break;
        }
        fflush(stdout);

        if (vectorSizetmp == 3)
        {
            source << "__kernel void " << testName << "( __global " << inName
                   << " *src, __global " << outName << " *dest )\n";
            source << "{\n";
            source << "   size_t i = get_global_id(0);\n";
            source << "   if( i + 1 < get_global_size(0))\n";
            source << "       vstore3( " << convertString
                   << "( vload3( i, src)), i, dest );\n";
            source << "   else\n";
            source << "   {\n";
            source << "       " << inName << "3 in;\n";
            source << "       " << outName << "3 out;\n";
            source << "       if( 0 == (i & 1) )\n";
            source << "           in.y = src[3*i+1];\n";
            source << "       in.x = src[3*i];\n";
            source << "       out = " << convertString << "( in ); \n";
            source << "       dest[3*i] = out.x;\n";
            source << "       if( 0 == (i & 1) )\n";
            source << "           dest[3*i+1] = out.y;\n";
            source << "   }\n";
            source << "}\n";
        }
        else
        {
            source << "__kernel void " << testName << "( __global " << inName
                   << " *src, __global " << outName << " *dest )\n";
            source << "{\n";
            source << "   size_t i = get_global_id(0);\n";
            source << "   dest[i] = " << convertString << "( src[i] );\n";
            source << "}\n";
        }
    }
    *outKernel = NULL;

    const char *flags = NULL;
    if ((gForceFTZ && (inType == kfloat || outType == kfloat))
        || (gForceHalfFTZ && (inType == khalf || outType == khalf)))
    {
        flags = "-cl-denorms-are-zero";
    }

    // build it
    std::string sourceString = source.str();
    const char *programSource = sourceString.c_str();
    error = create_single_kernel_helper(gContext, &program, outKernel, 1,
                                        &programSource, testName, flags);
    if (error)
    {
        vlog_error("Failed to build kernel/program (err = %d).\n", error);
        return NULL;
    }

    return program;
}

} // namespace conv_test
