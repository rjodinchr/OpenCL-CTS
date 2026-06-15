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
#include "harness/ThreadPool.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"
#include "harness/mt19937.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
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
#include <stdio.h>
#include <string.h>
#if !defined(_WIN32)
#include <libgen.h>
#include <sys/mman.h>
#endif
#include <time.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "Sleep.h"

#include "basic_test_conversions.h"
#include "conversions_data_info.h"

#include <climits>
#include <cstring>

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#include "fplib.h"
#endif

static test_status ParseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help);
static test_status InitCL(cl_device_id device);

static int gMultithread = 1;
static std::vector<ConversionsTest> tests;

int main(int argc, const char **argv)
{
    // Turn off sleep so our tests run to completion
    PreventSleep();
    atexit(ResumeSleep);

#if defined(_MSC_VER) && defined(_M_IX86)
    // VS2005 (and probably others, since long double got deprecated) sets
    // the x87 to 53-bit precision. This causes problems with the tests
    // that convert long and ulong to float and double, since they deal
    // with values that need more precision than that. So, set the x87
    // to 64-bit precision.
    unsigned int ignored;
    _controlfp_s(&ignored, _PC_64, _MCW_PC);
#endif

    int ret =
        runTestHarnessWithCheckAndParse(argc, argv, true, 0, InitCL, ParseArgs);

    for (auto &buffers : buffers_vec)
    {
        clReleaseMemObject(buffers.inBuffer);
        free(buffers.in);
        free(buffers.allowZ);
        free(buffers.ref);

        for (int i = 0; i < kCallStyleCount; i++)
        {
            clReleaseMemObject(buffers.outBuffers[i]);
            free(buffers.out[i]);
        }
        clReleaseCommandQueue(buffers.queue);
    }
    clReleaseContext(gContext);

    return ret;
}


static test_status ParseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{
    int i;

    help = R"(
        -d     Toggle testing of double precision.  On by default if cl_khr_fp64 is enabled, ignored otherwise.
        -l     Toggle link check mode. When on, testing is skipped, and we just check to see that the kernels build. (Off by default.)
        -m     Toggle Multithreading. (On by default.)
        -[2^n] Set wimpy reduction factor, recommended range of n is 1-12, default factor()"
        + std::to_string(gWimpyReductionFactor) + R"()
        -z     Toggle flush to zero mode  (Default: per device)
        -a     Test 2^32 values, not just special & random values. (default: off)
        -#     Test just vector size given by #, where # is an element of the set {1,2,3,4,8,16}

Test names:
        destFormat<_sat><_round>_sourceFormat
        Possible format types are:
            )";
    for (i = 0; i < kTypeCount; i++) help += std::string(gTypeNames[i]) + ", ";
    help += R"(
        Possible saturation values are: (empty) and _sat
        Possible rounding values are:
            (empty), )";
    for (i = 1; i < kRoundingModeCount; i++)
        help += std::string(gRoundingModeNames[i]) + ", ";
    help += R"(
        Examples:
          ulong_short          converts short to ulong
          char_sat_rte_float   converts float to char with saturated clipping in round to nearest rounding mode
)";

    for (unsigned dst = 0; dst < kTypeCount; dst++)
    {
        for (unsigned src = 0; src < kTypeCount; src++)
        {
            for (unsigned sat = 0; sat < 2; sat++)
            {
                // skip illegal saturated conversions to float type
                if (gSaturationNames[sat] == std::string("_sat")
                    && (gTypeNames[dst] == std::string("float")
                        || gTypeNames[dst] == std::string("half")
                        || gTypeNames[dst] == std::string("double")))
                {
                    continue;
                }
                for (unsigned rnd = 0; rnd < kRoundingModeCount; rnd++)
                {
                    std::string name;
                    name += gTypeNames[dst];
                    name += gSaturationNames[sat];
                    name += gRoundingModeNames[rnd];
                    name += "_";
                    name += gTypeNames[src];
                    tests.push_back({ (Type)dst, (Type)src, (SaturationMode)sat,
                                      (RoundingMode)rnd, name });
                }
            }
        }
    }
    for (const auto &test : tests)
    {
        test_registry::getInstance().add_test(RunTest, test.name.c_str(),
                                              Version(1, 2), (void *)&test);
    }

    std::vector<const char *> argList;
    argList.push_back(argv[0]);
    for (i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        if (NULL == arg) break;

        if (arg[0] == '-')
        {
            arg++;
            while (*arg != '\0')
            {
                switch (*arg)
                {
                    case 'd': gTestDouble ^= 1; break;
                    case 'h': gTestHalfs ^= 1; break;
                    case 'l': gSkipTesting ^= 1; break;
                    case 'm': gMultithread ^= 1; break;
                    case '[':
                        parseWimpyReductionFactor(arg, gWimpyReductionFactor);
                        break;
                    case 'z':
                        gForceFTZ ^= 1;
                        gForceHalfFTZ ^= 1;
                        break;
                    case 'a': gTestAll ^= 1; break;
                    case '1':
                        if (arg[1] == '6')
                        {
                            gMinVectorSize = 6;
                            gMaxVectorSize = 7;
                            arg++;
                        }
                        else
                        {
                            gMinVectorSize = 0;
                            gMaxVectorSize = 2;
                        }
                        break;

                    case '2':
                        gMinVectorSize = 2;
                        gMaxVectorSize = 3;
                        break;

                    case '3':
                        gMinVectorSize = 3;
                        gMaxVectorSize = 4;
                        break;

                    case '4':
                        gMinVectorSize = 4;
                        gMaxVectorSize = 5;
                        break;

                    case '8':
                        gMinVectorSize = 5;
                        gMaxVectorSize = 6;
                        break;

                    default:
                        vlog(" <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg);
                        return TEST_FAIL;
                }
                arg++;
            }
            removed_args.push_back(argv[i]);
        }
        else
        {
            argList.push_back(arg);
        }
    }
    update_argc_argv_from_args_list(argList, argc, argv);

    if (!gMultithread) SetThreadCount(1);

    return TEST_PASS;
}


static test_status InitCL(cl_device_id device)
{
    int error, i;
    cl_uint gComputeDevices = 0;
    if ((error =
             clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                             sizeof(gComputeDevices), &gComputeDevices, NULL)))
        gComputeDevices = 1;

    cl_uint gDeviceFrequency = 0;
    if ((error = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 sizeof(gDeviceFrequency), &gDeviceFrequency,
                                 NULL)))
        gDeviceFrequency = 0;

    cl_device_fp_config floatCapabilities = 0;
    if ((error = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
                                 sizeof(floatCapabilities), &floatCapabilities,
                                 NULL)))
        floatCapabilities = 0;
    if (0 == (CL_FP_DENORM & floatCapabilities)) gForceFTZ ^= 1;

    if (0 == (floatCapabilities & CL_FP_ROUND_TO_NEAREST))
    {
        char profileStr[128] = "";
        // Verify that we are an embedded profile device
        if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE,
                                     sizeof(profileStr), profileStr, NULL)))
        {
            vlog_error("FAILURE: Could not get device profile: error %d\n",
                       error);
            return TEST_FAIL;
        }

        if (strcmp(profileStr, "EMBEDDED_PROFILE"))
        {
            vlog_error("FAILURE: non-embedded profile device does not support "
                       "CL_FP_ROUND_TO_NEAREST\n");
            return TEST_FAIL;
        }

        if (0 == (floatCapabilities & CL_FP_ROUND_TO_ZERO))
        {
            vlog_error("FAILURE: embedded profile device supports neither "
                       "CL_FP_ROUND_TO_NEAREST or CL_FP_ROUND_TO_ZERO\n");
            return TEST_FAIL;
        }

        gIsRTZ = 1;
    }

    else if (is_extension_available(device, "cl_khr_fp64"))
    {
        gHasDouble = 1;
    }
    gTestDouble &= gHasDouble;

    if (is_extension_available(device, "cl_khr_fp16"))
    {
        gHasHalfs = 1;

        cl_device_fp_config floatCapabilities = 0;
        if ((error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG,
                                     sizeof(floatCapabilities),
                                     &floatCapabilities, NULL)))
            floatCapabilities = 0;

        if (0 == (CL_FP_DENORM & floatCapabilities)) gForceHalfFTZ ^= 1;

        if (0 == (floatCapabilities & CL_FP_ROUND_TO_NEAREST))
        {
            char profileStr[128] = "";
            // Verify that we are an embedded profile device
            if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE,
                                         sizeof(profileStr), profileStr, NULL)))
            {
                vlog_error("FAILURE: Could not get device profile: error %d\n",
                           error);
                return TEST_FAIL;
            }

            if (strcmp(profileStr, "EMBEDDED_PROFILE"))
            {
                vlog_error(
                    "FAILURE: non-embedded profile device does not support "
                    "CL_FP_ROUND_TO_NEAREST\n");
                return TEST_FAIL;
            }

            if (0 == (floatCapabilities & CL_FP_ROUND_TO_ZERO))
            {
                vlog_error("FAILURE: device supports neither "
                           "CL_FP_ROUND_TO_NEAREST or CL_FP_ROUND_TO_ZERO\n");
                return TEST_FAIL;
            }

            gIsHalfRTZ = 1;
            gDefaultHalfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            gDefaultHalfRoundingMode = CL_HALF_RTE;
        }
    }
    gTestHalfs &= gHasHalfs;

    // detect whether profile of the device is embedded
    char profile[1024] = "";
    if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile),
                                 profile, NULL)))
    {
        vlog_error("clGetDeviceInfo failed. (%d)\n", error);
        return TEST_FAIL;
    }
    else if (strstr(profile, "EMBEDDED_PROFILE"))
    {
        gIsEmbedded = 1;
        if (!is_extension_available(device, "cles_khr_int64")) gHasLong = 0;
    }

    gContext = clCreateContext(NULL, 1, &device, notify_callback, NULL, &error);
    if (NULL == gContext || error)
    {
        vlog_error("clCreateContext failed. (%d)\n", error);
        return TEST_FAIL;
    }

    char c[1024];
    static const char *no_yes[] = { "NO", "YES" };
    vlog("\nCompute Device info:\n");
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(c), c, NULL);
    vlog("\tDevice Name: %s\n", c);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(c), c, NULL);
    vlog("\tVendor: %s\n", c);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(c), c, NULL);
    vlog("\tDevice Version: %s\n", c);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(c), &c, NULL);
    vlog("\tCL C Version: %s\n", c);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(c), c, NULL);
    vlog("\tDriver Version: %s\n", c);
    vlog("\tProcessing with %u devices\n", gComputeDevices);
    vlog("\tDevice Frequency: %u MHz\n", gDeviceFrequency);
    vlog("\tSubnormal values supported for floats? %s\n",
         no_yes[0 != (CL_FP_DENORM & floatCapabilities)]);
    vlog("\tTesting with FTZ mode ON for floats? %s\n", no_yes[0 != gForceFTZ]);
    vlog("\tTesting with FTZ mode ON for halfs? %s\n",
         no_yes[0 != gForceHalfFTZ]);
    vlog("\tTesting with default RTZ mode for floats? %s\n",
         no_yes[0 != gIsRTZ]);
    vlog("\tTesting with default RTZ mode for halfs? %s\n",
         no_yes[0 != gIsHalfRTZ]);
    vlog("\tHas Double? %s\n", no_yes[0 != gHasDouble]);
    if (gHasDouble) vlog("\tTest Double? %s\n", no_yes[0 != gTestDouble]);
    vlog("\tHas Long? %s\n", no_yes[0 != gHasLong]);
    vlog("\tTesting vector sizes: ");
    for (i = gMinVectorSize; i < gMaxVectorSize; i++)
        vlog("\t%d", vectorSizes[i]);
    vlog("\n");
    return TEST_PASS;
}
