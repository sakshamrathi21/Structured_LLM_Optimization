// nvml_utilization.cpp
#include <nvml.h>
#include <cuda.h>
#include <stdio.h>
#include <unistd.h>   // for sleep
#include <vector>

int main() {
    nvmlReturn_t result;
    nvmlDevice_t device;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return -1;
    }

    // Get handle for GPU 0
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get handle for device 0: %s\n", nvmlErrorString(result));
        return -1;
    }

    // Also get device properties from CUDA driver API
    CUdevice cuDevice;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);

    int numSms = 0;
    cuDeviceGetAttribute(&numSms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);

    // Poll every second
    for (int i = 0; i < 20; i++) {
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if (result != NVML_SUCCESS) {
            printf("Failed to get utilization: %s\n", nvmlErrorString(result));
            break;
        }

        // Estimate number of active SMs
        int activeSms = (utilization.gpu * numSms) / 100;

        // Get memory info
        nvmlMemory_t memInfo;
        nvmlDeviceGetMemoryInfo(device, &memInfo);

        double memUsedMB = (double)memInfo.used / (1024.0 * 1024.0);
        double memTotalMB = (double)memInfo.total / (1024.0 * 1024.0);
        double memUsedGB = memUsedMB / 1024.0;
        double memTotalGB = memTotalMB / 1024.0;
        double memUsedPercent = (double)memInfo.used / memInfo.total * 100.0;

        printf("=== Global GPU Stats ===\n");
        printf("GPU Util: %u%% (%d / %d SMs)\n", utilization.gpu, activeSms, numSms);
        printf("Memory Used: %.2f MB (%.2f GB) / %.2f MB (%.2f GB)  [%.2f%%]\n",
               memUsedMB, memUsedGB, memTotalMB, memTotalGB, memUsedPercent);

        // --- Per-process GPU memory usage ---
        unsigned int infoCount = 64; // maximum processes we query
        std::vector<nvmlProcessInfo_t> infos(infoCount);
        result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, infos.data());
        if (result == NVML_SUCCESS && infoCount > 0) {
            printf("Per-Process GPU Memory Usage:\n");
            for (unsigned int j = 0; j < infoCount; j++) {
                double procMemMB = (double)infos[j].usedGpuMemory / (1024.0 * 1024.0);
                double procMemGB = procMemMB / 1024.0;
                printf("  PID %d -> %.2f MB (%.2f GB)\n",
                       infos[j].pid, procMemMB, procMemGB);
            }
        } else {
            printf("No compute processes found.\n");
        }

        // --- Per-process GPU utilization (SM + Memory) ---
        unsigned int utilCount = 64;
        std::vector<nvmlProcessUtilizationSample_t> samples(utilCount);
        unsigned long long lastSeenTimeStamp = 0; // start from now
        result = nvmlDeviceGetProcessUtilization(device, samples.data(), &utilCount, lastSeenTimeStamp);
        if (result == NVML_SUCCESS && utilCount > 0) {
            printf("Per-Process GPU Utilization:\n");
            for (unsigned int j = 0; j < utilCount; j++) {
                printf("  PID %d -> GPU Util: %u%%, Mem Util: %u%% (at %llu ms)\n",
                       samples[j].pid,
                       samples[j].smUtil,
                       samples[j].memUtil,
                       samples[j].timeStamp);
            }
        } else {
            printf("Per-process utilization not available (driver may not support it).\n");
        }

        printf("\n");
        sleep(1);
    }

    nvmlShutdown();
    return 0;
}
