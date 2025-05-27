#include <libbase/stats.h>
#include <libbase/fast_random.h>
#include <libutils/misc.h>

#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/timer.h>

#include "kernels/vk/kernels.h"
#include "kernels/vk/defines.h"


std::vector<float> evaluateCPU(const std::vector<float> &as, unsigned int n)
{
    std::vector<float> sorted_as;

    std::cout << "______________________CPU_______________________" << std::endl;
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        sorted_as = as;

        timer t;
        std::sort(sorted_as.begin(), sorted_as.end());
        times.push_back(t.elapsed());
    }
    std::cout << "std::sort median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность памяти
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "std::sort median RAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (single read + single write)" << std::endl;

    return sorted_as;
}

std::vector<float> evaluateGPUVulkan(const std::vector<float> &as, unsigned int n,
                        int argc, char **argv)
{
    bool verbose_debug = false;
    std::vector<float> sorted_as;

    std::cout << "______________________Vulkan____________________" << std::endl;

    gpu::Device device = gpu::chooseGPUVulkanDevices(argc, argv);
    gpu::Context context = activateVKContext(device);

    gpu::gpu_mem_32f a_gpu(n);
    gpu::gpu_mem_32f sorted_a_gpu(n);

    avk2::KernelSource kernel_merge_sort(avk2::get105MergeSort());
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        a_gpu.writeN(as.data(), n);

        timer t;

        unsigned int sorted_k = 1;
        while (sorted_k < n) {
            struct {
                unsigned int n;
                unsigned int sorted_k;
            } params = {n, sorted_k};
            if (verbose_debug) std::cout << "n=" << n << " sorted_k=" << sorted_k << std::endl;
            if (verbose_debug) std::cout << "input array: " << stats::vectorToString(a_gpu.readVector()) << std::endl;
            kernel_merge_sort.exec(params, gpu::WorkSize(VK_GROUP_SIZE, n),
                                   a_gpu, sorted_a_gpu);
            if (verbose_debug) std::cout << "output array: " << stats::vectorToString(sorted_a_gpu.readVector()) << std::endl;

            std::swap(a_gpu, sorted_a_gpu);
            sorted_k *= 2;
        }

        times.push_back(t.elapsed());

        sorted_as = a_gpu.readVector();
    }
    std::cout << "merge-sort median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "merge-sort median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (single read + single write)" << std::endl;

    return sorted_as;
}

void mainThrowable(int argc, char **argv)
{
    FastRandom r;

    unsigned int n = 10 * 1000 * 1000;
    std::vector<float> as(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = i + r.nextf();
    }
    std::cout << "Data generated for N=" << n << std::endl;

    auto res_cpu = evaluateCPU(as, n);
    auto res_vulkan = evaluateGPUVulkan(as, n, argc, argv);

    for (size_t i = 0; i < n; ++i) {
        auto a = res_cpu[i];
        auto b = res_vulkan[i];
        rassert(a == b, 433245253674, a, b);
    }
}

int main(int argc, char **argv)
{
    try {
        mainThrowable(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught." << std::endl;
        return 2;
    }

    return 0;
}
