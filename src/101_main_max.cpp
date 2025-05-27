#include <libbase/stats.h>
#include <libbase/fast_random.h>
#include <libutils/misc.h>

#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/timer.h>

#include "kernels/vk/kernels.h"
#include "kernels/vk/defines.h"


float evaluateCPU(const std::vector<float> &as, unsigned int n)
{
    float max;

    std::cout << "______________________CPU_______________________" << std::endl;
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        max = -FLT_MAX;
        for (size_t i = 0; i < n; ++i) {
            max = std::max(max, as[i]);
        }

        times.push_back(t.elapsed());
    }
    std::cout << "max median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность памяти
    double memory_size_gb = sizeof(unsigned int) * 1 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "max median RAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    return max;
}

float evaluateGPUVulkan(const std::vector<float> &as, unsigned int n,
                        int argc, char **argv)
{
    float max;

    std::cout << "______________________Vulkan____________________" << std::endl;

    gpu::Device device = gpu::chooseGPUVulkanDevices(argc, argv);
    gpu::Context context = activateVKContext(device);

    gpu::gpu_mem_32f a_gpu(n);
    a_gpu.writeN(as.data(), n);

    unsigned int reduction_ratio = VK_GROUP_SIZE;
    gpu::gpu_mem_32f buffer_reduced(div_ceil(n, reduction_ratio));
    gpu::gpu_mem_32f buffer_reduced2(div_ceil(n, reduction_ratio));

    avk2::KernelSource kernel_reduce_max(avk2::get101ReduceMax());
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        unsigned int current_n = n;
        rassert(n > 1, 23932421341231);
        while (current_n > 1) {
            auto &from_gpu = (current_n == n) ? a_gpu : buffer_reduced;
            auto &to_gpu = buffer_reduced2;

            kernel_reduce_max.exec( current_n, gpu::WorkSize(VK_GROUP_SIZE, current_n),
                                   from_gpu, to_gpu);

            current_n = div_ceil(current_n, reduction_ratio);
            std::swap(buffer_reduced, buffer_reduced2);
        }

        float result = buffer_reduced.readVector(1)[0];
        if (iter == 0) {
            max = result;
        } else {
            rassert(max == result, 3251324213);
        }

        times.push_back(t.elapsed());
    }
    std::cout << "max median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

    double memory_size_gb = sizeof(unsigned int) * 1 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "max median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    return max;
}

void mainThrowable(int argc, char **argv)
{
    FastRandom r;

    unsigned int n = 100 * 1000 * 1000;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = i + r.nextf();
        bs[i] = i + r.nextf();
    }
    std::cout << "Data generated for N=" << n << std::endl;

    auto res_cpu = evaluateCPU(as, n);
    auto res_vulkan = evaluateGPUVulkan(as, n, argc, argv);

    rassert(res_cpu == res_vulkan, 353245253674, res_cpu, res_vulkan);
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
