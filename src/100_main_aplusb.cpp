#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/timer.h>

#include "kernels/vk/kernels.h"


std::vector<unsigned int> evaluateCPU(const std::vector<unsigned int> &as, const std::vector<unsigned int> &bs, unsigned int n)
{
    std::cout << "______________________CPU_______________________" << std::endl;
    std::vector<unsigned int> cs(n);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        for (size_t i = 0; i < n; ++i) {
            cs[i] = as[i] + bs[i];
        }

        times.push_back(t.elapsed());
    }
    std::cout << "a + b median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность памяти
    double memory_size_gb = sizeof(unsigned int) * 3 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "a + b median RAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    return cs;
}

std::vector<unsigned int> evaluateCPUOpenMP(const std::vector<unsigned int> &as, const std::vector<unsigned int> &bs, unsigned int n)
{
    std::cout << "________________CPU (multi-core, OpenMP)________" << std::endl;
    std::vector<unsigned int> cs(n);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        #pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            cs[i] = as[i] + bs[i];
        }

        times.push_back(t.elapsed());
    }
    std::cout << "a + b median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность памяти
    double memory_size_gb = sizeof(unsigned int) * 3 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "a + b median RAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    return cs;
}

std::vector<unsigned int> evaluateGPUVulkan(const std::vector<unsigned int> &as, const std::vector<unsigned int> &bs, unsigned int n,
                                            int argc, char **argv)
{
    std::cout << "______________________Vulkan____________________" << std::endl;

	// chooseGPUVkDevices:
	// - Если не доступо ни одного устройства - кинет ошибку
	// - Если доступно ровно одно устройство - вернет это устройство
	// - Если доступно N>1 устройства:
	//   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
	//   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
	gpu::Device device = gpu::chooseGPUVulkanDevices(argc, argv);

	// Если получили ошибку:
	// Vulkan debug callback triggered with libVkLayer_khronos_validation.so: cannot open shared object file: No such file or directory
	// То перезапустите добавив env переменную LD_LIBRARY_PATH=/usr/local/lib или выключите валидационные слои через
	// Альтернативно - просто скопируйте .so библиотеку в системную папку:
	//  sudo cp /usr/local/lib/libVkLayer_khronos_validation.so /usr/lib/
	gpu::Context context = activateVKContext(device);

	// Аллоцируем буферы в VRAM
	gpu::gpu_mem_32u a_gpu(n), b_gpu(n), c_gpu(n);

	// Прогружаем входные данные
	a_gpu.writeN(as.data(), n);
	b_gpu.writeN(bs.data(), n);

	// Запускаем кернел (несколько раз и с замером времени выполнения)
	avk2::KernelSource kernel_aplusb(avk2::get100AplusB());
	std::vector<double> times;
	for (int iter = 0; iter < 10; ++iter) {
		timer t;
		kernel_aplusb.exec(n, gpu::WorkSize(256, n),
			a_gpu, b_gpu, c_gpu);
		times.push_back(t.elapsed());
	}
    std::cout << "a + b median time: " << stats::median(times) << " sec (+-" << stats::standardDeviation(times) << ")" << std::endl;

	// Вычисляем достигнутую эффективную пропускную способность видеопамяти
	double memory_size_gb = sizeof(unsigned int) * 3 * n / 1024.0 / 1024.0 / 1024.0;
	std::cout << "a + b median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

	// Считываем результат GPU VRAM -> CPU RAM
    auto cs = c_gpu.readVector();

    return cs;
}

void checkResults(const std::vector<unsigned int> &cs_reference, const std::vector<unsigned int> &cs)
{
    unsigned int n = cs.size();
    rassert(n == cs_reference.size(), 435123512351);

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cs_reference[i] == cs[i], 321418230421312, cs_reference[i], cs[i]);
    }
}

void mainThrowable(int argc, char **argv)
{
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> bs(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }
    std::cout << "Data generated for N=" << n << std::endl;

    auto res_cpu = evaluateCPU(as, bs, n);

    auto res_omp = evaluateCPUOpenMP(as, bs, n);
    checkResults(res_cpu, res_omp);

    auto res_vulkan = evaluateGPUVulkan(as, bs, n, argc, argv);
    checkResults(res_cpu, res_vulkan);
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
