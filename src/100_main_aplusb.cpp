#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/timer.h>

#include "vk/kernels.h"
#include "vk/defines.h"


int main(int argc, char **argv)
{
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

	unsigned int n = 100 * 1000 * 1000;
	std::vector<unsigned int> as(n, 0);
	std::vector<unsigned int> bs(n, 0);
	for (size_t i = 0; i < n; ++i) {
		as[i] = 3 * (i + 5) + 7;
		bs[i] = 11 * (i + 13) + 17;
	}

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
		kernel_aplusb.exec(n, gpu::WorkSize(VK_GROUP_SIZE, n),
			a_gpu, b_gpu, c_gpu);
		times.push_back(t.elapsed());
	}
	std::cout << "a + b kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

	// Вычисляем достигнутую эффективную пропускную способность видеопамяти
	double memory_size_gb = sizeof(unsigned int) * 3 * n / 1024.0 / 1024.0 / 1024.0;
	std::cout << "a + b kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

	// Считываем результат GPU VRAM -> CPU RAM
	std::vector<unsigned int> cs(n, 0);
	c_gpu.readN(cs.data(), n);

	// Сверяем результат
	for (size_t i = 0; i < n; ++i) {
		rassert(cs[i] == as[i] + bs[i], 321418230421312);
	}

	return 0;
}
