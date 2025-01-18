#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/timer.h>

#include "libimages/debug_io.h"
#include "vk/kernels.h"
#include "vk/defines.h"


int main(int argc, char **argv)
{
	gpu::Device device = gpu::chooseGPUVulkanDevices(argc, argv);
	gpu::Context context = activateVKContext(device);

	// Считываем jpg текстуру
	std::string texture_path("data/images/wood.jpg");
	image8u texture(texture_path);
	if (texture.isNull()) {
		std::cerr << "can't load texture image: " << texture_path << std::endl;
		std::cerr << "ensure that working directory is properly configured - it should point to the project directory root" << std::endl;
		rassert(false, 4365345234423);
	}
	rassert(texture.channels() == 3, 43563245423);
	std::cout << texture.width() << " x " << texture.height() << " texture loaded" << std::endl;

	int w=texture.width(), h=texture.height();
	int visualization_upscale_ratio = 1;

	// Аллоцируем в VRAM
	gpu_image8u						texture_gpu(texture.width(), texture.height(), 3);
	gpu_image8u						filtered_texture_gpu(texture.width(), texture.height(), 3);

	// Прогружаем входные данные
	texture_gpu.write(texture);

	// Создаем окна для отображения результата
	ImageWindow window_input("CImg input");
	ImageWindow window_output("CImg result");

	// Запускаем кернел (несколько раз и с замером времени выполнения)
	avk2::KernelSource kernel_fidelity_fx_sharpening(avk2::get130FidelityFXSharpening());
	std::vector<double> times;
	timer global_t;
	while (true) {
		timer t;

		int mouseX = -1; // двигаем автоматически вертикальную линию если мышка не была нажата
		float time = global_t.elapsed();
		float casLevel = 1.0;

		if (window_output.getMouseClick() & 1) { // если левая кнопка мыши зажата -  and the CAS level with mouse
			if (window_output.getMouseX() != -1) {
				mouseX = window_output.getMouseX() / visualization_upscale_ratio; // курсор по горизонтали контролирует вертикальную линию
			}
			if (window_output.getMouseY() != -1) {
				casLevel = window_output.getMouseY() * 1.0f / (visualization_upscale_ratio * h); // курсор по вертикали контролирует уровень CAS шарпенинга (чем мышка ниже - тем сильнее)
			}
		}

		struct {
			int     width;
			int     height;
			int     mouseX;
			float   iTime;
			float   iCASLevel;
		} params = {w, h, mouseX, time, casLevel};

		kernel_fidelity_fx_sharpening.exec(params, gpu::WorkSize(VK_GROUP_SIZE_X, VK_GROUP_SIZE_Y, w, h),
			texture_gpu, filtered_texture_gpu);

		// Сохраняем первый кадр картинкой
		if (times.empty()) {
			filtered_texture_gpu.read().savePNG("data/fidelity_fx_sharpening.png");
		}

		times.push_back(t.elapsed());

		// Отображаем результат
		window_input.display(debug_io::upscaleNearestNeighbor(texture, visualization_upscale_ratio));
		window_output.display(debug_io::upscaleNearestNeighbor(filtered_texture_gpu.read(), visualization_upscale_ratio));

		auto windows = {window_input, window_output};
		if (isAnyWindowClosed(windows) || isEscapePressed(windows)) {
			break;
		}
	}
	std::cout << "texture AMD Fidelity FX CAS sharpening times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

	// Вычисляем достигнутый FPS
	std::cout << "rendering median FPS: " << 1.0 / stats::median(times) << " FPS" << std::endl;

	return 0;
}
