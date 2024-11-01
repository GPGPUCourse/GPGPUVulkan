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

	size_t w=1024, h=768;

	std::vector<point3f> vertices = {
		// координаты (мы сами так решили) - от 0.0 до 1.0, отсчет сверху-слева
		{0.1,			0.1,		0.5}, // сверху-слева
		{0.9,			0.1,		0.5}, // сверху-справа
		{0.5,			0.9,		0.5}, // снизу-посередине
		// TODO давайте добавим еще одну вершину:
		{0.8,			0.8,		0.5}, // снизу-справа
	};

	std::vector<point3u> triangles = {
		{0,				1,		2}, // индексы указывающие на три вершины, на которых покоится данный треугольник
		// TODO давайте добавим еще один треугольник покоящийся в т.ч. на последней вершине:
		{2,				1,		3},
	};

	// Аллоцируем в VRAM
	gpu::gpu_mem_vertices_xyz	vertices_gpu(vertices.size());
	gpu::gpu_mem_faces_indices	triangles_gpu(triangles.size());

	gpu_image8u					output_color_attachment_gpu(w, h, 3);
	gpu_image32i				output_face_id_attachment_gpu(w, h);
	gpu_imageDepth				output_depth_framebuffer_gpu(w, h);

	// Прогружаем входные данные
	vertices_gpu.writeN((gpu::Vertex3D*) vertices.data(), vertices.size());
	triangles_gpu.writeN(triangles.data(), triangles.size());

	// Запускаем кернел (несколько раз и с замером времени выполнения)
	avk2::KernelSource kernel_render_triangle(avk2::get110RenderTriangle());
	std::vector<double> times;
	const int NO_FACE_ID = -1;
	for (int iter = 0; iter < 10; ++iter) {
		timer t;
		kernel_render_triangle.initRender(w, h)
			.geometry(vertices_gpu, triangles_gpu)
			.setDepthAttachment(output_depth_framebuffer_gpu, CLEAR_DEPTH_FRAMEBUFFER_WITH_MAX_VALUE)
			.addAttachment(output_color_attachment_gpu, {0, 0, 0, 0})
			.addAttachment(output_face_id_attachment_gpu, NO_FACE_ID)
			.params((unsigned int) triangles.size())
			.exec();
		times.push_back(t.elapsed());
	}
	std::cout << "face rendering times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

	// Вычисляем достигнутый FPS
	std::cout << "face rendering median FPS: " << 1.0 / stats::median(times) << " FPS" << std::endl;

	// Считываем результат GPU VRAM -> CPU RAM
	auto output_color = output_color_attachment_gpu.read();
	auto output_face_ids = output_face_id_attachment_gpu.read();
	auto output_depth_framebuffer = output_depth_framebuffer_gpu.read();

	// Отображаем результат
	auto window_colors = output_color.show("colors");
	auto window_face_ids = debug_io::randomMapping(output_face_ids, NO_FACE_ID).show("face_ids");
	auto window_depth = debug_io::depthMapping(output_depth_framebuffer, CLEAR_DEPTH_FRAMEBUFFER_WITH_MAX_VALUE).show("depth framebuffer");

	// Ждем пока все окна не будут закрыты или не будет нажата кнопка ESCAPE
	waitAllWindows({window_colors, window_face_ids, window_depth});

	return 0;
}
