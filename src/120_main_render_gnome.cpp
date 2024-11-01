#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libgpu/vulkan/tests/test_utils.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/timer.h>

#include "libimages/debug_io.h"
#include "vk/kernels.h"
#include "vk/defines.h"

#include "utils/read_ply_with_uv.h"


int main(int argc, char **argv)
{
	gpu::Device device = gpu::chooseGPUVulkanDevices(argc, argv);
	gpu::Context context = activateVKContext(device);

	size_t w=1024, h=768;

	// Считываем jpg текстуру
	std::string texture_path("data/gnome/gnome.jpg");
	image8u texture(texture_path);
	if (texture.isNull()) {
		std::cerr << "can't load texture image: " << texture_path << std::endl;
		std::cerr << "ensure that working directory is properly configured - it should point to the project directory root" << std::endl;
		rassert(false, 2141241245213);
	}
	std::cout << texture.width() << " x " << texture.height() << " texture loaded" << std::endl;

	// Считываем ply файл с геометрией и UV координатами
	auto [vertices_xyz, vertices_uv, faces] = loadPlyWithUV("data/gnome/gnome.ply");

	rassert(vertices_xyz.size() == vertices_uv.size(), 213412412);
	std::vector<gpu::Vertex3DUV> vertices_with_uv(vertices_xyz.size());
	for (size_t vi = 0; vi < vertices_xyz.size(); ++vi) {
		vertices_with_uv[vi].init(vertices_xyz[vi].x, vertices_xyz[vi].y, vertices_xyz[vi].z,
								  vertices_uv[vi].x, vertices_uv[vi].y);
	}

	// Аллоцируем в VRAM
	gpu::gpu_mem_vertices_xyz_uv	vertices_gpu(vertices_with_uv.size());
	gpu::gpu_mem_faces_indices		faces_gpu(faces.size());

	rassert(texture.channels() == 3, 123412451325423);
	gpu::gpu_mem_32f				min_max_values_gpu(6);
	gpu_image8u						texture_gpu(texture.width(), texture.height(), 3);

	gpu_image8u						output_color_attachment_gpu(w, h, 3);
	gpu_image32i					output_face_id_attachment_gpu(w, h);
	gpu_imageDepth					output_depth_framebuffer_gpu(w, h);

	// Прогружаем входные данные
	vertices_gpu.writeN(vertices_with_uv.data(), vertices_with_uv.size());
	faces_gpu.writeN(faces.data(), faces.size());
	texture_gpu.write(texture);

	// Запускаем кернел (несколько раз и с замером времени выполнения)
	avk2::KernelSource kernel_gnome_min_max(avk2::get120GnomeMinMax());
	avk2::KernelSource kernel_render_gnome(avk2::get121RenderGnome());
	std::vector<double> times;
	const int NO_FACE_ID = -1;
	for (int iter = 0; iter < 10; ++iter) {
		timer t;

		avk2::InstanceContext::renderDocStartCapture("render gnome");

		const float FLT_MAX = std::numeric_limits<float>::max();
		float initial_min_max_values[] = {FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
		min_max_values_gpu.writeN(initial_min_max_values, 6);

		unsigned int nvertices = vertices_with_uv.size();
		kernel_gnome_min_max.exec(nvertices, gpu::WorkSize(VK_GROUP_SIZE, nvertices),
			vertices_gpu, min_max_values_gpu);

		kernel_render_gnome.initRender(w, h)
			.geometry(vertices_gpu, faces_gpu)
			.setDepthAttachment(output_depth_framebuffer_gpu, CLEAR_DEPTH_FRAMEBUFFER_WITH_MAX_VALUE)
			.addAttachment(output_color_attachment_gpu, {0, 0, 0, 0})
			.addAttachment(output_face_id_attachment_gpu, NO_FACE_ID)
			.params((unsigned int) faces.size())
			.exec(min_max_values_gpu, texture_gpu);

		avk2::InstanceContext::renderDocEndCapture();

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
