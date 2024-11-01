#version 450

#define EXPLICITLY_DECLARED_ATTACHMENTS 1

#if EXPLICITLY_DECLARED_ATTACHMENTS
	#include <libgpu/vulkan/vk/common.vk>

	// в каждом фрагменте-пикселе в output attachments будут записаны RGB цвета:
	layout (location =	0)				out	float	outColor0;
	layout (location =	1)				out	float	outColor1;
	layout (location =	2)				out	float	outColor2;

	// давайте добавим в output attachments не только цвета, но и номер треугольника покрывшего этот фрагмент-пиксель:
	layout (location =	3)				out	int		outFaceIdx;
#else
	#define NCHANNELS 3
	#define REQUIRE_TEMPLATE_NCHANNELS
	#include <libgpu/vulkan/vk/common.vk>

	// эта строка благодаря макросу LAYOUT_NCHANNELS объявленному в common.vk аллоцирует те же три output attachments для RGB цвета пикселя:
	LAYOUT_NCHANNELS (	0,							outColor);

	layout (location =	0+NCHANNELS)	out	int		outFaceIdx;
#endif

layout (push_constant) uniform PushConstants {
	uint nfaces;
} params;

void main()
{
	int face_id = gl_PrimitiveID;
	// давайте проверим инвариант что ничего не вышло за рамки разумного:
	rassert(face_id >= 0 && face_id < params.nfaces, 675459179);

	vec3 resultColors = vec3(1.0, 0.0, 0.0);

#if EXPLICITLY_DECLARED_ATTACHMENTS
	outColor0 = resultColors.x;
	outColor1 = resultColors.y;
	outColor2 = resultColors.z;
#else
	ASSIGN_NCHANNELS(outColor, resultColors);
#endif

	outFaceIdx = face_id;
}