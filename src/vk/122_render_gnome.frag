#version 450

#define EXPLICITLY_DECLARED_ATTACHMENTS 1

#define NCHANNELS 3
#define REQUIRE_TEMPLATE_NCHANNELS
#include <libgpu/vulkan/vk/common.vk>

layout (std430, binding = 0) readonly	buffer MinMaxIn	{	float minValue; float maxValue;	};
layout (binding = 1)							uniform				sampler2DArray	input_image;

// эта строка благодаря макросу LAYOUT_NCHANNELS объявленному в common.vk аллоцирует те же три output attachments для RGB цвета пикселя:
LAYOUT_NCHANNELS (	0,							outColor);
layout (location =	0+NCHANNELS)	out	int		outFaceIdx;

// interpolated inside the face attributes (from vertex attributes)
layout (location = 0)				in	vec2	inUV;

layout (push_constant) uniform PushConstants {
	uint nfaces;
} params;

void main()
{
	int face_id = gl_PrimitiveID;
	// давайте проверим инвариант что ничего не вышло за рамки разумного:
	rassert(face_id >= 0 && face_id < params.nfaces, 4252134313);

	float res_channels[NCHANNELS];
	for (int c = 0; c < NCHANNELS; ++c) {
		vec2 uv = inUV;
		uv.y = 1.0 - uv.y;
		res_channels[c] = texture(input_image, vec3(uv, c)).r;
	}

	ASSIGN_NCHANNELS(outColor, res_channels);
	outFaceIdx = face_id;
}
