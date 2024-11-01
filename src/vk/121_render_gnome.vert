#version 450

#include <libgpu/vulkan/vk/atomic.vk>
#include <libgpu/vulkan/vk/common.vk>

layout (std430, binding = 0) readonly	buffer MinMaxIn	{	float minXYZ[3]; float maxXYZ[3];	};
//layout (binding = 1)							uniform				sampler2DArray	input_image;

// vertex attributes
layout (location = 0)	in			vec3			inPosAttribute;
layout (location = 1)	in			vec2			inUVAttribute;

layout (location = 0)	out			vec2			outUV;

void main()
{
    vec3 pos = inPosAttribute;
    for (int i = 0; i < 3; ++i) {
        rassert(pos[i] >= minXYZ[i] && pos[i] <= maxXYZ[i], 351242131);
        // нормализуем в диапазон [0, 1]
        pos[i] = (pos[i] - minXYZ[i]) / (maxXYZ[i] - minXYZ[i]);
    }

    // получили координаты вершины которую сейчас обрабатывает этот вершинный шейдер
    vec2 screenPos = 1.0-pos.zy;
    float depth = 1.0-pos.x;

    // gl_Position и gl_FragDepth - с помощью них задается позиция вершины в экране + глубина вершины в глубине экрана (в фреймбуфере глубины)
    gl_Position = vec4(toGLPositionRange(screenPos.xy), depth, 1.0);
    // при этом задавать gl_FragDepth не обязательно, т.к. по умолчанию он принимает значение gl_Position.z:
    // gl_FragDepth = gl_Position.z;

    // получили UV-текстурные координаты вершины которую сейчас обрабатывает этот вершинный шейдер
    // и передали их как есть нетронутыми - это аттрибут вершины который должен быть интерполирован внутри треугольника
    // т.е. три таких outUV соберутся вместе для каждого треугольника и взвешенно (барицентрическими координатами)
    // будут интерполированы для каждого фрагмента-пикселя
    // см. там inUV - именно туда будет передана интерполированая UV-текстурная координата внутренности треугольника
    outUV = inUVAttribute;
}
