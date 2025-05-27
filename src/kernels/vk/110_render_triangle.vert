#version 450

#include <libgpu/vulkan/vk/common.vk>

// vertex attributes
layout (location = 0)	in			vec3			inPos;

void main()
{
    // получили координаты вершины которую сейчас обрабатывает этот вершинный шейдер
    // координаты - в диапазоне [0, 1], отсчет сверху-слева (мы сами так решили)
    vec3 screenPos = inPos;

    // gl_Position и gl_FragDepth - переменные которые можно задать в вершинном шейдере
    // с помощью них задается позиция вершины в экране + глубина вершины в глубине экрана (в фреймбуфере глубины)
    // дальше hardware shape assembly (непрограммируемый этап) соберет для каждого треугольника эти экранные координаты вершин
    // и будет выполнена hardware rasterization (непрограммируемый этап) которая исполнит фрагментный шейдер во всех пикселях внутри треугольника
    // toGLPositionRange(xy) переводит xy из [0, 1] диапазона в [-1, 1] - см. подробнее про Normalized Device Coordinate - https://github.com/gpuweb/gpuweb/issues/416
    gl_Position = vec4(toGLPositionRange(screenPos.xy), screenPos.z, 1.0);

    // при этом задавать gl_FragDepth не обязательно, т.к. по умолчанию он принимает значение gl_Position.z:
    // gl_FragDepth = gl_Position.z;
}
