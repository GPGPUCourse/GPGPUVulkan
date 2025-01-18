cmake_minimum_required(VERSION 3.5)

project(GPGPUVulkan)

set(CMAKE_CXX_STANDARD 17)

option(GPU_CUDA_SUPPORT "CUDA support." ON)

if (NOT DEFINED GLSLC_BIN)
    set (GLSLC_BIN glslc)
    message (STATUS "Vulkan glslc compiler from system PATH will be used")
else ()
    message (STATUS "Vulkan glslc compiler is specified")
endif ()

# GTest позволяет писать удобные unit-test-ы
find_package(GTest REQUIRED)
set (GTEST_LIBRARIES GTest::gtest)
set (GTEST_MAIN_LIBRARIES GTest::gtest_main)
set (GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

add_subdirectory(libs)

set(HEADERS
        src/utils/happly.h
        src/utils/read_ply_with_uv.h
        src/vk/kernels.h
)

set(SOURCES
        src/utils/read_ply_with_uv.cpp
        src/vk/kernels.cpp
)

set(LIBRARIES
        libclew
        libgpu
        libutils
)

# compile_vulkan CMake-функция объявлена в libs/gpu/libgpu/vulkan/CMakeLists.txt:97
# Она считывает все байты из файла src/vk/100_aplusb.comp (т.е. весь исходный код кернела), компилирует с помощью glslc компилятора (в т.ч. выполняя все include)
# Скомпилированный SPIR байткод находится в файле src/vk/generated_kernels/100_aplusb_comp_spirv_vulkan.spir
# SPIR байткод можно проанализировать через рефлексию - /usr/local/bin/spirv-reflect src/vk/generated_kernels/100_aplusb_comp_spirv_vulkan.spir
# Затем получившийся скомпилированный SPIR байткод преобразует в массив байтов в файле src/vk/generated_kernels/100_aplusb_comp_spirv_vulkan.h
# (чтобы не нужно было в runtime читать шейдер/байткод с диска, чтобы был монолитный исполняемый файл)
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
set(VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/100_aplusb.comp                       VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/110_render_triangle.vert              VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/111_render_triangle.frag              VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/120_gnome_min_max.comp                VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/121_render_gnome.vert                 VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/122_render_gnome.frag                 VULKAN_DEFINES)
compile_vulkan(HEADERS src/vk/130_fildelity_fx_cas_sharpening.comp  VULKAN_DEFINES)

add_executable       (100_main_aplusb                       src/100_main_aplusb.cpp                     ${SOURCES} ${HEADERS})
add_executable       (110_main_render_triangle              src/110_main_render_triangle.cpp            ${SOURCES} ${HEADERS})
add_executable       (120_main_render_gnome                 src/120_main_render_gnome.cpp               ${SOURCES} ${HEADERS})
add_executable       (130_main_fidelity_fx_sharpening       src/130_main_fidelity_fx_sharpening.cpp     ${SOURCES} ${HEADERS})

target_link_libraries(100_main_aplusb                       ${LIBRARIES})
target_link_libraries(110_main_render_triangle              ${LIBRARIES})
target_link_libraries(120_main_render_gnome                 ${LIBRARIES})
target_link_libraries(130_main_fidelity_fx_sharpening       ${LIBRARIES})
