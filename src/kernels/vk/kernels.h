#pragma once

#include <libgpu/vulkan/engine.h>

namespace avk2 {
	const ProgramBinaries&				get100AplusB();

    const ProgramBinaries&				get101ReduceMax();

    const ProgramBinaries&				get105MergeSort();

	std::vector<const ProgramBinaries*>	get110RenderTriangle();

	const ProgramBinaries&				get120GnomeMinMax();
	std::vector<const ProgramBinaries*>	get121RenderGnome();

	const ProgramBinaries&				get130FidelityFXSharpening();
}
