#include "kernels.h"

#include "generated_kernels/100_aplusb_comp.h"
#include "generated_kernels/110_render_triangle_vert.h"
#include "generated_kernels/111_render_triangle_frag.h"
#include "generated_kernels/120_gnome_min_max_comp.h"
#include "generated_kernels/121_render_gnome_vert.h"
#include "generated_kernels/122_render_gnome_frag.h"
#include "generated_kernels/130_fildelity_fx_cas_sharpening_comp.h"

namespace avk2 {
	const ProgramBinaries& get100AplusB() {
		return vulkan_binaries_100_aplusb_comp;
	}

	std::vector<const ProgramBinaries*> get110RenderTriangle() {
		return {&vulkan_binaries_110_render_triangle_vert, &vulkan_binaries_111_render_triangle_frag};
	}

	const ProgramBinaries& get120GnomeMinMax() {
		return vulkan_binaries_120_gnome_min_max_comp;
	}

	std::vector<const ProgramBinaries*> get121RenderGnome() {
		return {&vulkan_binaries_121_render_gnome_vert, &vulkan_binaries_122_render_gnome_frag};
	}

	const ProgramBinaries& get130FidelityFXSharpening() {
		return vulkan_binaries_130_fildelity_fx_cas_sharpening_comp;
	}
}
