#include "read_ply_with_uv.h"

#include "happly.h"

#include <iostream>

#include "libbase/runtime_assert.h"


std::tuple<std::vector<point3f>, std::vector<point2f>, std::vector<point3u>> loadPlyWithUV(const std::string &path) {
	std::vector<point3f> vertices_xyz;
	std::vector<point2f> vertices_uv;
	std::vector<point3u> faces;

	// ensure that working directory is properly configured - it should point to the project directory root
	happly::PLYData plyIn(path);

	std::vector<std::array<double, 3>> verticesXYZ = plyIn.getVertexPositions();
	std::vector<double> verticesU = plyIn.getElement("vertex").getProperty<double>("s");
	std::vector<double> verticesV = plyIn.getElement("vertex").getProperty<double>("t");
	std::vector<std::vector<size_t>> facesIndices = plyIn.getFaceIndices();

	for (auto xyz: verticesXYZ) {
		vertices_xyz.push_back({xyz[0], xyz[1], xyz[2]});
	}
	rassert(verticesU.size() == verticesV.size(), 324142213411);
	for (size_t i = 0; i < verticesU.size(); ++i) {
		vertices_uv.push_back(point2f(verticesU[i], verticesV[i]));
	}
	for (auto face: facesIndices) {
		rassert(face.size() == 3, 21312421412);
		faces.push_back(point3u(face[0], face[1], face[2]));
	}

	rassert(vertices_xyz.size() == vertices_uv.size(), 3125315213421);
	std::cout << vertices_xyz.size() << " vertices, " << faces.size() << " faces" << std::endl;
	return {vertices_xyz, vertices_uv, faces};
}
