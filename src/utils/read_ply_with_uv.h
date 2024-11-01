#pragma once

#include <tuple>
#include <vector>
#include <libbase/point.h>


std::tuple<std::vector<point3f>, std::vector<point2f>, std::vector<point3u>> loadPlyWithUV(const std::string &path);
