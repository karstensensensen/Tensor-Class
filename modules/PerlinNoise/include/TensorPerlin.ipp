#pragma once
#include "TensorPerlin.h"
namespace TSlib {namespace Tools
{

	template<typename T, Mode device>
	Tensor<T, device> Perlin1D(unsigned int x, double scale)
	{
		PerlinNoise noise;
		Tensor<T, device> perlin_texture({ x });

		perlin_texture.Compute([&noise, scale](T& elem, const std::vector<size_t>& coords)
			{
				elem = noise.noise(double(coords[0]) * scale);
			}
		);

		return perlin_texture;
	}

	template<typename T, Mode device>
	Tensor<T, device> Perlin2D(unsigned int x, unsigned int y, double scale)
	{
		PerlinNoise noise;
		Tensor<T, device> perlin_texture({ x, y });

		perlin_texture.Compute([&noise, scale](T& elem, const std::vector<size_t>& coords)
			{
				elem = noise.noise(double(coords[0]) * scale, double(coords[1]) * scale);
			}
		);

		return perlin_texture;
	}

}}