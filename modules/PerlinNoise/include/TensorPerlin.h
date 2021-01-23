#pragma once
#include <Tensor.h>
#include <random>

namespace TSlib { namespace Tools
{
	// source: https://github.com/sol-prog/Perlin_Noise.git
	class PerlinNoise
	{
		typedef std::mt19937 rng_type;
		rng_type rng;
		int permutation[512];

	public:
		PerlinNoise();

		PerlinNoise(unsigned int seed);

		double noise(double x, double y = 0, double z = 0);

	private:
		double fade(double t);
		double lerp(double t, double a, double b);
		double grad(int hash, double x, double y, double z);

	};

	template<typename T, Mode device = Mode::CPU>
	Tensor<T, device> Perlin1D(unsigned int x, double scale);

	template<typename T, Mode device = Mode::CPU>
	Tensor<T, device> Perlin1D(double x, double scale, unsigned int seed);

	template<typename T, Mode device = Mode::CPU>
	Tensor<T, device> Perlin2D(unsigned int x, unsigned int y, double scale);

	template<typename T, Mode device = Mode::CPU>
	Tensor<T, device> Perlin2D(double x, double y, double scale, unsigned int seed);

	template<typename T, Mode device = Mode::CPU>
	Tensor<T, device> Perlin3D(double x, double y, double z, double scale);

	template<typename T, Mode device = Mode::CPU>
	Tensor<T, device> Perlin3D(double x, double y, double z, double scale, unsigned int seed);

}}

#include "TensorPerlin.ipp"