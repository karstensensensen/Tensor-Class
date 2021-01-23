#include "TensorPerlin.h"


TSlib::Tools::PerlinNoise::PerlinNoise()
{
	rng.seed(std::random_device{}());

	std::iota(permutation, permutation + 256, 0);

	std::shuffle(permutation, permutation + 256, rng);

	for (int i = 256; i < 512; i++)
	{
		permutation[i] = permutation[i - 256];
	}
}

TSlib::Tools::PerlinNoise::PerlinNoise(unsigned int seed)
{
	rng.seed(seed);

	std::iota(permutation, permutation + 256, 0);

	std::shuffle(permutation, permutation + 256, rng);

	for (int i = 256; i < 512; i++)
	{
		permutation[i] = permutation[i - 256];
	}
}

double TSlib::Tools::PerlinNoise::fade(double t)
{
	return t * t * t * (t * (t * 6 - 15) + 10);
}

double TSlib::Tools::PerlinNoise::lerp(double t, double a, double b)
{
	return a + t * (b - a);
}

double TSlib::Tools::PerlinNoise::grad(int hash, double x, double y, double z)
{
	int h = hash & 15;

	double u = h < 8 ? x : y, v = h > 4 ? y : h == 12 || h == 14 ? x : z;

	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

double TSlib::Tools::PerlinNoise::noise(double x, double y, double z)
{
	int X = (int)floor(x) & 255;
	int Y = (int)floor(y) & 255;
	int Z = (int)floor(z) & 255;

	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	double fade_x = fade(x);
	double fade_y = fade(y);
	double fade_z = fade(z);

	int A = permutation[X] + Y;
	int AA = permutation[A] + Z;
	int AB = permutation[A + 1] + Z;
	int B = permutation[X + 1] + Y;
	int BA = permutation[B] + Z;
	int BB = permutation[B + 1] + Z;

	double res = lerp(fade_z, lerp(fade_y, lerp(fade_x, grad(permutation[AA], x, y, z), grad(permutation[BA], x - 1, y, z)), lerp(fade_x, grad(permutation[AB], x, y - 1, z), grad(permutation[BB], x - 1, y - 1, z))), lerp(fade_y, lerp(fade_x, grad(permutation[AA + 1], x, y, z - 1), grad(permutation[BA + 1], x - 1, y, z - 1)), lerp(fade_x, grad(permutation[AB + 1], x, y - 1, z - 1), grad(permutation[BB + 1], x - 1, y - 1, z - 1))));
	return (res + 1.0) / 2.0;
}

int main()
{
	TSlib::Tensor<double> perlin = TSlib::Tools::Perlin2D<double>(1024, 1024, 0.25);

	std::ofstream out("D:/Dev/python/Projects/DisplayNoise/Noise.csv");

	out << perlin.printable();
	out.close();

	std::cout << "Saved Noise\n";

	std::cin.get();
}
