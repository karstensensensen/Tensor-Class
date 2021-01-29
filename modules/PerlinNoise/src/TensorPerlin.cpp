#include "TensorPerlin.h"
#include <cmath>

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

using TSlib::Tensor;
typedef std::mt19937_64 rng_type;

double ipow(double val, int exponent)
{
	double product = 1;

	for (int i = 0; i < exponent; i++)
	{
		product *= val;
	}

	return product;
}

double bicubic_interpolate_2d(Tensor<double>& a, double x, double y, int xc, int yc)
{
	double interpolation = 0;

	//offset coords to corners
	xc -= 2;
	yc -= 2;

	int xcoef[4];
	int ycoef[4];

	for (int i = 0; i < 4; i++)
	{
		xcoef[i] = (xc + i) * ((xc + i) >= 0) + (a.Shape()[1] - 1) * ((xc + i) >= int(a.Shape()[1]));
		ycoef[i] = (yc + i) * ((yc + i) >= 0) + (a.Shape()[0] - 1) * ((yc + i) >= int(a.Shape()[0]));
	}

	Tensor<double> coefficients({ 4, 4 }, [&a, &xcoef, &ycoef](const std::vector<size_t>& coords) {
			return a(ycoef[coords[0]], xcoef[coords[1]]);
		});

	Tensor<double> x_vector({ 4 }, [x](const size_t& index) {return ipow(x, index); });
	Tensor<double> y_vector({ 1, 4 }, [y](const size_t& index) {return ipow(y, index); });
	
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			interpolation += a(ycoef[i], xcoef[j]) * ipow(x, i) * ipow(y, j);
		}
	}

	/*double F11 = a(ycoef[0], xcoef[0]);
	double F21 = a(ycoef[0], xcoef[0]) + a(ycoef[1], xcoef[0]) + a(ycoef[2], xcoef[0]) + a(ycoef[3], xcoef[0]);
	double F12 = a(ycoef[1], xcoef[0]) + a(ycoef[1], xcoef[1]) + a(ycoef[1], xcoef[2]) + a(ycoef[1], xcoef[3]);
	double F22 = 0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			F22 += a(ycoef[i], xcoef[j]) * i * j;
		}
	}*/

	/*double Px = 0;

	for (int i = 1; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			x += 5;
		}
	}

	double Py = 0;

	double Pxy = 0;

	for (int i = 1; i < 3; i++)
	{
		for (int j = 1; j < 3; j++)
		{

			Pxy += a(ycoef[i], xcoef[j]) * i * ipow(x, i - 1) * j * ipow(y, j - 1);
		}
	}*/

	/*double Pxy = coefficients.Compute([&x_vector](double& elem, const double& coefficient, const std::vector<size_t>& coords) {
		elem += coefficient * x_vector[coords[0]];
		
		}, 1).Compute([&y_vector](double& elem, const double& coefficient, const std::vector<size_t>& coords) {
		
		elem += coefficient * y_vector[coords[1]];
			
		}, 0)[0];*/

	return interpolation;
}

Tensor<double> perlin2d(size_t xn, size_t yn, size_t sps)
{
	rng_type rng(std::random_device{}());
	std::uniform_real_distribution<double> urdist(-1., 1.);

	size_t m = yn * sps;
	size_t n = xn * sps;

	Tensor<double> UV({ yn + 1, xn + 1 });

	double hx = double(xn) / (n), hy = double(yn) / (m);

	Tensor<double> Z({ m, n });

	UV.Compute([&rng, &urdist](double& elem) {elem = urdist(rng); });

	Z.Compute([&UV, hx, hy](double& Z, const std::vector<size_t>& coords, const size_t& indx){
		try
		{
			int xc = int(hx * (double)coords[1]);
			int yc = int(hy * (double)coords[0]);

			/*if (!fmod(int(hx * (double)coords[1]), 1) == int(hx * (double)coords[1])) xc -= 1;
			if (!fmod(int(hy * (double)coords[0]), 1) == int(hy * (double)coords[0])) yc -= 1;*/

			double xr = hx * (double)coords[1] - xc;
			double yr = hy * (double)coords[0] - yc;

			//// set distance to each corner
			//double S11[] = { -xr,		-yr };
			//double S21[] = { -xr,		1 - yr };
			//double S22[] = { 1 - xr,	1 - yr };
			//double S12[] = { 1 - xr,	-yr };

			//// filter
			//double F22v[] = { 3 * xr * 3 * xr - 2 * xr * 2 * xr * 2 * xr, 3 * yr * 3 * yr - 2 * yr * 2 * yr * 2 * yr };
			//double F21v[] = { F22v[1],		F22v[0] };
			//double F12v[] = { 1 - F22v[0],	1 - F22v[1] };
			//double F11v[] = { 1 - F21v[0],	1 - F21v[1] };

			//double F22 = F22v[0] * F22v[1];
			//double F21 = F21v[0] * F21v[1];
			//double F12 = F12v[0] * F12v[1];
			//double F11 = F11v[0] * F11v[1];

			////calculate corner values
			//double Q11 = S11[0] * UV(0, yc, xc) + S11[1] * UV(1, yc, xc);
			//double Q21 = S21[0] * UV(0, yc + 1, xc) + S21[1] * UV(1, yc + 1, xc);
			//double Q22 = S22[0] * UV(0, yc + 1, xc + 1) + S22[1] * UV(1, yc + 1, xc + 1);
			//double Q12 = S12[0] * UV(0, yc, xc + 1) + S12[1] * UV(1, yc, xc + 1);

			//interpolate z value
			Z = bicubic_interpolate_2d(UV, xr, yr, xc, yc );
		}
		catch (std::exception& e)
		{
			std::cout << e.what();
		}
	});
	
	return Z;
}

int main()
{
	try
	{
		TSlib::Tensor<double> perlin = perlin2d(16, 16, 128);

		std::ofstream out("D:/Dev/python/Projects/DisplayNoise/Noise.csv");

		out << perlin.printable();
		out.close();

		std::cout << "Saved Noise\n";

		std::cin.get();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
		std::cin.get();
	}

	/*Tensor<int> matrix1({ 2 });
	matrix1 = { 5, 10 };

	Tensor<int> matrix2({ 2, 3 });
	matrix2 = { 5, 3, 5, 2, 9, 7 };

	Tensor<int> matrix3({ 1, 3 });
	matrix3 = { 2, 2, 2 };

	std::cout << matrix2.Compute([&matrix1](int& elem, const int& old_elem, const std::vector<size_t>& coords) {elem += old_elem * matrix1[coords[0]];
	std::cout << coords[1] << '\n'; }, 1).Compute([&matrix3](int& elem, const int& old_elem, const std::vector<size_t>& coords) {
		elem += old_elem * matrix3[coords[1]];
		}, 0);*/
}
