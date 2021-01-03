#include "TensorBones.h"

namespace TSlib
{
	double_t round(double_t x, double_t place)
	{
		return double_t(int(x) * place) / place;
	}
}