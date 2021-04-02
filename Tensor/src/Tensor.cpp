#include "Tensor.h"

namespace TSlib
{
	double round(double x, double place)
	{
		return double(int(x) * place) / place;
	}
}