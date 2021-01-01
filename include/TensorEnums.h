#pragma once

namespace TSlib
{
	enum class Mode
	{
		GPU,
		CPU,
		Multiply,
		Divide,
		Cube,
		Plane,
		Line
	};

	#ifdef _CUDA
	constexpr Mode default_device = Mode::GPU;
	#else
	constexpr Mode default_device = Mode::CPU;
	#endif
}
