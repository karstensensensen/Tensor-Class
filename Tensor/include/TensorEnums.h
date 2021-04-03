#pragma once

namespace TSlib
{
	enum class Mode
	{
		Multiply,
		Divide,
		Cube,
		Plane,
		Line
	};

	enum class Device
	{
		GPU,
		CPU
	};

	#ifdef _TS_CUDA
	constexpr Device default_device = Device::GPU;
	#else
	constexpr Device default_device = Device::CPU;
	#endif
}
