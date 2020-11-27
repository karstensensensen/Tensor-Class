#pragma once

namespace TSlib
{
	template<typename Ta, typename Tb>
	static bool Equal(const Ta& a, const Tb& b)
	{
		return a == b;
	}

	template<typename Ta, typename Tb>
	static bool NotEqual(const Ta& a, const Tb& b)
	{
		return a != b;
	}

	template<typename Ta, typename Tb>
	static bool LessThan(const Ta& a, const Tb& b)
	{
		return a < b;
	}

	template<typename Ta, typename Tb>
	static bool GreaterThan(const Ta& a, const Tb& b)
	{
		return a > b;
	}

	template<typename Ta, typename Tb>
	static bool LessThanEqual(const Ta& a, const Tb& b)
	{
		return a <= b;
	}

	template<typename Ta, typename Tb>
	static bool GreaterThanEqual(const Ta& a, const Tb& b)
	{
		return a >= b;
	}
}