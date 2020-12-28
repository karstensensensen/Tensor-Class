#pragma once

namespace TSlib
{
	template<typename Ta, typename Tb>
	inline static bool Equal(const Ta& a, const Tb& b)
	{
		return a == b;
	}

	template<typename Ta, typename Tb>
	inline static bool NotEqual(const Ta& a, const Tb& b)
	{
		return a != b;
	}

	template<typename Ta, typename Tb>
	inline static bool LessThan(const Ta& a, const Tb& b)
	{
		return a < b;
	}

	template<typename Ta, typename Tb>
	inline static bool GreaterThan(const Ta& a, const Tb& b)
	{
		return a > b;
	}

	template<typename Ta, typename Tb>
	inline static bool LessThanEqual(const Ta& a, const Tb& b)
	{
		return a <= b;
	}

	template<typename Ta, typename Tb>
	inline static bool GreaterThanEqual(const Ta& a, const Tb& b)
	{
		return a >= b;
	}


}