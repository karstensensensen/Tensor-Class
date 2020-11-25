#pragma once
#include <cstdint>
#include <initializer_list>

namespace TSlib
{
	struct TSlice
	{
		intmax_t from;
		intmax_t to;


		uint32_t from_max = 0;
		uint32_t to_max = 0;

		TSlice(const intmax_t& from, const intmax_t& to, const uint32_t& from_max = 0, const uint32_t& to_max = 0);

		/*template<typename T>
		TSlice(const std::initializer_list<T>& val);*/

		TSlice(const intmax_t& val);

		TSlice();

		bool contains(intmax_t val) const;

		uint32_t get_from() const;

		uint32_t get_to() const;

		uint32_t width() const;

		bool operator()(intmax_t val, size_t from_max = 0, size_t to_max = 0) const;

		class iterator;

		iterator begin();

		iterator end();
	};

	inline TSlice To(const size_t& pos)
	{
		return TSlice(0, pos);
	}

	inline TSlice From(const size_t& pos)
	{
		return TSlice(pos, -1);
	}

	inline TSlice Center(const size_t& pos)
	{
		return TSlice(pos, -pos - 2);
	}

	static const TSlice All = TSlice(0, -1);
}

