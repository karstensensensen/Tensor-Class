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

		template<typename T>
		TSlice(const std::initializer_list<T>& val);

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
}

