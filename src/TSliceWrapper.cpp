#include "TSliceWrapper.h"
#include "TensorExceptions.h"

#ifdef TENSOR_PROFILING
#include <Profiler.h>
#else
#define MEASURE()
#endif

namespace TSlib
{
	TSlice::TSlice(const intmax_t& from, const intmax_t& to, const uint32_t& from_max, const uint32_t& to_max)
		:from(from), to(to), from_max(from_max), to_max(to_max)
	{
		MEASURE();
		#ifdef _DEBUG

		if (from < 0 && to < 0 && to >= from)
		{
			throw BadValue("negative from value must be larger than negative to value", ExceptValue<intmax_t>("from", from), ExceptValue<intmax_t>("to", to));
		}
		else if (from > 0 && to > 0 && to <= from)
		{
			throw BadValue("from value must be less than to value", ExceptValue<intmax_t>("from", from), ExceptValue<intmax_t>("to", to));
		}
		#endif
	}

	TSlice::TSlice(const intmax_t& val)
		: from(val), to(val + 1), from_max(0), to_max(0)
	{
		MEASURE();
	}

	/*template<typename T>
	TSlice::TSlice(const std::initializer_list<T>& val)
		: from(0), to(0), from_max(0), to_max(0)
	{
		MEASURE();

		from = *val.begin();
		to = *(val.begin() + 1);

		#ifdef _DEBUG

		if (from < 0 && to < 0 && to <= from)
		{
			throw BadValue("negative from value must be more than negative to value", ExceptValue<intmax_t>{ "from:", from }, ExceptValue<intmax_t>{ "to:", to });
		}
		else if (to <= from)
		{
			throw BadValue("from value must be less than to value", ExceptValue<intmax_t>{ "from:", from }, ExceptValue<intmax_t>{ "to:", to });
		}
		#endif
	}*/

	TSlice::TSlice()
		: from(NULL), to(NULL), from_max(NULL), to_max(NULL)
	{
		MEASURE();
	}

	bool TSlice::contains(intmax_t val) const
	{
		MEASURE();
		return (get_from() <= val &&
			get_to() >= val);
	}

	uint32_t TSlice::get_from() const
	{
		MEASURE();
		return (uint32_t)std::abs(int((from < 0) * from_max + from));
	}

	uint32_t TSlice::get_to() const
	{
		MEASURE();
		return (uint32_t)std::abs(int((to < 0) * (to_max + 1) + to));
	}

	uint32_t TSlice::width() const
	{
		MEASURE();
		return get_to() - get_from();
	}

	bool TSlice::operator()(intmax_t val, size_t from_max, size_t to_max) const
	{
		MEASURE();
		return contains(val);
	}

	class TSlice::iterator
	{
		uint32_t num;
		TSlice& slice;

	public:
		iterator(uint32_t start, TSlice& slice)
			: num(start), slice(slice)
		{
			MEASURE();
		}

		iterator& operator++()
		{
			num++;
			return *this;
		}

		iterator operator++(int)
		{
			iterator retval = *this;
			++(*this);
			return retval;
		}

		bool operator==(const iterator& other) const
		{
			return num == other.num;
		}

		bool operator!=(const iterator& other) const
		{
			return num != other.num;
		}

		uint32_t operator*()
		{
			return num;
		}

		using diffrence_type = intmax_t;
		using value_type = intmax_t;
		using pointer = const intmax_t*;
		using refrence = const intmax_t&;
		using iterator_category = std::forward_iterator_tag;
	};

	TSlice::iterator TSlice::begin()
	{
		MEASURE();
		return { get_from(), *this };
	}

	TSlice::iterator TSlice::end()
	{
		MEASURE();
		return { get_to() + 1, *this };
	}
}