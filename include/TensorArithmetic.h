#pragma once
#include "TensorBones.h"

namespace TSlib
{
	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::add(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG

		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&result, &other](const T& val, const size_t& index) {result[index] = val + other[index]; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::add(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) + other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::add(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val + other; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::subtract(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val - other[index]; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::subtract(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) - other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::subtract(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val + other; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::multiply(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val * other[index]; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::multiply(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) * other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::multiply(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val + other; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::divide(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val / other[index]; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::divide(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) / other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::divide(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val / other; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::modulous(const Tensor<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val % other[index]; });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::modulous(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result = *this;

		for (size_t i = 0; i < other.size(); i++)
		{
			result[other.map_index(i)] = At(other.map_index(i)) % other[i];
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::modulous(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val % other; });

		return result;
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::additionAsgmt(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Compute([&](const T& val, const size_t& index) {val += other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::additionAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) += other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::additionAsgmt(const OT& other)
	{
		MEASURE();

		Compute([&](const T& val) {val += other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::subtractionAsgmt(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Compute([&](const T& val, const size_t& index) {val -= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::subtractionAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) -= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::subtractionAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](const T& val, const size_t& index) {val -= other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::multiplicationAsgmt(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Compute([&](const T& val, const size_t& index) {val *= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::multiplicationAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) *= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::multiplicationAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](const T& val, const size_t& index) {val *= other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::divisionAsgmt(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Compute([&](const T& val, const size_t& index) {val /= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::divisionAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) /= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::divisionAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](const T& val, const size_t& index) {val /= other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::modulouAsgmt(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Compute([&](const T& val, const size_t& index) {val %= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::modulouAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		for (size_t i = 0; i < other.size(); i++)
		{
			At(other.map_index(i)) %= other[i];
		}
	}

	template<typename T, Mode device>
	template<typename OT>
	inline void Tensor<T, device>::modulouAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](const T& val, const size_t& index) {val %= other; });
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape());

		Compute([&](const T& val, const size_t& index) {result = comp_func(val, other[index]); });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape("Must have the same number of dimensions in each Tensor", other.Shape(), Shape());
		}

		for (size_t i = 0; i < Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape("Must have same dimension length in each Tensor", other.Shape(), Shape());
			}
		}
		#endif

		Tensor<RT, device> result(Shape());

		Compute([&](const T& val, const size_t& index) {result[index] = comp_func(val, other); });

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::compare(const OT& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();

		Tensor<RT, device> result(Shape());

		Compute([&](const T& val, const size_t& index) {result[index] = comp_func(val, other); });

		return result;
	}

	
}