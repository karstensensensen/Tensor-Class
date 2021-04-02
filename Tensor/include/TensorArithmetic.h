#pragma once
#include "Tensor.h"
#include "TensorEnums.h"

namespace TSlib
{
	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	Tensor<RT, device> Tensor<T, device>::Add(const Tensor<OT, o_device>& other) const
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

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Add(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			result[other.MapIndex(i)] = At(other.MapIndex(i)) + other[i];
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::Add(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val + other; });

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Subtract(const Tensor<OT, o_device>& other) const
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

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Subtract(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			result[other.MapIndex(i)] = At(other.MapIndex(i)) - other[i];
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::Subtract(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val + other; });

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Multiply(const Tensor<OT, o_device>& other) const
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

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Multiply(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			result[other.MapIndex(i)] = At(other.MapIndex(i)) * other[i];
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::Multiply(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val + other; });

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Divide(const Tensor<OT, o_device>& other) const
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

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Divide(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			result[other.MapIndex(i)] = At(other.MapIndex(i)) / other[i];
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::Divide(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val / other; });

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Modulous(const Tensor<OT, o_device>& other) const
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

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Modulous(const TensorSlice<OT, o_device>& other) const
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			result[other.MapIndex(i)] = At(other.MapIndex(i)) % other[i];
		}

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::Modulous(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([&](const T& val, const size_t& index) {result[index] = val % other; });

		return result;
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::AdditionAsgmt(const Tensor<OT, o_device>& other)
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

		Compute([&](T& val, const size_t& index) {val += other[index]; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::AdditionAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			At(other.MapIndex(i)) += other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	inline void Tensor<T, device>::AdditionAsgmt(const OT& other)
	{
		MEASURE();

		Compute([&](T& val) {val += other; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::SubtractionAsgmt(const Tensor<OT, o_device>& other)
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

		Compute([&](T& val, const size_t& index) {val -= other[index]; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::SubtractionAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			At(other.MapIndex(i)) -= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	inline void Tensor<T, device>::SubtractionAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](T& val, const size_t& index) {val -= other; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::MultiplicationAsgmt(const Tensor<OT, o_device>& other)
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

		Compute([&](T& val, const size_t& index) {val *= other[index]; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::MultiplicationAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			At(other.MapIndex(i)) *= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	inline void Tensor<T, device>::MultiplicationAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](const T& val, const size_t& index) {val *= other; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::DivisionAsgmt(const Tensor<OT, o_device>& other)
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

		Compute([&](T& val, const size_t& index) {val /= other[index]; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::DivisionAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			At(other.MapIndex(i)) /= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	inline void Tensor<T, device>::DivisionAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](T& val, const size_t& index) {val /= other; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::ModulouAsgmt(const Tensor<OT, o_device>& other)
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

		Compute([&](T& val, const size_t& index) {val %= other[index]; });
	}

	template<typename T, Device device>
	template<typename OT, Device o_device>
	inline void Tensor<T, device>::ModulouAsgmt(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape("Must have less than or the same number of dimensions in each Tensor", other.Shape(), Shape());
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
			At(other.MapIndex(i)) %= other[i];
		}
	}

	template<typename T, Device device>
	template<typename OT>
	inline void Tensor<T, device>::ModulouAsgmt(const OT& other)
	{
		MEASURE();
		Compute([&](T& val, const size_t& index) {val %= other; });
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&)) const
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

		Tensor<RT, device> result(Shape());

		Compute([&](const T& val, const size_t& index) {result[index] = comp_func(val, other[index]); });

		return result;
	}

	template<typename T, Device device>
	template<typename RT, typename OT, Device o_device>
	inline Tensor<RT, device> Tensor<T, device>::Compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&)) const
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

	template<typename T, Device device>
	template<typename RT, typename OT, std::enable_if_t<!is_tensor_type<OT>::value, int>>
	inline Tensor<RT, device> Tensor<T, device>::Compare(const OT& other, bool(*comp_func)(const T&, const OT&)) const
	{
		MEASURE();

		Tensor<RT, device> result(Shape());

		Compute([&](const T& val, const size_t& index) {result[index] = comp_func(val, other); });

		return result;
	}
}
