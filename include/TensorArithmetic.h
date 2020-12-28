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
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val + other[index]; });

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
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline Tensor<RT, device> Tensor<T, device>::addSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val + other; });

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
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		fCompute([result, other](T& val, const size_t& index) {result[index] = val - other[index]; });

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
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline Tensor<RT, device> Tensor<T, device>::subtractSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val + other; });

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
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val * other[index]; });

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
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline Tensor<RT, device> Tensor<T, device>::multiplySingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val + other; });

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
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val / other[index]; });

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
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline Tensor<RT, device> Tensor<T, device>::divideSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val / other; });

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
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val % other[index]; });

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
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline Tensor<RT, device> Tensor<T, device>::modulousSingle(const OT& other) const
	{
		MEASURE();
		Tensor<RT, device> result(this->Shape(), RT());

		Compute([result, other](T& val, const size_t& index) {result[index] = val % other; });

		return result;
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::additionAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Compute([other](T& val, const size_t& index) {val += other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::additionAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline void Tensor<T, device>::additionAssignmentSingle(const OT& other)
	{
		MEASURE();

		Compute([other](T& val) {val += other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::subtractionAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Compute([other](T& val, const size_t& index) {val -= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::subtractionAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline void Tensor<T, device>::subtractionAssignmentSingle(const OT& other)
	{
		MEASURE();
		Compute([other](T& val, const size_t& index) {val -= other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::multiplicationAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Compute([other](T& val, const size_t& index) {val *= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::multiplicationAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline void Tensor<T, device>::multiplicationAssignmentSingle(const OT& other)
	{
		MEASURE();
		Compute([other](T& val, const size_t& index) {val *= other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::divisionAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Compute([other](T& val, const size_t& index) {val /= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::divisionAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline void Tensor<T, device>::divisionAssignmentSingle(const OT& other)
	{
		MEASURE();
		Compute([other](T& val, const size_t& index) {val /= other; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::modulouAssignment(const Tensor<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Compute([other](T& val, const size_t& index) {val %= other[index]; });
	}

	template<typename T, Mode device>
	template<typename OT, Mode o_device>
	inline void Tensor<T, device>::modulouAssignment(const TensorSlice<OT, o_device>& other)
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() < other.Dims())
		{
			throw BadShape(this, "Must have less than or the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
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
	inline void Tensor<T, device>::modulouAssignmentSingle(const OT& other)
	{
		MEASURE();
		Compute([other](T& val, const size_t& index) {val %= other; });
	}

	template<typename T, Mode device>
	template<typename RT, typename OT, Mode o_device>
	inline Tensor<RT, device> Tensor<T, device>::compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();
		#ifdef _TS_DEBUG
		if (Dims() != other.Dims())
		{
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape());

		Compute([comp_func, result, other](T& val, const size_t& index) {result = comp_func(val, other[index]); });

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
			throw BadShape(this, "Must have the same number of dimensions in each Tensor", other.Shape());
		}

		for (size_t i = 0; i < this->Dims(); i++)
		{
			if (Shape()[i] != other.Shape()[i])
			{
				throw BadShape(this, "Must have same dimension length in each Tensor", other.Shape());
			}
		}
		#endif

		Tensor<RT, device> result(this->Shape());

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other[i]);
		}

		return result;
	}

	template<typename T, Mode device>
	template<typename RT, typename OT>
	inline Tensor<RT, device> Tensor<T, device>::compareSingle(const OT& other, bool(*comp_func)(const T&, const OT&))
	{
		MEASURE();

		Tensor<RT, device> result(this->Shape());

		Compute([comp_func, result, other](T& val, const size_t& index) {result[index] = comp_func(val, other); });

		for (size_t i = 0; i < this->size(); i++)
		{
			result[i] = comp_func(At(i), other);
		}

		return result;
	}
}