#pragma once

#include <stdint.h>
#include "TSliceWrapper.h"
#include "TensorCompareOperators.h"

namespace TSlib
{
	template<typename T, Device device>
	class Tensor;

	template<typename T, Device device>
	class TensorSlice
	{
		Tensor<T, device>* source;
		std::vector<TSlice> m_slice_shape;
		std::vector<size_t> m_real_shape;
		std::vector<size_t> m_shape;
		size_t m_offset;

		#ifdef _TS_DEBUG

		template<typename First, typename ... Args>
		void bounds_check(size_t& i, First first, Args ... remaining);

		template<typename First>
		void bounds_check(size_t& i, First first);

		#endif

		void calc_offset();

		class iterator;

		T copy_generator(const size_t& index);

		size_t get_real_size(const size_t& index) const;
		size_t get_dim_length(const size_t& index) const;

		template<typename First>
		void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord);
		template<typename First, typename... Args>
		void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining);

	public:

		typedef T Type;
		static constexpr Device processor = device;

		TensorSlice(Tensor<T, device>* source, const std::vector<TSlice>& slices);

		void update();

		template<typename OT, Device device_other>
		void Fill(const Tensor<OT, device_other>& other);
		template<typename OT, Device device_other>
		void Fill(const TensorSlice<OT, device_other>& other);
		void Fill(const T& val);
		void Fill(std::function<T(const size_t&)> generator);
		void Fill(std::function<T(const std::vector<size_t>&)> generator);
		void Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator);
		void Fill(const std::vector<T>& vals);

		inline void Compute(std::function<void(T&)> compute_func);
		inline void Compute(std::function<void(T&, const size_t&)> compute_func);
		inline void Compute(std::function<void(T&, const std::vector<size_t>&)> compute_func);
		inline void Compute(std::function<void(T&, const std::vector<size_t>&, const size_t&)> compute_func);

		inline Tensor<T, device> Compute(std::function<void(T&, const T&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true)const;
		inline Tensor<T, device> Compute(std::function<void(T&, const T&, const size_t&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;
		inline Tensor<T, device> Compute(std::function<void(T&, const T&, const std::vector<size_t>&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;
		inline Tensor<T, device> Compute(std::function<void(T&, const T&, const std::vector<size_t>&, const size_t&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;

		void Replace(const T& target, const T& value);

		size_t size() const;

		size_t Dims() const;

		size_t MapIndex(size_t index) const;

		const std::vector<size_t>& Shape() const;
		const std::vector<TSlice>& TSliceShape() const;

		TensorSlice<T, device>& Reshape(const std::vector<long long>& shape);

		TensorSlice<T, device>& Exp();
		TensorSlice<T, device>& Normalize();

		template<typename TReturn = T>
		TReturn Sum() const;
		template<typename TReturn = T>
		Tensor<TReturn, device> Sum(size_t axis, bool keepDims = false) const;

		template<typename TReturn = T>
		TReturn Prod() const;
		template<typename TReturn = T>
		Tensor<TReturn, device> Prod(size_t axis, bool keepDims = true) const;

		T Max() const;

		T Min() const;

		template<typename RT = T>
		RT Avg() const;

		TensorSlice<T, device>& Sin();
		TensorSlice<T, device>& Cos();
		TensorSlice<T, device>& Tan();

		TensorSlice<T, device>& ArcSin();
		TensorSlice<T, device>& ArcCos();
		TensorSlice<T, device>& ArcTan();

		TensorSlice<T, device>& ConvDeg();
		TensorSlice<T, device>& ConvRad();

		template<typename ... Args>
		T& Get(Args ... coords);
		T& Get(const std::vector<size_t>& coords);
		template<typename ... Args>
		T Get(Args ... coords) const;
		T Get(const std::vector<size_t>& coords) const;

		T& At(size_t index);
		T At(size_t index) const;

		template<typename RT, Device return_device = device>
		Tensor<RT, return_device> asVector();

		T operator[](size_t index) const;
		T& operator[](size_t index);

		template<typename RT, Device return_device = device>
		operator Tensor<RT, return_device>();

		std::string printable() const;

		iterator begin();

		iterator end();

		template<typename OT, Device other_device>
		Tensor<T, device> Add(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> Add(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> Add(const OT& other);

		template<typename OT, Device other_device>
		void AddAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void AddAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void AddAsgmt(const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> Subtract(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> Subtract(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> Subtract(const OT& other);

		template<typename OT, Device other_device>
		void SubtractAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void SubtractAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void SubtractAsgmt(const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> Multiply(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> Multiply(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> Multiply(const OT& other);

		template<typename OT, Device other_device>
		void MultiplyAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void MultiplyAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void MultiplyAsgmt(const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> dot(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> dot(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> dot(const OT& other);

		template<typename OT, Device other_device>
		void dotAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void dotAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void dotAsgmt(const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> Divide(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> Divide(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> Divide(const OT& other);

		template<typename OT, Device other_device>
		void divideAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void divideAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void divideAsgmt(const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> modulou(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> modulou(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> modulou(const OT& other);

		template<typename OT, Device other_device>
		void ModulouAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void ModulouAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void ModulouAsgmt(const OT& other);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> Compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> Compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename RT = char, typename OT>
		Tensor<RT, device> Compare(const OT& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename OT, Device other_device>
		Tensor<char, device> LessThan(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<char, device> LessThan(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> LessThan(const OT& other);

		template<typename OT, Device other_device>
		Tensor<char, device> GreaterThan(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<char, device> GreaterThan(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> GreaterThan(const OT& other);

		template<typename OT, Device other_device>
		Tensor<char, device> LessThanEqual(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<char, device> LessThanEqual(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> LessThanEqual(const OT& other);

		template<typename OT, Device other_device>
		Tensor<char, device> GreaterThanEqual(const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<char, device> GreaterThanEqual(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> GreaterThanEqual(const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> operator+ (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> operator+ (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator+ (const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> operator- (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> operator- (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator- (const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> operator* (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> operator* (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator* (const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> operator/ (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> operator/ (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator/ (const OT& other);

		template<typename OT, Device other_device>
		Tensor<T, device> operator% (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		Tensor<T, device> operator% (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator% (const OT& other);

		template<typename OT, Device other_device>
		void operator+= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void operator+= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void operator+= (const OT& other);

		template<typename OT, Device other_device>
		void operator-= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void operator-= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void operator-= (const OT& other);

		template<typename OT, Device other_device>
		void operator*= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void operator*= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void operator*= (const OT& other);

		template<typename OT, Device other_device>
		void operator/= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void operator/= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void operator/= (const OT& other);

		template<typename OT, Device other_device>
		void operator%= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		void operator%= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void operator%= (const OT& other);

		template<typename OT, Device other_device>
		bool operator== (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		bool operator== (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator== (const OT& other);

		template<typename OT, Device other_device>
		bool operator!= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		bool operator!= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator!= (const OT& other);

		template<typename OT, Device other_device>
		bool operator< (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		bool operator< (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator< (const OT& other);

		template<typename OT, Device other_device>
		bool operator> (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		bool operator> (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator> (const OT& other);

		template<typename OT, Device other_device>
		bool operator<= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		bool operator<= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator<= (const OT& other);

		template<typename OT, Device other_device>
		bool operator>= (const Tensor<OT, other_device>& other);
		template<typename OT, Device other_device>
		bool operator>= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator>= (const OT& other);
	};
}
