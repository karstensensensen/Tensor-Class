#pragma once

#include <stdint.h>
#include "TSliceWrapper.h"
#include "TensorCompareOperators.h"

namespace TSlib
{
	template<typename T, Mode device>
	class Tensor;

	template<typename T, Mode device>
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

		template<typename First>
		void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord);
		template<typename First, typename... Args>
		void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining);

	public:
		TensorSlice(Tensor<T, device>* source, const std::vector<TSlice>& slices);

		void update();

		template<typename OT, Mode device_other>
		void Fill(const Tensor<OT, device_other>& other);
		template<typename OT, Mode device_other>
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

		void Replace(const T& target, const T& value);

		size_t size() const;

		size_t Dims() const;

		template<typename RT = T>
		RT sum();

		size_t map_index(size_t index) const;

		const std::vector<size_t>& Shape() const;

		TensorSlice<T, device>& Reshape(const std::vector<long long>& shape);

		template<typename ... Args>
		T& Get(Args ... coords);
		template<typename ... Args>
		T Get(Args ... coords) const;

		T& At(size_t index);
		T At(size_t index) const;

		template<typename RT, Mode return_device = device>
		Tensor<RT, return_device> asVector();

		T operator[](size_t index) const;
		T& operator[](size_t index);

		template<typename RT, Mode return_device = device>
		operator Tensor<RT, return_device>();

		std::string printable() const;

		iterator begin();

		iterator end();

		template<typename OT, Mode other_device>
		Tensor<T, device> add(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> add(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> add(const OT& other);

		template<typename OT, Mode other_device>
		void addAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		void addAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void addAsgmt(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> subtract(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> subtract(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> subtract(const OT& other);

		template<typename OT, Mode other_device>
		void subtractAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		void subtractAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void subtractAsgmt(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> multiply(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> multiply(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> multiply(const OT& other);

		template<typename OT, Mode other_device>
		void multiplyAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		void multiplyAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void multiplyAsgmt(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> dot(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> dot(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> dot(const OT& other);

		template<typename OT, Mode other_device>
		void dotAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		void dotAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void dotAsgmt(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> divide(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> divide(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> divide(const OT& other);

		template<typename OT, Mode other_device>
		void divideAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		void divideAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void divideAsgmt(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> modulou(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> modulou(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> modulou(const OT& other);

		template<typename OT, Mode other_device>
		void modulouAsgmt(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		void modulouAsgmt(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		void modulouAsgmt(const OT& other);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename RT = char, typename OT>
		Tensor<RT, device> compare(const OT& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename OT, Mode other_device>
		Tensor<char, device> lessThan(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<char, device> lessThan(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> lessThan(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<char, device> greaterThan(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<char, device> greaterThan(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> greaterThan(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<char, device> lessThanEqual(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<char, device> lessThanEqual(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> lessThanEqual(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<char, device> greaterThanEqual(const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<char, device> greaterThanEqual(const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<char, device> greaterThanEqual(const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> operator+ (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> operator+ (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator+ (const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> operator- (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> operator- (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator- (const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> operator* (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> operator* (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator* (const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> operator/ (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> operator/ (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator/ (const OT& other);

		template<typename OT, Mode other_device>
		Tensor<T, device> operator% (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		Tensor<T, device> operator% (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		Tensor<T, device> operator% (const OT& other);

		template<typename OT, Mode other_device>
		bool operator== (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		bool operator== (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator== (const OT& other);

		template<typename OT, Mode other_device>
		bool operator!= (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		bool operator!= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator!= (const OT& other);

		template<typename OT, Mode other_device>
		bool operator< (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		bool operator< (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator< (const OT& other);

		template<typename OT, Mode other_device>
		bool operator> (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		bool operator> (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator> (const OT& other);

		template<typename OT, Mode other_device>
		bool operator<= (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		bool operator<= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator<= (const OT& other);

		template<typename OT, Mode other_device>
		bool operator>= (const Tensor<OT, other_device>& other);
		template<typename OT, Mode other_device>
		bool operator>= (const TensorSlice<OT, other_device>& other);
		template<typename OT>
		bool operator>= (const OT& other);
	};
}
