#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>
#include <functional>
#include "TensorEnums.h"
#include "TensorSliceBones.h"
#include "TensorExceptions.h"
#include "TensorCompareOperators.h"

namespace TSlib
{
	#ifdef _CUDA

	#ifndef DEVICE_MAX_THREADS
	#define DEVICE_MAX_THREADS 1024
	#endif

	constexpr Mode default_device = Mode::GPU;

	template<typename T>
	class CTBase;

	template<typename T>
	class CUDATensor1D;

	template<typename T>
	class CUDATensor2D;

	template<typename T>
	class CUDATensor3D;

	template<Mode layout>
	class CUDALayout;

	#else
	constexpr Mode default_device = Mode::CPU;
	#endif

	template<typename T, Mode device = default_device>
	class Tensor
	{
		friend TensorSlice<T, device>;

	protected:

		std::vector<T> m_vector;
		std::vector<size_t> m_shape;

		#ifdef _CUDA
	public:
		T* gpu_mem;
	protected:
		bool allocated = false;
		constexpr static unsigned char MAX_THREADS = DEVICE_MAX_THREADS / 32;
		unsigned short m_threads = MAX_THREADS;
		#endif

		size_t calc_new_size(const std::initializer_list<size_t>& sizes);
		size_t calc_new_size(const std::vector<size_t>& sizes);

		size_t get_real_size(const size_t& index) const;
		size_t get_dim_length(const size_t& index) const;

		void upscale_dim(const size_t& index, const size_t& row_size, const size_t& amount, const T& pad_val);

		void downscale_dim(const size_t& index, const size_t& row_size, const size_t& amount);

		template<typename First>
		void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord);

		template<typename First, typename ... Args>
		void get_indx(size_t& indx, size_t& iter, size_t& tmp_multiply, First coord, Args ... remaining);

		template<typename First>
		void to_vector(std::vector<TSlice>& vec, const std::initializer_list<First>& val);

		template<typename ... Args, typename First>
		void to_vector(std::vector<TSlice>& vec, const std::initializer_list<First>& first, const std::initializer_list<Args>& ... args);

		std::vector<size_t> based_sort(const std::vector<size_t>& target);

		std::vector<size_t> based_sort(const std::vector<TSlice>& target);

	public:

		Tensor();
		Tensor(const std::vector<size_t>& sizes, const T& pad_val = T());
		Tensor(const std::vector<size_t>& sizes, std::function<T()> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<T(const size_t&)> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&)> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&, const size_t&)> generator);
		Tensor(const TensorSlice<T, device>& slicee);

		Tensor(const Tensor<T, device>& other);

		~Tensor();

		void ResizeDim(const size_t& index, const size_t& amount, const T& pad_val = T());

		std::vector<size_t> FlattenDims(size_t dims) const;
		size_t FlattenDims() const;

		void Resize(const std::vector<size_t>& sizes, const T& pad_val = T());

		void Reshape(const std::initializer_list<size_t>& shape);
		void Reshape(const std::vector<size_t>& shape);

		void SetDims(const size_t& dims);

		void AddDims(const size_t& dims = 1);

		void RemoveDims(const size_t& dims = 1);

		template<typename OT, Mode o_device>
		void Append(const Tensor<OT, o_device>& other, const size_t& dimension);

		void Fill(const T& val = NULL);
		void Fill(std::function<T(const size_t&)> generator);
		void Fill(std::function<T(const std::vector<size_t>&)> generator);
		void Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator);
		void Fill(const std::vector<T>& vals);

		inline void Compute(std::function<void(T&)> compute_func);
		inline void Compute(std::function<void(T&, const size_t&)> compute_func);
		inline void Compute(std::function<void(T&, const std::vector<size_t>&)> compute_func);
		inline void Compute(std::function<void(T&, const std::vector<size_t>&, const size_t&)> compute_func);

		inline void Compute(std::function<void(const T&)> compute_func) const;
		inline void Compute(std::function<void(const T&, const size_t&)> compute_func) const;
		inline void Compute(std::function<void(const T&, const std::vector<size_t>&)> compute_func) const;
		inline void Compute(std::function<void(const T&, const std::vector<size_t>&, const size_t&)> compute_func) const;

		void Replace(const T& target, const T& value);

		template<typename ... Args>
		T& Get(const Args& ... coords);

		TensorSlice<T, device> Slice(const std::vector<TSlice>& slices);

		std::string printable() const;

		/*template<typename ... Args>
		TensorSlice<T, device> Slice(const std::initializer_list<Args>& ... slices);*/

		T& At(size_t indx);
		T At(size_t indx) const;

		#ifdef _CUDA

		void allocate();

		void deallocate();

		bool isAllocated() const;
		bool isDeallocated() const;

		void copyGPU();

		void copyCPU();

		void push();

		void pull();

		__device__ __host__ T* getGPU();
		__device__ __host__ const T* getGPU() const;

		void setTargetThreads(const unsigned short&);
		short getTargetThreads() const;

		#endif

		size_t get_dim_offset(const size_t& index) const;

		size_t Dims() const;
		const std::vector<size_t>& Shape() const;

		//Iterator setup
		typename std::vector<T>::const_iterator begin() const;
		typename std::vector<T>::iterator begin();

		typename std::vector<T>::const_iterator end() const;
		typename std::vector<T>::iterator end();

		size_t size() const;

		const std::vector<T>& asVector() const;

		const T* Data() const;
		T* Data();

		template<typename ... Args>
		T& operator()(Args ... coords);

		T& operator[](size_t indx);
		T operator[](size_t indx) const;

		Tensor<T>& operator=(const std::vector<T>& other);

		template<typename RT = T>
		RT sum();

		#ifdef _CUDA
		operator T* ();
		#endif

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> add(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> add(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> add(const OT& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> subtract(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> subtract(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> subtract(const OT& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> multiply(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> multiply(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> multiply(const OT& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> divide(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> divide(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> divide(const OT& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> modulous(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> modulous(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> modulous(const OT& other) const;

		template<typename OT, Mode o_device>
		void additionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Mode o_device>
		void additionAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void additionAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void subtractionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Mode o_device>
		void subtractionAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void subtractionAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void multiplicationAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Mode o_device>
		void multiplicationAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void multiplicationAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void divisionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Mode o_device>
		void divisionAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void divisionAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void modulouAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Mode o_device>
		void modulouAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void modulouAsgmt(const OT& other);

		#ifdef _CUDA
		template<Mode L, typename FT, typename RT = T, typename ... Args>
		Tensor<RT, device> Kernel1DR(CUDALayout<L> layout, FT(kernel_p), Args&& ... args);

		template<Mode L, typename FT, typename ... Args>
		void Kernel1D(CUDALayout<L> layout, FT(kernel_p), Args&& ... args);

		template<Mode L, typename FT, typename RT = T, typename ... Args>
		Tensor<RT, device> Kernel2DR(CUDALayout<L> layout, FT(kernel_p), Args&& ... args);

		template<Mode L, typename FT, typename ... Args>
		void Kernel2D(CUDALayout<L> layout, FT(kernel_p), Args&& ... args);

		template<Mode L, typename FT, typename RT = T, typename ... Args>
		Tensor<RT, device> Kernel3DR(CUDALayout<L> layout, FT(kernel_p), Args&& ... args);

		template<Mode L, typename FT, typename ... Args>
		void Kernel3D(CUDALayout<L> layout, FT(kernel_p), Args&& ... args);

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> Cadd(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cadd(const OT& other);

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> Csubtract(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Csubtract(const OT& other);

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> Cmultiply(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cmultiply(const OT& other);

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> Cdivide(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cdivide(const OT& other);

		template<typename RT = T, typename OT, Mode o_device>
		Tensor<RT, device> Cmodulous(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cmodulous(const OT& other);

		template<typename OT, Mode o_device>
		void CadditionAsgmt(Tensor<OT, o_device>& other);

		template<typename OT>
		void CadditionAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void CsubtractionAsgmt(Tensor<OT, o_device>& other);

		template<typename OT>
		void CsubtractionAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void CmultiplicationAsgmt(Tensor<OT, o_device>& other);

		template<typename OT>
		void CmultiplicationAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void CdivisionAsgmt(Tensor<OT, o_device>& other);

		template<typename OT>
		void CdivisionAsgmt(const OT& other);

		template<typename OT, Mode o_device>
		void CmodulouAsgmt(Tensor<OT, o_device>& other);

		template<typename OT>
		void CmodulouAsgmt(const OT& other);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> Ccompare(Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> Ccompare(const OT& other);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> ClessThan(Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> ClessThan(const OT& other);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> CgreaterThan(Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> CgreaterThan(const OT& other);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> ClessThanEqual(Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> ClessThanEqual(const OT& other);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> CgreaterThanEqual(Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> CgreaterThanEqual(const OT& other);

		#endif

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename RT = char, typename OT, Mode o_device>
		Tensor<RT, device> compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename RT = char, typename OT>
		Tensor<RT, device> compare(const OT& other, bool(*comp_func)(const T&, const OT&) = Equal);

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator==(Tensor<OT, o_device>& other)
		{
			#ifdef __clang__
			return compare(other).template sum<size_t>() == other.size();
			#else
			return compare(other).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator==(TensorSlice<OT, o_device>& other)
		{
			#ifdef __clang__
			return compare(other).template sum<size_t>() == other.size();
			#else
			return compare(other).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator==(const OT& other)
		{
			#ifdef __clang__
			return compare(other).template sum<size_t>() == other.size();
			#else
			return compare(other).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator!=(Tensor<OT, o_device>& other)
		{
			#ifdef __clang__
			return compare(other, NotEqual).template sum<size_t>() == other.size();
			#else
			return compare(other, NotEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator!=(TensorSlice<OT, o_device>& other)
		{
			#ifdef __clang__
			return !(bool)compare(other, NotEqual).template sum<size_t>() == other.size();
			#else
			return !(bool)compare(other, NotEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator!=(const OT& other)
		{
			#ifdef __clang__
			return !(bool)compare(other, NotEqual).template sum<size_t>() == other.size();
			#else
			return !(bool)compare(other, NotEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator<(Tensor<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, LessThan).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, LessThan).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator<(TensorSlice<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, LessThan).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, LessThan).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator<(const OT& other)
		{
			#ifdef __clang__
			return (bool)compare(other, LessThan).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, LessThan).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator>(Tensor<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, GreaterThan).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, GreaterThan).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator>(TensorSlice<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, GreaterThan).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, GreaterThan).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator>(const OT& other)
		{
			#ifdef __clang__
			return (bool)compare(other, GreaterThan).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, GreaterThan).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator<=(Tensor<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, LessThanEqual).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, LessThanEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator<=(TensorSlice<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, LessThanEqual).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, LessThanEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator<=(const OT& other)
		{
			#ifdef __clang__
			return (bool)compare(other, LessThanEqual).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, LessThanEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator>=(Tensor<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, GreaterThanEqual).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, GreaterThanEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator>=(TensorSlice<OT, o_device>& other)
		{
			#ifdef __clang__
			return (bool)compare(other, GreaterThanEqual).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, GreaterThanEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
		inline bool operator>=(const OT& other)
		{
			#ifdef __clang__
			return (bool)compare(other, GreaterThanEqual).template sum<size_t>() == other.size();
			#else
			return (bool)compare(other, GreaterThanEqual).sum<size_t>() == other.size();
			#endif
		}

		template<typename OT>
		inline Tensor<T, device> operator+(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator+(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator-(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator-(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator*(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator*(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator/(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator/(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator%(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator%(const OT& other);


		template<typename OT>
		inline Tensor<T, device> operator+=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator+=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator-=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator-=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator*=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator*=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator/=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator/=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator%=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator%=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator==(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator==(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator!=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator!=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator<(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator<(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator>(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator>(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator<=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator<=(const OT& other);

		template<typename OT>
		inline Tensor<T, device> operator>=(OT& other);
		template<typename OT>
		inline Tensor<T, device> operator>=(const OT& other);

		#ifdef _CUDA
		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator==(Tensor<OT, o_device>& other)
		{
			return Ccompare(other).sum<size_t>() == other.size();
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator==(const OT& other)
		{
			return (bool)Ccompare(other).sum<size_t>() == other.size();
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator!=(Tensor<OT, o_device>& other)
		{
			return !(bool)Ccompare(other).sum<size_t>() == other.size();
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator!=(const OT& other)
		{
			return !(bool)Ccompare(other).sum<size_t>() == other.size();
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator<(Tensor<OT, o_device>& other)
		{
			return (bool)ClessThan(other).sum<size_t>() == other.size();
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator<(const OT& other)
		{
			return (bool)ClessThan(other).sum<size_t>() == other.size();
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator>(Tensor<OT, o_device>& other)
		{
			return (bool)CmoreThan(other).sum<size_t>() == other.size();
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator>(const OT& other)
		{
			return (bool)CmoreThan(other).sum<size_t>() == other.size();
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator<=(Tensor<OT, o_device>& other)
		{
			return (bool)ClessThanEqual(other).sum<size_t>() == other.size();
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator<=(const OT& other)
		{
			return (bool)ClessThanEqual(other).sum<size_t>() == other.size();
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator>=(Tensor<OT, o_device>& other)
		{
			return (bool)CmoreThanEqual(other).sum<size_t>() == other.size();
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline bool operator>=(const OT& other)
		{
			return (bool)CmoreThanEqual(other).sum<size_t>() == other.size();
		}

		/*template<typename OT, Mode o_device>
		typename std::enable_if_t<device == Mode::GPU, Tensor<T, device>>
		operator+(Tensor<OT, o_device>& other);*/

		/*template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator+(const OT& other)
		{
			return Cadd<T, OT>(other);
		}*/

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator-(Tensor<OT, o_device>& other)
		{
			return Csubtract(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator-(const OT& other)
		{
			return Csubtract(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator*(Tensor<OT, o_device>& other)
		{
			return Cmultiply(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator*(const OT& other)
		{
			return Cmultiply(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator/(Tensor<OT, o_device>& other)
		{
			return Cdivide(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator/(const OT& other)
		{
			return Cdivide(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator%(Tensor<OT, o_device>& other)
		{
			return Cmodulous(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline Tensor<T, device> operator%(const OT& other)
		{
			return Cmodulous(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator+=(Tensor<OT, o_device>& other)
		{
			CadditionAsgmt(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator+=(const OT& other)
		{
			CadditionAsgmt(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator-=(Tensor<OT, o_device>& other)
		{
			CsubtractionAsgmt(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator-=(const OT& other)
		{
			CsubtractionAsgmt(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator*=(Tensor<OT, o_device>& other)
		{
			CmultiplicationAsgmt(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator*=(const OT& other)
		{
			CmultiplicationAsgmt(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator/=(Tensor<OT, o_device>& other)
		{
			CdivisionAsgmt(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator/=(const OT& other)
		{
			CdivisionAsgmt(other);
		}

		template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator%=(Tensor<OT, o_device>& other)
		{
			CmodulouAsgmt(other);
		}

		template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
		inline void operator%=(const OT& other)
		{
			CmodulouAsgmt(other);
		}

		operator CUDATensor1D<T>();

		operator CUDATensor2D<T>();

		operator CUDATensor3D<T>();

		#endif

		template<typename CT, Mode o_device>
		operator Tensor<CT, o_device>() const
		{
			Tensor<CT, device> new_Tensor(Shape(), CT());

			for (size_t i = 0; i < size(); i++)
			{
				new_Tensor[i] = (CT)At(i);
			}

			return new_Tensor;
		}

		template<typename CT, Mode o_device>
		operator Tensor<CT, o_device>()
		{
			Tensor<CT, o_device> new_Tensor(Shape(), CT());

			for (size_t i = 0; i < size(); i++)
			{
				new_Tensor[i] = (CT)At(i);
			}

			return new_Tensor;
		}
	};
}
