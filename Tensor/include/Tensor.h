#pragma once

#include <iostream>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <assert.h>
#include <functional>
#include <type_traits>
#include "TensorEnums.h"
#include "TensorSlice.h"

#if (defined(_DEBUG) || defined(DEBUG)) && !defined(_TS_DEBUG)
#ifndef _TS_NO_DEB_WARN
#pragma message("Warning: build is running in debug mode but the marco \"_TS_DEBUG\" has not been defined for the Tensor library.\nDisable this warning by defining \"_TS_NO_DEB_WARN\"")
#endif
#endif

#if __cplusplus == 199711L && !defined(_TS_NO_STD_WARN)
#pragma message("Warning: __cplusplus is set to its disabled value, the minimum standard version required to compile is c++17\nTo disable this warning define \"_TS_NO_STD_WARN\"")
#elif __cplusplus < 201703L
#error "std version must be greater than or equal to c++17"
#endif

namespace TSlib
{
	template<typename T>
	struct is_tensor_type : std::false_type {};

	template<typename T, Device device>
	struct is_tensor_type<Tensor<T, device>> : std::true_type {};

	template<typename T, Device device>
	struct is_tensor_type<TensorSlice<T, device>> : std::true_type {};

	template<typename T>
	struct is_tensor : std::false_type {};

	template<typename T, Device device>
	struct is_tensor<Tensor<T, device>> : std::true_type {};

	template<typename T>
	struct is_tensor_slice : std::false_type {};

	template<typename T, Device device>
	struct is_tensor_slice<TensorSlice<T, device>> : std::true_type {};

	#ifdef _CUDA

	#ifndef DEVICE_MAX_THREADS
	#define DEVICE_MAX_THREADS 1024
	#endif

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
	#endif

	template<typename T, Device device = default_device>
	class Tensor
	{
		friend TensorSlice<T, device>;

	protected:

		std::vector<T> m_vector;
		std::vector<size_t> m_shape;
		
		#ifdef _CUDA
		public:
		T* gpu_mem;
		#endif
	

	public:

		typedef T Type;
		static constexpr Device processor = device;

		Tensor();
		Tensor(const std::vector<size_t>& sizes, const T& pad_val = T());
		Tensor(const std::vector<size_t>& sizes, std::function<T()> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<T(const size_t&)> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&)> generator);
		Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&, const size_t&)> generator);
		Tensor(const TensorSlice<T, device>& slicee);

		Tensor(const Tensor<T, device>& other);
		Tensor(Tensor<T, device>&& other);

		void Save(std::string dir) const;
		Tensor<T, device>& Load(std::string dir);

		~Tensor();

		Tensor<T, device>& ResizeDim(const size_t& index, const size_t& amount, const T& pad_val = T());

		std::vector<size_t> FlattenDims(size_t dims) const;
		size_t FlattenDims() const;

		Tensor<T, device>& Resize(const std::vector<size_t>& sizes, const T& pad_val = T());
		template<size_t n>
		Tensor<T, device>& Resize(const std::array<size_t, n>& sizes, const T& pad_val = T());

		template<typename Ts, std::enable_if_t<std::is_integral<Ts>::value, int> = 0>
		Tensor<T, device>& Reshape(const std::vector<Ts>& shape);

		TensorSlice<T, device> AsShape(const std::vector<long long>& shape);

		Tensor<T, device>& SetDims(const size_t& dims);

		Tensor<T, device>& AddDims(const size_t& dims = 1);

		Tensor<T, device>& RemoveDims(const size_t& dims = 1);

		template<typename OT, Device o_device>
		Tensor<T, device>& Append(const Tensor<OT, o_device>& other, const size_t& dimension);

		Tensor<T, device>& Fill(const T& val = NULL);
		Tensor<T, device>& Fill(std::function<T(const size_t&)> generator);
		Tensor<T, device>& Fill(std::function<T(const std::vector<size_t>&)> generator);
		Tensor<T, device>& Fill(std::function<T(const std::vector<size_t>&, const size_t&)> generator);
		Tensor<T, device>& Fill(const std::vector<T>& vals);

		inline Tensor<T, device>& Compute(std::function<void(T&)> compute_func);
		inline Tensor<T, device>& Compute(std::function<void(T&, const size_t&)> compute_func);
		inline Tensor<T, device>& Compute(std::function<void(T&, const std::vector<size_t>&)> compute_func);
		inline Tensor<T, device>& Compute(std::function<void(T&, const std::vector<size_t>&, const size_t&)> compute_func);

		inline void Compute(std::function<void(const T&)> compute_func) const;
		inline void Compute(std::function<void(const T&, const size_t&)> compute_func) const;
		inline void Compute(std::function<void(const T&, const std::vector<size_t>&)> compute_func) const;
		inline void Compute(std::function<void(const T&, const std::vector<size_t>&, const size_t&)> compute_func) const;

		inline Tensor<T, device> Compute(std::function<void(T&, const T&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;
		inline Tensor<T, device> Compute(std::function<void(T&, const T&, const size_t&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;
		inline Tensor<T, device> Compute(std::function<void(T&, const T&, const std::vector<size_t>&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;
		inline Tensor<T, device> Compute(std::function<void(T&, const T&, const std::vector<size_t>&, const size_t&)> compute_func, size_t axis, T pad_val = T(), bool keepDims = true) const;

		Tensor<T, device>& Replace(const T& target, const T& value);

		//math functions

		template<typename TReturn = T>
		TReturn Sum() const;
		template<typename TReturn = T>
		Tensor<TReturn, device> Sum(size_t axis, bool keepDims = true) const;

		template<typename TReturn = T>
		TReturn Prod() const;
		template<typename TReturn = T>
		Tensor<TReturn, device> Prod(size_t axis, bool keepDims = true) const;

		Tensor<T, device>& Exp();

		Tensor<T, device>& Normalize();

		T Max() const;

		T Min() const;

		template<typename RT = T>
		RT Avg() const;

		Tensor<T, device>& Sin();
		Tensor<T, device>& Cos();
		Tensor<T, device>& Tan();

		Tensor<T, device>& ArcSin();
		Tensor<T, device>& ArcCos();
		Tensor<T, device>& ArcTan();

		Tensor<T, device>& ConvDeg();
		Tensor<T, device>& ConvRad();

		inline T& Get(const std::vector<size_t>& coords);
		template<typename ... Args>
		inline T& Get(const Args& ... coords);

		inline T Get(const std::vector<size_t>& coords) const;
		template<typename ... Args>
		inline T Get(const Args& ... coords) const;

		TensorSlice<T, device> Slice(const std::vector<TSlice>& slices);

		std::string printable() const;

		/*template<typename ... Args>
		TensorSlice<T, device> Slice(const std::initializer_list<Args>& ... slices);*/

		T& At(size_t indx);
		T At(size_t indx) const;

		#ifdef _CUDA

		void Allocate();

		void Deallocate();

		bool IsAllocated() const;
		bool IsDeallocated() const;

		void CopyGPU();

		void CopyCPU();

		void Push();

		void Pull();

		__device__ __host__ T* getGPU();
		__device__ __host__ const T* getGPU() const;

		void SetTargetThreads(const unsigned short&);
		short GetTargetThreads() const;

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
		inline T& operator()(Args ... coords);
		template<typename ... Args>
		inline T operator()(Args ... coords) const;

		T& operator[](size_t indx);
		T operator[](size_t indx) const;

		Tensor<T, device>& operator=(const std::vector<T>& other);
		Tensor<T, device>& operator=(const Tensor<T, device>& other);
		Tensor<T, device>& operator=(Tensor<T, device>&& other);

		#ifdef _CUDA
		operator T* ();

		operator const T* () const;
		#endif

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Add(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Add(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> Add(const OT& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Subtract(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Subtract(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> Subtract(const OT& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Multiply(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Multiply(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> Multiply(const OT& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Divide(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Divide(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> Divide(const OT& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Modulous(const Tensor<OT, o_device>& other) const;

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Modulous(const TensorSlice<OT, o_device>& other) const;

		template<typename RT = T, typename OT>
		Tensor<RT, device> Modulous(const OT& other) const;

		template<typename OT, Device o_device>
		void AdditionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void AdditionAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void AdditionAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void SubtractionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void SubtractionAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void SubtractionAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void MultiplicationAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void MultiplicationAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void MultiplicationAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void DivisionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void DivisionAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void DivisionAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void ModulouAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void ModulouAsgmt(const TensorSlice<OT, o_device>& other);

		template<typename OT>
		void ModulouAsgmt(const OT& other);

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

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cadd(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cadd(const Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cadd(const OT& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Csubtract(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Csubtract(const Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Csubtract(const OT& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cmultiply(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cmultiply(const Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cmultiply(const OT& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cdivide(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cdivide(const Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cdivide(const OT& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cmodulous(Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT, Device o_device>
		Tensor<RT, device> Cmodulous(const Tensor<OT, o_device>& other);

		template<typename RT = T, typename OT>
		Tensor<RT, device> Cmodulous(const OT& other);

		template<typename OT, Device o_device>
		void CadditionAsgmt(Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void CadditionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT>
		void CadditionAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void CsubtractionAsgmt(Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void CsubtractionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT>
		void CsubtractionAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void CmultiplicationAsgmt(Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void CmultiplicationAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT>
		void CmultiplicationAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void CdivisionAsgmt(Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void CdivisionAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT>
		void CdivisionAsgmt(const OT& other);

		template<typename OT, Device o_device>
		void CmodulouAsgmt(Tensor<OT, o_device>& other);

		template<typename OT, Device o_device>
		void CmodulouAsgmt(const Tensor<OT, o_device>& other);

		template<typename OT>
		void CmodulouAsgmt(const OT& other);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> Ccompare(const Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> Ccompare(const OT& other);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> ClessThan(const Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> ClessThan(const OT& other);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> CgreaterThan(const Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> CgreaterThan(const OT& other);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> ClessThanEqual(const Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> ClessThanEqual(const OT& other);

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> CgreaterThanEqual(const Tensor<OT, o_device>& other);

		template<typename RT = char, typename OT>
		Tensor<RT, device> CgreaterThanEqual(const OT& other);

		#endif

		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> Compare(const Tensor<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal) const;
		template<typename RT = char, typename OT, Device o_device>
		Tensor<RT, device> Compare(const TensorSlice<OT, o_device>& other, bool(*comp_func)(const T&, const OT&) = Equal) const;
		template<typename RT = char, typename OT, std::enable_if_t<!is_tensor_type<OT>::value, int> = 0>
		Tensor<RT, device> Compare(const OT& other, bool(*comp_func)(const T&, const OT&) = Equal) const;

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
		inline void operator+=(OT& other);
		template<typename OT>
		inline void operator+=(const OT& other);

		template<typename OT>
		inline void operator-=(OT& other);
		template<typename OT>
		inline void operator-=(const OT& other);

		template<typename OT>
		inline void operator*=(OT& other);
		template<typename OT>
		inline void operator*=(const OT& other);

		template<typename OT>
		inline void operator/=(OT& other);
		template<typename OT>
		inline void operator/=(const OT& other);

		template<typename OT>
		inline void operator%=(OT& other);
		template<typename OT>
		inline void operator%=(const OT& other);

		template<typename OT>
		inline bool operator==(OT& other);
		template<typename OT>
		inline bool operator==(const OT& other);

		template<typename OT>
		inline bool operator!=(OT& other);
		template<typename OT>
		inline bool operator!=(const OT& other);

		template<typename OT>
		inline bool operator<(OT& other);
		template<typename OT>
		inline bool operator<(const OT& other);

		template<typename OT>
		inline bool operator>(OT& other);
		template<typename OT>
		inline bool operator>(const OT& other);

		template<typename OT>
		inline bool operator<=(OT& other);
		template<typename OT>
		inline bool operator<=(const OT& other);

		template<typename OT>
		inline bool operator>=(OT& other);
		template<typename OT>
		inline bool operator>=(const OT& other);

		#ifdef _CUDA
		operator CUDATensor1D<T>();
		operator const CUDATensor1D<T>() const;

		operator CUDATensor2D<T>();
		operator const CUDATensor2D<T>() const;

		operator CUDATensor3D<T>();
		operator const CUDATensor3D<T>() const;
		#endif

		template<typename Tprint, Device device_print>
		friend std::ostream& operator<<(std::ostream& stream, const Tensor<Tprint, device_print>& tensor);

		template<typename CT, Device o_device>
		operator Tensor<CT, o_device>() const
		{
			Tensor<CT, device> new_Tensor(Shape(), CT());

			for (size_t i = 0; i < size(); i++)
			{
				new_Tensor[i] = (CT)At(i);
			}

			return new_Tensor;
		}

		template<typename CT, Device o_device>
		operator Tensor<CT, o_device>()
		{
			Tensor<CT, o_device> new_Tensor(Shape(), CT());

			for (size_t i = 0; i < size(); i++)
			{
				new_Tensor[i] = (CT)At(i);
			}

			return new_Tensor;
		}

	#ifdef _CUDA
	protected:
		bool allocated = false;
		constexpr static unsigned char MAX_THREADS = DEVICE_MAX_THREADS / 32;
		unsigned short m_threads = MAX_THREADS;
	#endif

		size_t calc_new_size(const std::initializer_list<size_t>& sizes);
		size_t calc_new_size(const std::vector<size_t>& sizes);
		template<size_t n>
		size_t calc_new_size(const std::array<size_t, n>& sizes);

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
		template<size_t n>
		std::array<size_t, n> based_sort(const std::array<size_t, n>& target);
		std::vector<size_t> based_sort(const std::vector<TSlice>& target);
	};

	double round(double x, double place);
}

#include "Tensor.ipp"
