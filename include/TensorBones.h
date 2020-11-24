#pragma once

#include <iostream>
#include <vector>
#include <assert.h>
#include <functional>
#include "TensorEnums.h"
#include "TensorSliceBones.h"
#include "TensorExceptions.h"

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

	Tensor(const int& dims);
	Tensor(const std::vector<size_t>& sizes, const T& pad_val = T(), const bool& add_extra_dim = true);
	Tensor(const std::vector<size_t>& sizes, std::function<T(const size_t&)> generator, const bool& add_extra_dim = true);
	Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&)> generator, const bool& add_extra_dim = true);
	Tensor(const std::vector<size_t>& sizes, std::function<T(const std::vector<size_t>&, const size_t&)> generator, const bool& add_extra_dim = true);
	Tensor(const TensorSlice<T, device>& slice, const bool& add_extra_dim = true);
	
	Tensor(const Tensor<T, device>& other);

	~Tensor();

	void ResizeDim(const size_t& index, const size_t& amount, const T& pad_val = T());

	std::vector<size_t> FlattenDims(size_t dims) const;
	size_t FlattenDims() const;

	void Resize(const std::vector<size_t>& sizes, const T& pad_val = T());

	void Reshape(const std::initializer_list<size_t>& shape, bool add_extra_dim = true);
	void Reshape(const std::vector<size_t>& shape, bool add_extra_dim = true);

	void AddDims(const size_t& dims = 1);

	void RemoveDims(const size_t& dims = 1);

	void Fill(const T& val = NULL);
	void Fill(const size_t& dim, const T& val = NULL, const size_t& index = 0);

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

	size_t get_dim_size(const size_t& index) const;
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
	Tensor<RT, device> addSingle(const OT& other) const;
	
	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> subtract(const Tensor<OT, o_device>& other) const;

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> subtract(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = T, typename OT>
	Tensor<RT, device> subtractSingle(const OT& other) const;


	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> multiply(const Tensor<OT, o_device>& other) const;

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> multiply(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = T, typename OT>
	Tensor<RT, device> multiplySingle(const OT& other) const;

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> divide(const Tensor<OT, o_device>& other) const;

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> divide(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = T, typename OT>
	Tensor<RT, device> divideSingle(const OT& other) const;

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> modulous(const Tensor<OT, o_device>& other) const;

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> modulous(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = T, typename OT>
	Tensor<RT, device> modulousSingle(const OT& other) const;

	template<typename OT, Mode o_device>
	void additionAssignment(const Tensor<OT, o_device>& other);

	template<typename OT, Mode o_device>
	void additionAssignment(const TensorSlice<OT, o_device>& other);

	template<typename OT>
	void additionAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void subtractionAssignment(const Tensor<OT, o_device>& other);

	template<typename OT, Mode o_device>
	void subtractionAssignment(const TensorSlice<OT, o_device>& other);

	template<typename OT>
	void subtractionAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void multiplicationAssignment(const Tensor<OT, o_device>& other);

	template<typename OT, Mode o_device>
	void multiplicationAssignment(const TensorSlice<OT, o_device>& other);

	template<typename OT>
	void multiplicationAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void divisionAssignment(const Tensor<OT, o_device>& other);
	
	template<typename OT, Mode o_device>
	void divisionAssignment(const TensorSlice<OT, o_device>& other);

	template<typename OT>
	void divisionAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void modulouAssignment(const Tensor<OT, o_device>& other);
	
	template<typename OT, Mode o_device>
	void modulouAssignment(const TensorSlice<OT, o_device>& other);

	template<typename OT>
	void modulouAssignmentSingle(const OT& other);

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
	Tensor<RT, device> CaddSingle(const OT& other);

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> Csubtract(Tensor<OT, o_device>& other);

	template<typename RT = T, typename OT>
	Tensor<RT, device> CsubtractSingle(const OT& other);

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> Cmultiply(Tensor<OT, o_device>& other);

	template<typename RT = T, typename OT>
	Tensor<RT, device> CmultiplySingle(const OT& other);

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> Cdivide(Tensor<OT, o_device>& other);
	
	template<typename RT = T, typename OT>
	Tensor<RT, device> CdivideSingle(const OT& other);

	template<typename RT = T, typename OT, Mode o_device>
	Tensor<RT, device> Cmodulous(Tensor<OT, o_device>& other);

	template<typename RT = T, typename OT>
	Tensor<RT, device> CmodulousSingle(const OT& other);

	template<typename OT, Mode o_device>
	void CadditionAssignment(Tensor<OT, o_device>& other);

	template<typename OT>
	void CadditionAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void CsubtractionAssignment(Tensor<OT, o_device>& other);

	template<typename OT>
	void CsubtractionAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void CmultiplicationAssignment(Tensor<OT, o_device>& other);

	template<typename OT>
	void CmultiplicationAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void CdivisionAssignment(Tensor<OT, o_device>& other);

	template<typename OT>
	void CdivisionAssignmentSingle(const OT& other);

	template<typename OT, Mode o_device>
	void CmodulouAssignment(Tensor<OT, o_device>& other);

	template<typename OT>
	void CmodulouAssignmentSingle(const OT& other);

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> Ccompare(Tensor<OT, o_device>& other);

	template<typename RT = char, typename OT>
	Tensor<RT, device> CcompareSingle(const OT& other);

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> ClessThan(Tensor<OT, o_device>& other);

	template<typename RT = char, typename OT>
	Tensor<RT, device> ClessThanSingle(const OT& other);

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> CgreaterThan(Tensor<OT, o_device>& other);

	template<typename RT = char, typename OT>
	Tensor<RT, device> CgreaterThanSingle(const OT& other);

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> ClessThanEqual(Tensor<OT, o_device>& other);

	template<typename RT = char, typename OT>
	Tensor<RT, device> ClessThanEqualSingle(const OT& other);

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> CgreaterThanEqual(Tensor<OT, o_device>& other);

	template<typename RT = char, typename OT>
	Tensor<RT, device> CgreaterThanEqualSingle(const OT& other);

	#endif

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> compare(const Tensor<OT, o_device>& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> compare(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = char, typename OT>
	Tensor<RT, device> compareSingle(const OT& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> lessThan(const Tensor<OT, o_device>& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> lessThan(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = char, typename OT>
	Tensor<RT, device> lessThanSingle(const OT& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> greaterThan(const Tensor<OT, o_device>& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> greaterThan(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = char, typename OT>
	Tensor<RT, device> greaterThanSingle(const OT& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> lessThanEqual(const Tensor<OT, o_device>& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> lessThanEqual(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = char, typename OT>
	Tensor<RT, device> lessThanEqualSingle(const OT& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> greaterThanEqual(const Tensor<OT, o_device>& other) const;

	template<typename RT = char, typename OT, Mode o_device>
	Tensor<RT, device> greaterThanEqual(const TensorSlice<OT, o_device>& other) const;

	template<typename RT = char, typename OT>
	Tensor<RT, device> greaterThanEqualSingle(const OT& other) const;

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
		return compareSingle(other).template sum<size_t>() == other.size();
		#else
		return compareSingle(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator!=(Tensor<OT, o_device>& other)
	{
		
		#ifdef __clang__
		return !(bool)compare(other).template sum<size_t>() == other.size();
		#else
		return !(bool)compare(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator!=(TensorSlice<OT, o_device>& other)
	{
		#ifdef __clang__
		return !(bool)compare(other).template sum<size_t>() == other.size();
		#else
		return !(bool)compare(other).sum<size_t>() == other.size();
		#endif
	}


	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator!=(const OT& other)
	{
		#ifdef __clang__
		return !(bool)compareSingle(other).template sum<size_t>() == other.size();
		#else
		return !(bool)compareSingle(other).sum<size_t>() == other.size();
		#endif
	}


	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator<(Tensor<OT, o_device>& other)
	{
		#ifdef __clang__
		return (bool)lessThan(other).template sum<size_t>() == other.size();
		#else
		return (bool)lessThan(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator<(TensorSlice<OT, o_device>& other)
	{
		#ifdef __clang__
		return (bool)lessThan(other).template sum<size_t>() == other.size();
		#else
		return (bool)lessThan(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator<(const OT& other)
	{
		#ifdef __clang__
		return (bool)lessThanSingle(other).template sum<size_t>() == other.size();
		#else
		return (bool)lessThanSingle(other).sum<size_t>() == other.size();
		#endif
	}


	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator>(Tensor<OT, o_device>& other)
	{
		#ifdef __clang__
		return (bool)moreThan(other).template sum<size_t>() == other.size();
		#else
		return (bool)moreThan(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator>(TensorSlice<OT, o_device>& other)
	{
		#ifdef __clang__
		return (bool)moreThan(other).template sum<size_t>() == other.size();
		#else
		return (bool)moreThan(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator>(const OT& other)
	{
		#ifdef __clang__
		return (bool)moreThanSingle(other).template sum<size_t>() == other.size();
		#else
		return (bool)moreThanSingle(other).sum<size_t>() == other.size();
		#endif

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator<=(Tensor<OT, o_device>& other)
	{
		
		#ifdef __clang__
		return (bool)lessThanEqual(other).template sum<size_t>() == other.size();
		#else
		return (bool)lessThanEqual(other).sum<size_t>() == other.size();
		#endif

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator<=(TensorSlice<OT, o_device>& other)
	{
		#ifdef __clang__
		return (bool)lessThanEqual(other).template sum<size_t>() == other.size();
		#else
		return (bool)lessThanEqual(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator<=(const OT& other)
	{
		#ifdef __clang__
		return (bool)lessThanEqualSingle(other).template sum<size_t>() == other.size();
		#else
		return (bool)lessThanEqualSingle(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator>=(Tensor<OT, o_device>& other)
	{
		
		#ifdef __clang__
		return (bool)moreThanEqual(other).template sum<size_t>() == other.size();
		#else
		return (bool)moreThanEqual(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator>=(TensorSlice<OT, o_device>& other)
	{

		#ifdef __clang__
		return (bool)moreThanEqual(other).template sum<size_t>() == other.size();
		#else
		return (bool)moreThanEqual(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline bool operator>=(const OT& other)
	{
		
		#ifdef __clang__
		return (bool)moreThanEqualSingle(other).template sum<size_t>() == other.size();
		#else
		return (bool)moreThanEqualSingle(other).sum<size_t>() == other.size();
		#endif
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator+(Tensor<OT, o_device>& other)
	{
		
		return add(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator+(const TensorSlice<OT, o_device>& other)
	{

		return add(other);

	}


	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator+(const OT& other)
	{
		
		return addSingle(other);

	}
	
	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator-(Tensor<OT, o_device>& other)
	{
		
		return subtract(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator-(const TensorSlice<OT, o_device>& other)
	{

		return subtract(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator-(const OT& other)
	{
		
		return subtractSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator*(Tensor<OT, o_device>& other)
	{
		
		return multiply(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator*(const TensorSlice<OT, o_device>& other)
	{

		return multiply(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator*(const OT& other)
	{
		
		return multiplySingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator/(Tensor<OT, o_device>& other)
	{
		
		return divide(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator/(const TensorSlice<OT, o_device>& other)
	{

		return divide(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator/(const OT& other)
	{
		
		return divideSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator%(Tensor<OT, o_device>& other)
	{
		
		return modulous(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator%(const TensorSlice<OT, o_device>& other)
	{

		return modulous(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline Tensor<T, device> operator%(const OT& other)
	{
		
		return modulousSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator+=(Tensor<OT, o_device>& other)
	{
		
		additionAssignment(other);

	}
	
	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator+=(const TensorSlice<OT, o_device>& other)
	{

		additionAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator+=(const OT& other)
	{
		
		additionAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator-=(Tensor<OT, o_device>& other)
	{
		
		subtractionAssignment(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator-=(const TensorSlice<OT, o_device>& other)
	{

		subtractionAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator-=(const OT& other)
	{
		
		subtractionAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator*=(Tensor<OT, o_device>& other)
	{
		
		multiplicationAssignment(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator*=(const TensorSlice<OT, o_device>& other)
	{

		multiplicationAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator*=(const OT& other)
	{
		
		multiplicationAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator/=(Tensor<OT, o_device>& other)
	{
		
		divisionAssignment(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator/=(const TensorSlice<OT, o_device>& other)
	{

		divisionAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator/=(const OT& other)
	{
		
		divisionAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator%=(Tensor<OT, o_device>& other)
	{
		
		modulouAssignment(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator%=(const TensorSlice<OT, o_device>& other)
	{

		modulouAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::CPU, OT>* = nullptr>
	inline void operator%=(const OT& other)
	{
		
		modulouAssignmentSingle(other);

	}


	#ifdef _CUDA
	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator==(Tensor<OT, o_device>& other)
	{
		
		return Ccompare(other).sum<size_t>() == other.size();
	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator==(const OT& other)
	{
		
		return (bool)CcompareSingle(other).sum<size_t>() == other.size();
	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator!=(Tensor<OT, o_device>& other)
	{
		
		return !(bool)Ccompare(other).sum<size_t>() == other.size();

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator!=(const OT& other)
	{
		
		return !(bool)CcompareSingle(other).sum<size_t>() == other.size();

	}


	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator<(Tensor<OT, o_device>& other)
	{
		
		return (bool)ClessThan(other).sum<size_t>() == other.size();

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator<(const OT& other)
	{
		
		return (bool)ClessThanSingle(other).sum<size_t>() == other.size();

	}


	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator>(Tensor<OT, o_device>& other)
	{
		
		return (bool)CmoreThan(other).sum<size_t>() == other.size();

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator>(const OT& other)
	{
		
		return (bool)CmoreThanSingle(other).sum<size_t>() == other.size();

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator<=(Tensor<OT, o_device>& other)
	{
		
		return (bool)ClessThanEqual(other).sum<size_t>() == other.size();

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator<=(const OT& other)
	{
		
		return (bool)ClessThanEqualSingle(other).sum<size_t>() == other.size();

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator>=(Tensor<OT, o_device>& other)
	{
		
		return (bool)CmoreThanEqual(other).sum<size_t>() == other.size();
	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline bool operator>=(const OT& other)
	{
		
		return (bool)CmoreThanEqualSingle(other).sum<size_t>() == other.size();

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator+(Tensor<OT, o_device>& other)
	{
		
		return Cadd(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator+(const OT& other)
	{
		
		return CaddSingle<T, OT>(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator-(Tensor<OT, o_device>& other)
	{
		
		return Csubtract(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator-(const OT& other)
	{
		
		return CsubtractSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator*(Tensor<OT, o_device>& other)
	{
		
		return Cmultiply(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator*(const OT& other)
	{
		
		return CmultiplySingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator/(Tensor<OT, o_device>& other)
	{
		
		return Cdivide(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator/(const OT& other)
	{
		
		return CdivideSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator%(Tensor<OT, o_device>& other)
	{
		
		return Cmodulous(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline Tensor<T, device> operator%(const OT& other)
	{
		
		return CmodulousSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator+=(Tensor<OT, o_device>& other)
	{
		
		CadditionAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator+=(const OT& other)
	{
		
		CadditionAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator-=(Tensor<OT, o_device>& other)
	{
		
		CsubtractionAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator-=(const OT& other)
	{
		
		CsubtractionAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator*=(Tensor<OT, o_device>& other)
	{
		
		CmultiplicationAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator*=(const OT& other)
	{
		
		CmultiplicationAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator/=(Tensor<OT, o_device>& other)
	{
		CdivisionAssignment(other);
	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator/=(const OT& other)
	{
		
		CdivisionAssignmentSingle(other);

	}

	template<typename OT, Mode o_device, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator%=(Tensor<OT, o_device>& other)
	{
		
		CmodulouAssignment(other);

	}

	template<typename OT, typename std::enable_if_t<device == Mode::GPU, OT>* = nullptr>
	inline void operator%=(const OT& other)
	{
		
		CmodulouAssignmentSingle(other);

	}

	operator CUDATensor1D<T>();

	operator CUDATensor2D<T>();

	operator CUDATensor3D<T>();

	#endif


	template<typename CT, Mode o_device>
	operator Tensor<CT, o_device>() const
	{
		Tensor<CT, device> new_Tensor(this->Shape(), CT(), false);

		for (size_t i = 0; i < this->size(); i++)
		{
			new_Tensor[i] = (CT)this->At(i);
		}

		return new_Tensor;
	}

	template<typename CT, Mode o_device>
	operator Tensor<CT, o_device>()
	{
		Tensor<CT, device> new_Tensor(this->Shape(), CT(), false);

		for (size_t i = 0; i < this->size(); i++)
		{
			new_Tensor[i] = (CT)this->At(i);
		}

		return new_Tensor;
	}
};
}
