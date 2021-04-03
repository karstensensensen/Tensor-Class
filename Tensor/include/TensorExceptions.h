#pragma once

#include <exception>
#include <string>

namespace TSlib
{
	/*
	* NOTE: Exceptions will only be thrown in DEBUG mode
	*/

	template<typename T>
	struct ExceptValue
	{
		std::string msg;
		T val;

		ExceptValue(std::string msg, T val)
			: msg(msg), val(val)
		{}
	};

	/// <summary>
	/// All Tensor function exceptions
	/// </summary>

	class OutOfBounds : public std::exception
	{
		std::string err_msg = "Out of bounds error was thrown, without additional information";

	public:

		OutOfBounds()
		{}

		OutOfBounds(std::vector<size_t> shape, std::string message, size_t dim, size_t index)
		{
			err_msg = message + "\nTarget dimension: " + std::to_string(dim + 1) + " Target Index: " + std::to_string(index);

			err_msg += "\nTensor bounds was: ";

			for (size_t i = 0; i < shape.size() - 1; i++)
			{
				err_msg += std::to_string(shape[i]) + ' ';
			}
		}

		virtual const char* what() const throw()
		{
			return err_msg.c_str();
		}
	};

	class BadShape : public std::exception
	{
		std::string err_msg;

	public:
		BadShape()
		{
			err_msg = "Bad shape exception was thrown, wihtout additional information";
		}

		template<typename T, Device device>
		BadShape(Tensor<T, device>* tensor, std::string message, const std::vector<size_t>& shape)
		{
			err_msg = message + "\nTarget shape was: ";
			size_t target_shape_sum = 1;

			for (const size_t& shape_elem : shape)
			{
				err_msg += std::to_string(shape_elem) + ' ';
				target_shape_sum *= shape_elem;
			}

			err_msg += "\nTensor shape was: ";

			for (const size_t& shape_elem : tensor->Shape())
			{
				err_msg += std::to_string(shape_elem) + ' ';
			}

			err_msg += "\nTarget shape sum was: " + std::to_string(target_shape_sum);
			err_msg += "\nTensor shape sum was: " + std::to_string(tensor->size());
		}

		template<typename T, Device device>
		BadShape(Tensor<T, device>* tensor, std::string message, const std::vector<TSlice>& shape)
		{
			err_msg = message + "\nTarget shape was: ";
			size_t target_shape_sum = 1;

			for (const TSlice& shape_elem : shape)
			{
				err_msg += std::to_string(shape_elem.width()) + ' ';
				target_shape_sum *= shape_elem.width();
			}

			err_msg += "\nTensor shape was: ";

			for (const size_t& shape_elem : tensor->Shape())
			{
				err_msg += std::to_string(shape_elem) + ' ';
			}

			err_msg += "\nTarget shape sum was: " + std::to_string(target_shape_sum);
			err_msg += "\nTensor shape sum was: " + std::to_string(tensor->size());
		}

		template<typename T1, typename T2>
		BadShape(std::string message, const std::vector<T1>& target, const std::vector<T2>& expected)
		{
			err_msg = message + "\nGotten shape was: ";
			size_t target_shape_sum = 1;
			size_t expected_shape_sum = 1;

			for (const T1& shape_elem : target)
			{
				err_msg += std::to_string(shape_elem) + ' ';
				target_shape_sum *= shape_elem;
			}

			err_msg += "\nExpected shape was: ";

			for (const T2& shape_elem : expected)
			{
				err_msg += std::to_string(shape_elem) + ' ';
				expected_shape_sum *= shape_elem;
			}

			err_msg += "\nGotten shape sum was: " + std::to_string(target_shape_sum);
			err_msg += "\nExpected shape sum was: " + std::to_string(expected_shape_sum);
		}

		template<typename T>
		BadShape(std::string message, const std::vector<TSlice>& target, const std::vector<T>& expected)
		{
			err_msg = message + "\nGotten shape was: ";
			size_t target_shape_sum = 1;
			size_t expected_shape_sum = 1;

			for (const TSlice& shape_elem : target)
			{
				err_msg += std::to_string(shape_elem.width()) + ' ';
				target_shape_sum *= shape_elem.width();
			}

			err_msg += "\nExpected shape was: ";

			for (const T& shape_elem : expected)
			{
				err_msg += std::to_string(shape_elem) + ' ';
				expected_shape_sum *= shape_elem;
			}

			err_msg += "\nGotten shape sum was: " + std::to_string(target_shape_sum);
			err_msg += "\nExpected shape sum was: " + std::to_string(expected_shape_sum);
		}

		BadShape(std::string message, const std::initializer_list<size_t>& shape)
		{
			err_msg = message;

			for (const size_t& shape_elem : shape)
			{
				err_msg += std::to_string(shape_elem) + ", ";
			}
			err_msg += '\n';
		}

		BadShape(std::string message, const std::vector<size_t>& shape)
		{
			err_msg = message;

			for (const size_t& shape_elem : shape)
			{
				err_msg += std::to_string(shape_elem) + ", ";
			}
			err_msg += '\n';
		}

		BadShape(std::string message, const std::initializer_list<double>& shape)
		{
			err_msg = message;

			for (const double& shape_elem : shape)
			{
				err_msg += std::to_string(shape_elem) + ", ";
			}
			err_msg += '\n';
		}

		BadShape(std::string message, const std::vector<double>& shape)
		{
			err_msg = message;

			for (const double& shape_elem : shape)
			{
				err_msg += std::to_string(shape_elem) + ", ";
			}
			err_msg += '\n';
		}

		BadShape(std::string message)
		{
			err_msg = message;
		}

		virtual const char* what() const throw()
		{
			return err_msg.c_str();
		}
	};

	class BadType : public std::exception
	{
		std::string err_msg;

	public:
		BadType()
		{
			err_msg = "BadType exception was thrown, wihtout additional information";
		}

		BadType(std::string Target, std::string Argument)
		{
			err_msg = "type must be: " + Target + ", type gotten was: " + Argument + '\n';
		}

		BadType(std::string Target)
		{
			err_msg = "type must be: " + Target + ", type gotten was invalid";
		}

		virtual const char* what() const throw()
		{
			return err_msg.c_str();
		}
	};

	class BadValue : public std::exception
	{
		std::string err_msg;

		template<typename First, typename ... Args>
		void append_info(ExceptValue<First> first, ExceptValue<Args> ... rest)
		{
			append_info(first);
			append_info(rest...);
		}

		template<typename First>
		void append_info(ExceptValue<First> first)
		{
			err_msg += std::string("\n") + first.msg + ": " + std::to_string(first.val);
		}

	public:

		template<typename ... Args>
		BadValue(std::string message, ExceptValue<Args> ... args)
			: err_msg(message)
		{
			//append_info(args...);
		}

		template<typename T>
		BadValue(std::string message, T gotten_value)
			: err_msg(message)
		{
			err_msg += std::string("\nGot: ") + std::to_string(gotten_value);
		}

		BadValue(std::string message)
			: err_msg(message)
		{}

		virtual const char* what() const throw()
		{
			return err_msg.c_str();
		}
	};

	/// <summary>
	/// All Tensor cuda function exceptions
	/// </summary>

	#ifdef _TS_CUDA
	class BadThreadTarget : public std::exception
	{
		unsigned int threads;
		std::string err_msg;

	public:
		BadThreadTarget(unsigned int threads)
			:threads(threads)
		{
			err_msg = "Tensor target was set to a value larger than 1024 or the value was not a multiple of 32 (" + std::to_string(threads) + ')';
		}

		BadThreadTarget(std::string message, unsigned int threads)
			:threads(threads)
		{
			err_msg = message + '(' + std::to_string(threads) + ')';
		}

		virtual const char* what() const throw()
		{
			return err_msg.c_str();
		}
	};
	#endif
}
