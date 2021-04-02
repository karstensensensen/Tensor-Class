#include "SaveSequence.h"

namespace TSlib
{
	namespace Tools
	{
	template<typename T>
	otnsr_sequence<T>::otnsr_sequence(std::string path)
		: dir(path)
	{
		if (!dir.has_extension())
		{
			dir += ".tnsrs";
		}
	}

	template<typename T>
	otnsr_sequence<T>::~otnsr_sequence()
	{
		if (is_open)
		{
			close();
		}
	}

	template<typename T>
	void otnsr_sequence<T>::begin_sequence(const std::vector<size_t>&storage_shape, size_t buf_size)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (is_open)
		{
			throw std::runtime_error("Directory is already open\n");
		}

		if (!std::filesystem::exists(dir))
		{
			write_header();
		}
		#endif

		is_open = true;

		buffer_size = buf_size;

		buffer = new char[buffer_size];

		out_file.rdbuf()->pubsetbuf(buffer, buffer_size);
		out_file.open(dir, std::ios::binary | std::ios::trunc);

		dimensions = storage_shape.size();
		shape = storage_shape;

		write_header();

		size = 1;

		for (size_t& dim : shape)
		{
			size *= dim;
		}
	}

	template<typename T>
	template<Device device>
	void otnsr_sequence<T>::begin_sequence(const Tensor<T, device>&base, size_t buf_size)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (is_open)
		{
			throw std::runtime_error("Directory is already open\n");
		}
		#endif

		is_open = true;

		buffer_size = buf_size;

		buffer = new char[buffer_size];

		out_file.rdbuf()->pubsetbuf(buffer, buffer_size);
		out_file.open(dir, std::ios::binary | std::ios::trunc);

		dimensions = base.Dims();
		shape = base.Shape();

		write_header();

		size = 1;

		for (size_t& dim : shape)
		{
			size *= dim;
		}
	}

	template<typename T>
	void otnsr_sequence<T>::write_header()
	{
		if (dir.has_parent_path())
		{
			std::filesystem::create_directories(dir.parent_path());
		}

		out_file.write((char*)&dimensions, sizeof(size_t));

		out_file.write((const char*)shape.data(), sizeof(size_t) * dimensions);

		size_t data_size = sizeof(T);

		out_file.write((char*)&data_size, sizeof(size_t));
	}

	template<typename T>
	void otnsr_sequence<T>::reset_sequence()
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("Directory must be open before sequence can be reset");
		}
		#endif

		out_file.close();

		out_file.open(dir, std::ios::binary | std::ios::trunc);

		write_header();
	}

	template<typename T>
	template<Device device>
	void otnsr_sequence<T>::append(const Tensor<T, device>&source)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("Directory is not open\n");
		}

		for (size_t i = 0; i < dimensions; i++)
		{
			if (shape[i] != source.Shape()[i])
			{
				throw BadShape("The source tensor must have the same shape as the storage shape", shape, source.Shape());
			}
		}
		#endif

		out_file.write((const char*)source.Data(), sizeof(T) * size);
	}

	template<typename T>
	template<Device device>
	void otnsr_sequence<T>::append_seq(const Tensor<T, device>&source_seq)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("Directory is not open\n");
		}

		if (source_seq.Dims() != dimensions + 1)
		{
			throw BadShape("Source Tensor must have one more dimension than the stored Tensor", source_seq.Shape(), shape);
		}

		for (size_t i = 1; i < dimensions + 1; i++)
		{
			if (shape[i - 1] != source_seq.Shape()[i])
			{
				throw BadShape("Destination Tensor must have the same shape as the stored Tensor, except for the last dimension", source_seq.Shape(), shape);
			}
		}
		#endif

		out_file.write((char*)source_seq.Data(), sizeof(T) * size * source_seq.Shape()[0]);
	}

	template<typename T>
	void otnsr_sequence<T>::open_sequence(size_t buf_size)
	{
		#ifdef _TS_NO_FILE_CHECK
		if (is_open)
		{
			throw std::runtime_error("Sequence is already open\nDirectory: \"" + dir.string() + "\"\n");
		}

		if (!std::filesystem::exists(dir))
		{
			throw std::runtime_error("The directory \"" + dir.string() + "\" Does not exist\nTo create the directory, run the \'begin_sequence\' function\n");
		}
		#endif

		is_open = true;

		buffer_size = buf_size;
		buffer = new char[buffer_size];

		out_file.rdbuf()->pubsetbuf(buffer, buffer_size);
		out_file.open(dir, std::ios::out | std::ios::binary | std::ios::app);

		std::ifstream in_file(dir);

		in_file.read((char*)&dimensions, sizeof(size_t));

		shape.resize(dimensions);

		in_file.read((char*)shape.data(), sizeof(size_t) * dimensions);

		size_t data_size;

		in_file.read((char*)&data_size, sizeof(size_t));

		#ifdef _TS_NO_FILE_CHECK
		if (data_size != sizeof(T))
		{
			throw BadType("The data size of the sequence and the file must be the same\nGot: " + std::to_string(data_size) + " Expected: " + std::to_string(sizeof(T)) + "\n");
		}
		#endif

		size = 1;

		for (size_t& dim : shape)
		{
			size *= dim;
		}
	}

	template<typename T>
	template<Device device>
	void otnsr_sequence<T>::open_sequence(const Tensor<T, device>&base, size_t buf_size)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (is_open)
		{
			throw std::runtime_error("Sequence is already open\nDirectory: \"" + dir.string() + "\"\n");
		}
		#endif

		if (!std::filesystem::exists(dir)) [[unlikely]]
		{
			begin_sequence(base, buf_size);
		}
		else [[likely]]
		{
			is_open = true;

			buffer_size = buf_size;
			buffer = new char[buffer_size];

			out_file.rdbuf()->pubsetbuf(buffer, buffer_size);
			out_file.open(dir, std::ios::out | std::ios::binary | std::ios::app);

			std::ifstream in_file(dir);

			in_file.read((char*)&dimensions, sizeof(size_t));

			shape.resize(dimensions);

			in_file.read((char*)shape.data(), sizeof(size_t)* dimensions);

			size_t data_size;

			in_file.read((char*)&data_size, sizeof(size_t));

			#ifndef _TS_NO_FILE_CHECK
			if (data_size != sizeof(T))
			{
				throw BadType("The data size of the sequence and the file must be the same\nGot: " + std::to_string(data_size) + " Expected: " + std::to_string(sizeof(T)) + "\n");
			}
			#endif

			size = 1;

			for (size_t& dim : shape)
			{
				size *= dim;
			}
		}
	}

	template<typename T>
	void otnsr_sequence<T>::close()
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("Directory is already closed \n");
		}
		#endif

		is_open = false;
		out_file.close();
		delete[] buffer;
	}

	template<typename T>
	itnsr_sequence<T>::itnsr_sequence(std::string path)
		: dir(path)
	{
		if (!dir.has_extension())
		{
			dir += ".tnsrs";
		}

		#ifndef _TS_NO_FILE_CHECK
		if (!std::filesystem::exists(dir))
		{
			throw std::runtime_error("The directory \"" + dir.string() + "\" does not exist\n");
		}
		#endif
	}

	template<typename T>
	template<Device device>
	void itnsr_sequence<T>::read(Tensor<T, device>& dest)
	{
		if (!is_open)
		{
			throw std::runtime_error("The file must be open before data can be read");
		}

		#ifndef _TS_NO_FILE_CHECK
		if (dest.Dims() != dimensions)
		{
			throw BadShape("Destination Tensor must have the same shape as the stored Tensor", dest.Shape(), shape);
		}

		for (size_t i = 0; i < dimensions; i++)
		{
			if (shape[i] != dest.Shape()[i])
			{
				throw BadShape("Destination Tensor must have the same shape as the stored Tensor", dest.Shape(), shape);
			}
		}

		if (!length)
		{
			throw std::runtime_error("Cannot read more of the sequence!\nEnd of sequence reached!");
		}

		#endif

		length--;

		in_file.read((char*)dest.Data(), sizeof(T) * size);
	}

	template<typename T>
	template<Device device>
	Tensor<T, device> itnsr_sequence<T>::read()
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("The file must be open before data can be read");
		}

		if (!length)
		{
			throw std::runtime_error("Cannot read more of the sequence!\nEnd of sequence reached!");
		}

		#endif

		length--;

		Tensor<T, device> result(shape);

		in_file.read((char*)result.Data(), sizeof(T) * size);

		return result;
	}

	template<typename T>
	template<Device device>
	void itnsr_sequence<T>::read_seq(Tensor<T, device>& dest)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("The file must be open before data can be read");
		}

		if (dest.Dims() != dimensions + 1)
		{
			throw BadShape("Destination Tensor must have one more dimension than the stored Tensor", dest.Shape(), shape);
		}

		for (size_t i = 1; i < dimensions + 1; i++)
		{
			if (shape[i - 1] != dest.Shape()[i])
			{
				throw BadShape("Destination Tensor must have the same shape as the stored Tensor, except for the last dimension", dest.Shape(), shape);
			}
		}

		if (length < dest.Shape()[0])
		{
			throw std::runtime_error("sequence is not long enough for the destination tensor!");
		}
		#endif

		length -= dest.Shape()[0];

		in_file.read((char*)dest.Data(), sizeof(T) * size * dest.Shape()[0]);
	}

	template<typename T>
	template<Device device>
	Tensor<T, device> itnsr_sequence<T>::read_seq(size_t seq_length)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("The file must be open before data can be read");
		}

		if (length < seq_length)
		{
			throw std::runtime_error("sequence is not long enough for the destination tensor!");
		}
		#endif

		std::vector<size_t> seq_shape(dimensions + 1);

		memcpy(seq_shape.data() + 1, shape.data(), sizeof(size_t) * dimensions);

		seq_shape[0] = seq_length;

		Tensor<T, device> seq(seq_shape);

		length -= seq_length;

		in_file.read((char*)seq.Data(), sizeof(T) * size * seq_length);

		return seq;
	}

	template<typename T>
	void itnsr_sequence<T>::skip(size_t amount)
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("The file must be open before data can be read");
		}

		if (amount > length)
		{
			throw std::runtime_error("Cannot skip more of the sequence than what is left\nSequence length: " + std::to_string(length) + "\nSkip amount: " + std::to_string(amount));
		}
		#endif

		in_file.ignore(size * sizeof(T) * amount);
		length -= amount;
	}

	template<typename T>
	void itnsr_sequence<T>::open()
	{
		#ifndef _TS_NO_FILE_CHECK
		if (is_open)
		{
			throw std::runtime_error("Sequence is already open\nDirectory: \"" + dir.string() + "\"\n");
		}
		#endif

		is_open = true;

		in_file.open(dir, std::ios::binary);

		in_file.read((char*)&dimensions, sizeof(size_t));

		shape.resize(dimensions);

		in_file.read((char*)shape.data(), sizeof(size_t) * dimensions);

		size_t data_size;

		in_file.read((char*)&data_size, sizeof(size_t));

		#ifndef _TS_NO_FILE_CHECK
		if (data_size != sizeof(T))
		{
			throw BadType("The data size of the sequence and the file must be the same\nGot: " + std::to_string(data_size) + " Expected: " + std::to_string(sizeof(T)) + "\n");
		}
		#endif

		size = 1;

		for (size_t& dim : shape)
		{
			size *= dim;
		}

		length = std::filesystem::file_size(dir);

		//subtract header length
		length -= (dimensions + 2) * sizeof(size_t);

		//divide by the type size to get the number of elements instead of bytes
		length /= sizeof(T) * size;
	}

	template<typename T>
	void itnsr_sequence<T>::close()
	{
		#ifndef _TS_NO_FILE_CHECK
		if (!is_open)
		{
			throw std::runtime_error("Sequence is already closed\nDirectory: \"" + dir.string() + "\"\n");
		}
		#endif

		is_open = false;

		in_file.close();
	}

	template<typename T>
	size_t itnsr_sequence<T>::GetLength()
	{
		return length;
	}
}
}
