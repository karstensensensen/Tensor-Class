#include "SaveSequence.h"

template<typename T>
TSlib::Tools::otnsr_sequence<T>::otnsr_sequence(std::string path)
	: dir(path)
{
	if (!dir.has_extension())
	{
		dir += ".tnsrs";
	}
}

template<typename T>
TSlib::Tools::otnsr_sequence<T>::~otnsr_sequence()
{
	if (is_open)
	{
		close();
	}
}

template<typename T>
void TSlib::Tools::otnsr_sequence<T>::begin_sequence(const std::vector<size_t>& storage_shape, size_t buf_size)
{
	if (!std::filesystem::exists(dir))
	{
		write_header();
	}

	if (is_open)
	{
		throw std::runtime_error("Directory is already open\n");
	}

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
template<TSlib::Device device>
void TSlib::Tools::otnsr_sequence<T>::begin_sequence(const Tensor<T, device>& base, size_t buf_size)
{
	
	if (is_open)
	{
		throw std::runtime_error("Directory is already open\n");
	}

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
void TSlib::Tools::otnsr_sequence<T>::write_header()
{
	std::filesystem::create_directories(dir.parent_path());

	out_file.write((char*)&dimensions, sizeof(size_t));

	out_file.write((const char*)shape.data(), sizeof(size_t) * dimensions);

	size_t data_size = sizeof(T);

	out_file.write((char*)&data_size, sizeof(size_t));
}

template<typename T>
void TSlib::Tools::otnsr_sequence<T>::reset_sequence()
{
	out_file.close();

	out_file.open(dir, std::ios::binary | std::ios::trunc);
	
	write_header();
}

template<typename T>
template<TSlib::Device device>
void TSlib::Tools::otnsr_sequence<T>::append(const Tensor<T, device>& source)
{

	#ifdef _TS_DEBUG
	for (size_t i = 0; i < dimensions; i++)
	{
		if (shape[i] != source.Shape()[i])
		{
			throw BadShape("The source tensor must have the same shape as the storage shape", shape, source.Shape());
		}
	}

	if (!is_open)
	{
		throw std::runtime_error("Directory is not open\n");
	}

	#endif

	out_file.write((const char*)source.Data(), sizeof(T) * size);
}

template<typename T>
void TSlib::Tools::otnsr_sequence<T>::open_sequence(size_t buf_size)
{
	if (is_open)
	{
		throw std::runtime_error("Sequence is already open\nDirectory: \"" + dir.string() + "\"\n");
	}

	if (!std::filesystem::exists(dir))
	{
		throw std::runtime_error("The directory \"" + dir.string() + "\" Does not exist\nTo create the directory, run the \'begin_sequence\' function\n");
	}


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

	if (data_size != sizeof(T))
	{
		throw TSlib::BadType("The data size of the sequence and the file must be the same\nGot: " + std::to_string(data_size) + " Expected: " + std::to_string(sizeof(T)) + "\n");
	}

	size = 1;

	for (size_t& dim : shape)
	{
		size *= dim;
	}
}

template<typename T>
template<TSlib::Device device>
void TSlib::Tools::otnsr_sequence<T>::open_sequence(const Tensor<T, device>& base, size_t buf_size)
{
	if (is_open)
	{
		throw std::runtime_error("Sequence is already open\nDirectory: \"" + dir.string() + "\"\n");
	}

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

		in_file.read((char*)shape.data(), sizeof(size_t) * dimensions);

		size_t data_size;

		in_file.read((char*)&data_size, sizeof(size_t));

		if (data_size != sizeof(T))
		{
			throw TSlib::BadType("The data size of the sequence and the file must be the same\nGot: " + std::to_string(data_size) + " Expected: " + std::to_string(sizeof(T)) + "\n");
		}

		size = 1;

		for (size_t& dim : shape)
		{
			size *= dim;
		}
	}
}

template<typename T>
void TSlib::Tools::otnsr_sequence<T>::close()
{
	if (!is_open)
	{
		throw std::runtime_error("Directory is already closed \n");
	}

	is_open = false;
	out_file.close();
	delete[] buffer;

}

template<typename T>
TSlib::Tools::itnsr_sequence<T>::itnsr_sequence(std::string path)
	: dir(path)
{
	if (!dir.has_extension())
	{
		dir += ".tnsrs";
	}

	if (!std::filesystem::exists(dir))
	{
		throw std::runtime_error("The directory \"" + dir + "\" does not exist\n");
	}

	in_file.open(dir, std::ios::binary);

	in_file.read((char*)&dimensions, sizeof(size_t));

	shape.resize(dimensions);

	in_file.read((char*)shape.data(), sizeof(size_t) * dimensions);

	size_t data_size;

	in_file.read((char*)&data_size, sizeof(size_t));

	if (data_size != sizeof(T))
	{
		throw TSlib::BadType("The data size of the sequence and the file must be the same\nGot: " + std::to_string(data_size) + " Expected: " + std::to_string(sizeof(T)) + "\n");
	}

	size = 1;

	for (size_t& dim : shape)
	{
		size *= dim;
	}
}

template<typename T>
template<TSlib::Device device>
void TSlib::Tools::itnsr_sequence<T>::read(Tensor<T, device>& source)
{
	#ifndef _TS_NO_FILE_CHECK
	if (source.Dims() != dimensions)
	{
		throw TSlib::BadShape("Destination Tensor must have the same shape as the stored Tensor", source.Shape(), shape);
	}
	
	for (size_t i = 0; i < dimensions; i++)
	{
		if (shape[i] != source.Shape()[i])
		{
			throw TSlib::BadShape("Destination Tensor must have the same shape as the stored Tensor", source.Shape(), shape);
		}
	}
	#endif

	in_file.read((char*)source.Data(), sizeof(T) * size);
}

template<typename T>
template<TSlib::Device device>
TSlib::Tensor<T, device> TSlib::Tools::itnsr_sequence<T>::read()
{
	Tensor<T, device> result(shape);

	in_file.read((char*)result.Data(), sizeof(T) * size);

	return result;
}
