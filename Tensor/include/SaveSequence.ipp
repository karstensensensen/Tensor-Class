#include "SaveSequence.h"

template<typename T>
TSlib::Tools::otnsr_sequence<T>::otnsr_sequence(std::string dir, std::vector<size_t> storage_shape, size_t buffer_size)
	: dir(dir + ".tnsrs"), out_file(dir, std::ios::app | std::ios::binary), shape(storage_shape), dimensions(shape.size()), buffer_size(buffer_size), buffer(new char[buffer_size])
{

	out_file.rdbuf()->pubsetbuf(buffer, buffer_size);

	if (!std::filesystem::exists(dir))
	{
		write_header();
	}

	size = 1;

	for (size_t& dim : shape)
	{
		size *= dim;
	}
}

template<typename T>
template<TSlib::Device device>
TSlib::Tools::otnsr_sequence<T>::otnsr_sequence(std::string dir, const Tensor<T, device>& base, size_t buffer_size)
	: dir(dir + ".tnsrs"), out_file(dir, std::ios::app | std::ios::binary), shape(base.Shape()), dimensions(shape.size()), buffer_size(buffer_size), buffer(new char[buffer_size])
{
	out_file.rdbuf()->pubsetbuf(buffer, buffer_size);

	if (!std::filesystem::exists(dir))
	{
		write_header();
	}

	size = 1;

	for (size_t& dim : shape)
	{
		size *= dim;
	}
}

template<typename T>
TSlib::Tools::otnsr_sequence<T>::~otnsr_sequence()
{
	close();
}

template<typename T>
void TSlib::Tools::otnsr_sequence<T>::write_header()
{
	std::filesystem::create_directories(dir.parent_path());

	out_file.write((char*)&dimensions, sizeof(dimensions));

	out_file.write((const char*)shape.data(), sizeof(size_t) * dimensions);

	size_t data_size = sizeof(T);

	out_file.write((char*)&data_size, sizeof(data_size));
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
	#endif

	out_file.write((char*)source.Data(), sizeof(T) * size);
}

template<typename T>
void TSlib::Tools::otnsr_sequence<T>::close()
{
	out_file.close();
}
