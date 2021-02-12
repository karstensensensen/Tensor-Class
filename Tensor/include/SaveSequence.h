#pragma once
#include "Tensor.h"
#include <filesystem>

namespace TSlib
{
	namespace Tools
	{

		template<size_t gb>
		size_t gigabytes = gb * 1024ULL * 1024ULL * 1024ULL;

		template<size_t mb>
		size_t megabytes = mb * 1024ULL * 1024ULL;

		template<size_t kb>
		size_t kilobytes = kb * 1024ULL;

		template<typename T>
		class otnsr_sequence
		{
				std::filesystem::path dir;
				std::ofstream out_file;
				std::vector<size_t> shape;
				const size_t dimensions;
				size_t size;
				size_t buffer_size;
				char* buffer;
				
				void write_header();

		public:

				otnsr_sequence(std::string dir, std::vector<size_t> storage_shape, size_t buffer_size = 8192);
				template<Device device>
				otnsr_sequence(std::string dir, const Tensor<T, device>& base, size_t buffer_size = 8192);
				~otnsr_sequence();

				void begin_sequence();

				void reset_sequence();

				template<Device device>
				void append(const Tensor<T, device>& source);

				void close();

		};

		template<typename T>
		class itnsr_sequence
		{
			std::ofstream in_file;
			std::vector<size_t> shape;
			size_t length;

		public:

			itnsr_sequence(std::string dir, std::vector<size_t> storage_shape);

			template<Device device>
			itnsr_sequence(std::string dir, const Tensor<T, device>& base);

			template<Device device>
			void read(Tensor<T, device>& source);

			template<Device device = default_device>
			Tensor<T, device> read();

		};
	}
}

#include "SaveSequence.ipp"
