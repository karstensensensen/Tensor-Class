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
				size_t dimensions = 0;
				size_t size = 0;
				size_t buffer_size = 0;
				char* buffer = nullptr;

				bool is_open = false;
				
				void write_header();

		public:

				otnsr_sequence(std::string path);
				~otnsr_sequence();

				void begin_sequence(const std::vector<size_t>& storage_shape, size_t buf_size = 8192);
				template<Device device>
				void begin_sequence(const Tensor<T, device>& base, size_t buf_size = 8192);

				void reset_sequence();

				template<Device device>
				void append(const Tensor<T, device>& source);

				void open_sequence(size_t buf_size = 8192);
				template<Device device>
				void open_sequence(const Tensor<T, device>& base, size_t buf_size = 8192);
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

			void skip(size_t amount = 1);

			void open();
			void close(); 

		};
	}
}

#include "SaveSequence.ipp"
