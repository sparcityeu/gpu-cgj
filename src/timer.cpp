#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace custom {

	// Public

	class Timer {
		private:
			bool done = false;
			std::string name;
			std::chrono::steady_clock::time_point t_start;
			std::chrono::steady_clock::time_point t_end;

		public:
			Timer(std::string name) : name(name) {
				t_start = std::chrono::steady_clock::now();
			}

			void report() {
				if (!done) {
					t_end = std::chrono::steady_clock::now();
					done = true;
				}
				std::cout << name << " took " << seconds() << " seconds." << std::endl;
			}

			double seconds() {
				return std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
			}
	};
}
