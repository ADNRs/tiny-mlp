CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -fno-math-errno -fno-trapping-math \
					 -ffp-contract=fast -funroll-loops -fconstexpr-steps=100000000 \
					 -Wall -Wextra -Wpedantic

all:
	$(CXX) $(CXXFLAGS) -I../ -o matMul matMul.cpp

clean:
	rm -rf matMul
