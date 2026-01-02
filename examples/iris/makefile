CXX = clang++
CXXFLAGS = -std=c++17 -O3 -march=native -fno-math-errno -fno-trapping-math \
					 -ffp-contract=fast -funroll-loops -Wall -Wextra -Wpedantic

all:
	$(CXX) $(CXXFLAGS) -I../../ -o main main.cpp

clean:
	rm -rf main
