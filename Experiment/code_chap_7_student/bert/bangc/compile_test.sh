g++ test_int_matmul.cc \
  -I ${NEUWARE}/include  \
  -L ${NEUWARE}/lib64 \
  -o ./matmul_test -lcnml -lcnrt -lcnplugin -lopenblas --std=c++11
