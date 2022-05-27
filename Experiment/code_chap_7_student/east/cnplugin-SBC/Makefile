CXXFLAGS+= -I . -I ${NEUWARE}/include -std=c++11 -g  -D__BANG_ARCH__=270 -D__DEBUG
LDFLAGS+= -L ${NEUWARE}/lib64  -Wl,-rpath=${NEUWARE}/lib64 -lcnrt  -lcnml -lpthread

CPP_SRCS=$(wildcard *.cpp)
CPP_OBJS=$(CPP_SRCS:%.cpp=%.o)
  
MLU_SRCS=$(wildcard *.mlu)
MLU_OBJS=$(MLU_SRCS:%.mlu=%.o)

OBJS=$(CPP_OBJS) $(MLU_OBJS)

TARGET=test

all:$(TARGET)

$(TARGET):$(OBJS)
	$(CXX) -O3 -o $@ -g $(OBJS) $(LDFLAGS)

%.o : %.cpp
	g++ $(CXXFLAGS) -c $^ -o $@

%.o : %.mlu
	cncc -c $^ -o $@  -O2 --bang-mlu-arch=MLU270 -g -D__DEBUG	
	
clean:
	rm -f $(TARGET) $(OBJS) mluoutput.txt
