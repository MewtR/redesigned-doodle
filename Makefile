RM = rm -rf

SRCS = main.cpp
OBJS = $(subst .cpp,.o,$(SRCS))
#Include opencv
CXXFLAGS += -I/usr/include/opencv4
# Get the libs
LDLIBS += $(shell pkg-config --libs opencv4)

main: $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDLIBS)
#implicit rule: make understands that main.cpp is a prereq and 
# knows the g++ command to compile it
# We can even omit the following line 
# and it will still work because main.o is a dependency of main
#main.o: 

.PHONY: clean
clean:
	$(RM) $(OBJS) main
