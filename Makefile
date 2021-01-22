RM = rm -rf
APP = main
SRCS = $(APP).cpp
OBJS = $(subst .cpp,.o,$(SRCS))
#Include opencv
CXXFLAGS += -I/usr/include/opencv4
# Get the libs
LDLIBS += $(shell pkg-config --libs opencv4)

$(APP): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDLIBS)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(APP)
