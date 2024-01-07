SRC=./src

SRCS = $(SRC)/main.cpp \
	$(SRC)/params.cpp \
	$(SRC)/util.cpp \
	 
OBJSDIR = ./obj

EXE = ./exe/train

CFLAGS = -O3 -Wall -std=c++17 -Wno-unused-local-typedefs

LIBLIST = boost_program_options
#	boost_filesystem \
#	boost_regex \
#	boost_system \
#    boost_unit_test_framework \
#	opencv_features2d \
#	opencv_highgui \
#	opencv_imgproc \
#	opencv_core \
#	opencv_imgcodecs \
#	stdc++ \
#	assimp \
#    OSMesa \
#    GLdispatch \
#	pcl_features \
#	pcl_kdtree \
#	pcl_visualization \
#	pcl_common \
#	pcl_filters \
#	pcl_search \
#    GLU \

#LIBDIRLIST = /usr/local/opencv/build/lib \
#	/usr/local/boost/lib \
#    /usr/lib64
			
#DEFLIST = NDEBUG \
#	MESSAGE_ON=0 \
#	UNIT_TEST=0

#INCLUDELIST = /usr/include/opencv4/ \
#	/usr/include/boost\
#	/usr/include/pcl-1.12 \
#	/usr/include/eigen3/ \
#    /usr/local/cpplinq/CppLinq

#TESTS = $(patsubst %, -D%, $(TESTLIST))
INCLUDES = $(patsubst %, -I%, $(INCLUDELIST))
DEFS = $(patsubst %, -D%, $(DEFLIST))

OBJS = $(patsubst %.cpp, $(OBJSDIR)/%.o, $(SRCS))
LIBS = $(patsubst %, -l%, $(LIBLIST)) 
LIBDIR = $(patsubst %, -L%, $(LIBDIRLIST))

# make rules -----------------------------------------------------------

$(EXE): $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)  

$(OBJSDIR)/%.o: %.cpp
	$(CXX) $(INCLUDES) $(DEFS) $(TESTS) $(CFLAGS) -o $@ -c $<

# dependencies ---------------------------------------------------------

$(OBJSDIR)/main.o: $(SRC)/params.h
	$(SRC)/util.h
$(OBJSDIR)/params.o: $(SRC)/params.h
$(OBJSDIR)/util.o: $(SRC)/util.h
#
#$(OBJSDIR)/$(SRC)/Viewer.o: $(SRC)/Viewer.h \
#	$(SRC)/ClientStateCapability.h \
#	$(SRC)/GeometryDescriptor.h
#
#$(OBJSDIR)/$(SRC)/ObjectLoader.o: $(SRC)/ObjectLoader.h \
#	$(SRC)/Message.h \
#	$(SRC)/common.h
#
#$(OBJSDIR)/$(SRC)/main.o: $(SRC)/ObjectLoader.h \
#	$(SRC)/common.h \
#	$(SRC)/Viewer.h \
#	$(SRC)/RangeImageGeneratorParameters.h
#
#$(OBJSDIR)/$(SRC)/Message.o: $(SRC)/Message.h

.PHONY: clean
clean:
	 $(RM) $(OBJS) $(EXE) 

load:
	./$(EXE) 
