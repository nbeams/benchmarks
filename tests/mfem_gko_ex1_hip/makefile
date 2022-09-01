# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project
# (17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
# organizations (Office of Science and the National Nuclear Security
# Administration) responsible for the planning and preparation of a capable
# exascale ecosystem, including software, applications, hardware, advanced
# system engineering and early testbed platforms, in support of the nation's
# exascale computing imperative.

# Use the MFEM build directory
MFEM_DIR = ../../mfem-serial
SRC =
BLD =
CONFIG_MK = $(or $(wildcard\
 $(MFEM_DIR)/config/config.mk),$(MFEM_DIR)/share/mfem/config.mk)

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

LIBCEED_DIR = 
LIBCEED_LIB = -Wl,-rpath,$(LIBCEED_DIR)/lib -L$(LIBCEED_DIR)/lib -lceed 
LIBCEED_FLAGS = -I$(LIBCEED_DIR)/include
 
GINKGO_DIR ?=
GINKGO_FLAGS ?= -isystem $(GINKGO_DIR)/include -std=c++14
GINKGO_LIB_DIR = $(sort $(dir $(wildcard $(GINKGO_DIR)/lib*/libginkgo*.a  $(GINKGO_DIR)/lib*/libginkgo*.so $(GINKGO_DIR)/lib*/libginkgo*.dylib $(GINKGO_DIR)/lib*/libginkgo*.dll)))
GINKGO_BUILD_TYPE=Release
ALL_GINKGO_LIBS_DEBUG = $(notdir $(basename $(wildcard $(GINKGO_DIR)/lib*/libginkgo*d.a  $(GINKGO_DIR)/lib*/libginkgo*d.so $(GINKGO_DIR)/lib*/libginkgo*d.dylib $(GINKGO_DIR)/lib*/libginkgo*d.dll)))
ALL_GINKGO_LIBS = $(notdir $(basename $(wildcard $(GINKGO_DIR)/lib*/libginkgo*.a  $(GINKGO_DIR)/lib*/libginkgo*.so $(GINKGO_DIR)/lib*/libginkgo*.dylib $(GINKGO_DIR)/lib*/libginkgo*.dll)))
ALL_GINKGO_LIBS_RELEASE = $(filter-out $(ALL_GINKGO_LIBS_DEBUG),$(ALL_GINKGO_LIBS))
GINKGO_LINK = $(subst libginkgo,-lginkgo,$(ALL_GINKGO_LIBS_RELEASE))
ifeq ($(GINKGO_BUILD_TYPE),Debug)
  ifneq (,$(ALL_GINKGO_LIBS_DEBUG))
    GINKGO_LINK = $(subst libginkgo,-lginkgo,$(ALL_GINKGO_LIBS_DEBUG))
  endif
else
endif
GINKGO_LIBS ?= -Wl,-rpath,$(GINKGO_LIB_DIR) -L$(GINKGO_LIB_DIR) $(GINKGO_LINK)

EX1 = $(BLD)ex1

EX1MULT  = $(BLD)ex1-mult

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

filter=--std=c++11
SHORT_MFEM_FLAGS := $(filter-out $(filter), $(MFEM_FLAGS))

SRC.hip = $(SRC)$(wildcard *.hip.cpp)
# Replace the default implicit rule for *.cpp files
$(BLD)%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(LIBCEED_FLAGS) $(GINKGO_FLAGS) $(SHORT_MFEM_FLAGS) $(SRC.hip) $< -o $@ $(MFEM_LIBS) $(GINKGO_LIBS) $(LIBCEED_LIB)

mult: $(EX1MULT)

all: $(EX1)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f *.o *~ $(EX1)
	rm -f *.o *~ $(EX1MULT)
	rm -rf *.dSYM *.TVD.*breakpoints
