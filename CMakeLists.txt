# CMakeFiles.txt for the minc2-simple
#
# Vladimir S. FONOV - vladimir.fonov@gmail.com

cmake_minimum_required(VERSION 3.5...4.2 FATAL_ERROR)

PROJECT(minc2-simple)
INCLUDE(CTest)

FIND_PATH( MINC_TOOLKIT_DIR bin/mincstats  HINTS $ENV{MINC_TOOLKIT} DOC "minc-toolkit install prefix")
FIND_PACKAGE( LIBMINC REQUIRED HINTS ${MINC_TOOLKIT_DIR}/lib${LIB_SUFFIX})

# Resolve the libminc target name:
#  - installed libminc provides LIBMINC::minc2 (via NAMESPACE in install(EXPORT))
#  - build-tree libminc provides minc2 (no namespace)
if(TARGET LIBMINC::minc2)
  set(LIBMINC_TARGET LIBMINC::minc2)
elseif(TARGET minc2)
  set(LIBMINC_TARGET minc2)
else()
  message(FATAL_ERROR "LIBMINC found but neither LIBMINC::minc2 nor minc2 target exists")
endif()

OPTION(BUILD_EXAMPLES "Build examples." OFF)

OPTION(BUILD_MINC2_SIMPLE_SHARED "Build shared library" ON)
OPTION(BUILD_MINC2_SIMPLE_STATIC "Build static library" ON)


ENABLE_TESTING()

SET(CMAKE_MACOSX_RPATH ON)
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})


ADD_SUBDIRECTORY(src)

if(BUILD_EXAMPLES)
  ADD_SUBDIRECTORY(example)
endif(BUILD_EXAMPLES)

IF(BUILD_TESTING)
    ADD_SUBDIRECTORY(test)
ENDIF(BUILD_TESTING)

#kate: indent-width 2; replace-tabs on; 
