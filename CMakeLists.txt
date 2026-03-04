# CMakeFiles.txt for the minc2-simple
#
# Vladimir S. FONOV - vladimir.fonov@gmail.com

cmake_minimum_required(VERSION 3.5...4.2 FATAL_ERROR)

PROJECT(minc2-simple)
INCLUDE(CTest)

# ── Locate libminc ─────────────────────────────────────────────────────
# Checked in order:
#   1. LIBMINC_DIR       – path to a libminc build or install tree
#                          (directory that contains LIBMINCConfig.cmake,
#                           or a prefix whose lib/cmake/ contains it)
#   2. MINC_TOOLKIT_DIR  – path to a full MINC Toolkit install
#                          (auto-detected from $MINC_TOOLKIT env var)
#   3. Default cmake search paths

set(_libminc_hints "")

# LIBMINC_DIR: accept a build tree, an install prefix, or the exact
# cmake config directory.
if(DEFINED LIBMINC_DIR)
  list(APPEND _libminc_hints
    "${LIBMINC_DIR}"
    "${LIBMINC_DIR}/lib"
    "${LIBMINC_DIR}/lib/cmake"
  )
endif()

# Backward-compatible MINC_TOOLKIT_DIR detection
FIND_PATH( MINC_TOOLKIT_DIR bin/mincstats  HINTS $ENV{MINC_TOOLKIT} DOC "minc-toolkit install prefix")
if(MINC_TOOLKIT_DIR)
  list(APPEND _libminc_hints "${MINC_TOOLKIT_DIR}/lib${LIB_SUFFIX}")
endif()

FIND_PACKAGE( LIBMINC REQUIRED HINTS ${_libminc_hints})

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
