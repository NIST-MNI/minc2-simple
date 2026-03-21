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

# Direct hints from MINC_TOOLKIT env var (conda/custom installs without mincstats)
if(DEFINED ENV{MINC_TOOLKIT})
  list(APPEND _libminc_hints
    "$ENV{MINC_TOOLKIT}"
    "$ENV{MINC_TOOLKIT}/lib"
    "$ENV{MINC_TOOLKIT}/lib/cmake"
  )
endif()

# Backward-compatible MINC_TOOLKIT_DIR detection (full toolkit with mincstats)
FIND_PATH( MINC_TOOLKIT_DIR bin/mincstats  HINTS $ENV{MINC_TOOLKIT} DOC "minc-toolkit install prefix")
if(MINC_TOOLKIT_DIR)
  list(APPEND _libminc_hints "${MINC_TOOLKIT_DIR}/lib${LIB_SUFFIX}")
endif()

FIND_PACKAGE( LIBMINC REQUIRED HINTS ${_libminc_hints})

# Resolve the libminc target name:
#  - installed libminc (modernize_cmake) provides LIBMINC::minc2
#  - build-tree libminc (modernize_cmake) provides minc2
#  - older (released) libminc provides only variables, no targets
if(TARGET LIBMINC::minc2)
  set(LIBMINC_TARGET LIBMINC::minc2)
elseif(TARGET minc2)
  set(LIBMINC_TARGET minc2)
elseif(LIBMINC_FOUND AND DEFINED LIBMINC_LIBRARIES)
  # Legacy libminc: create an imported target from the old-style variables
  message(STATUS "Creating LIBMINC::minc2 imported target from legacy variables")
  add_library(LIBMINC::minc2 INTERFACE IMPORTED)
  set_target_properties(LIBMINC::minc2 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LIBMINC_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES      "${LIBMINC_LIBRARIES}"
  )
  if(DEFINED LIBMINC_LIBRARY_DIRS)
    set_property(TARGET LIBMINC::minc2 APPEND PROPERTY
      INTERFACE_LINK_DIRECTORIES "${LIBMINC_LIBRARY_DIRS}")
  endif()
  set(LIBMINC_TARGET LIBMINC::minc2)
else()
  message(FATAL_ERROR
    "LIBMINC found but no imported targets or legacy variables are available.\n"
    "Set LIBMINC_DIR to a libminc build tree or install prefix.")
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

# ── Package export ──────────────────────────────────────────────────────
include(CMakePackageConfigHelpers)

set(MINC2_SIMPLE_INCLUDE_DIR include)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/minc2-simpleConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/minc2-simpleConfig.cmake
  INSTALL_DESTINATION lib/cmake/minc2-simple
  PATH_VARS MINC2_SIMPLE_INCLUDE_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/minc2-simpleConfigVersion.cmake
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

install(
  EXPORT minc2-simple-targets
  NAMESPACE MINC2_SIMPLE::
  DESTINATION lib/cmake/minc2-simple
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/minc2-simpleConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/minc2-simpleConfigVersion.cmake
  DESTINATION lib/cmake/minc2-simple
)

# Build-tree export so the build directory can be used directly
export(EXPORT minc2-simple-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/minc2-simple-targets.cmake
  NAMESPACE MINC2_SIMPLE::
)

#kate: indent-width 2; replace-tabs on; 
