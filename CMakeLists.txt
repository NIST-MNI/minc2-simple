# CMakeFiles.txt for the minc2-simple
#
# Vladimir S. FONOV - vladimir.fonov@gmail.com

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(minc2-simple C)
INCLUDE(CTest)

FIND_PATH( MINC_TOOLKIT_DIR bin/mincstats  HINTS /opt/minc DOC "minc-toolkit install prefix")
FIND_PACKAGE( LIBMINC REQUIRED HINTS ${MINC_TOOLKIT_DIR}/lib${LIB_SUFFIX})


ENABLE_TESTING()

INCLUDE( ${LIBMINC_USE_FILE} )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(example)

IF(BUILD_TESTING)
  
  ADD_SUBDIRECTORY(test)
ENDIF(BUILD_TESTING)

#kate: indent-width 2; replace-tabs on; 
