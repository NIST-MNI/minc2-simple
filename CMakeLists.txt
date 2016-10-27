# CMakeFiles.txt for the minc2-simple
#
# Vladimir S. FONOV - vladimir.fonov@gmail.com

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(minc2-simple C)

INCLUDE(CTest)

ENABLE_TESTING()

FIND_PACKAGE( LIBMINC REQUIRED )
INCLUDE( ${LIBMINC_USE_FILE} )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(example)

IF(BUILD_TESTING)
  find_path(MINC_TOOLKIT_DIR bin/mincstats
          HINTS /opt/minc )
  ADD_SUBDIRECTORY(test)
ENDIF(BUILD_TESTING)

#kate: indent-width 2; replace-tabs on; 
