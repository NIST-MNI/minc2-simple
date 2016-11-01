# Vladimir S. FONOV - vladimir.fonov@gmail.com

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)
PROJECT(minc2-simple C)

find_path(MINC_TOOLKIT_DIR bin/mincstats   HINTS /opt/minc )

FIND_PACKAGE( LIBMINC REQUIRED PATHS ${MINC_TOOLKIT_DIR}/lib )
INCLUDE( ${LIBMINC_USE_FILE} )

INCLUDE_DIRECTORIES(../src)


#TODO: finish this ?


IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(test)
ENDIF(BUILD_TESTING)

#kate: indent-width 2; replace-tabs on; 
