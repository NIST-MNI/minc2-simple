# from https://github.com/eddelbuettel/rinside/blob/master/inst/examples/standard/cmake/CMakeLists.txt

set(NUM_TRUNC_CHARS 2)

set (RPATH "R")
set (RSCRIPT_PATH "Rscript")

if (CMAKE_HOST_WIN32)
    execute_process(COMMAND ${RSCRIPT_PATH} -e "cat(.Platform$r_arch)"
                    OUTPUT_VARIABLE R_ARCH)
		
	execute_process(COMMAND ${RPATH} --arch ${R_ARCH} RHOME
                    OUTPUT_VARIABLE R_HOME)
	
    string(REPLACE "\\" "/" R_HOME ${R_HOME})	
	
	set (RPATH ${R_HOME}/bin/R)
endif()

set (RCPPFLAGS_CMD " ${RPATH} " " CMD " " config " " --cppflags ") 

execute_process(COMMAND ${RPATH} CMD config --cppflags
                OUTPUT_VARIABLE RCPPFLAGS)
				
if (CMAKE_HOST_WIN32)					
	if (${RCPPFLAGS} MATCHES "[-][I]([^ ;])+")
	    set (RCPPFLAGS ${CMAKE_MATCH_0})
    endif()
endif()

string(SUBSTRING ${RCPPFLAGS} ${NUM_TRUNC_CHARS} -1 RCPPFLAGS)
SET(RINSIDE_INCLUDE_DIR ${RCPPFLAGS})
#include_directories(${RCPPFLAGS})
SET(RINSIDE_LIB_DIR )


execute_process(COMMAND ${RPATH} CMD config --ldflags
                OUTPUT_VARIABLE RLDFLAGS)
string(LENGTH ${RLDFLAGS} RLDFLAGS_LEN)

if (${RLDFLAGS} MATCHES "[-][L]([^ ;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RLDFLAGS_L)
    string(STRIP ${RLDFLAGS_L} RLDFLAGS_L )
    #link_directories(${RLDFLAGS_L} )
    SET(RINSIDE_LIB_DIR ${RINSIDE_LIB_DIR} ${RLDFLAGS_L})
endif()

if (${RLDFLAGS} MATCHES "[-][l]([^;])+")
    string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RLDFLAGS_l)
    string(STRIP ${RLDFLAGS_l} RLDFLAGS_l )
endif()

execute_process(COMMAND ${RSCRIPT_PATH} -e "Rcpp:::CxxFlags()"
                OUTPUT_VARIABLE RCPPINCL)
string(SUBSTRING ${RCPPINCL} ${NUM_TRUNC_CHARS} -1 RCPPINCL)
#include_directories(${RCPPINCL})
SET(RINSIDE_INCLUDE_DIR ${RINSIDE_INCLUDE_DIR} ${RCPPINCL})

execute_process(COMMAND ${RSCRIPT_PATH} -e "Rcpp:::LdFlags()"
                OUTPUT_VARIABLE RCPPLIBS)

execute_process(COMMAND ${RSCRIPT_PATH} -e "RInside:::CxxFlags()"
                OUTPUT_VARIABLE RINSIDEINCL)
string(SUBSTRING ${RINSIDEINCL} ${NUM_TRUNC_CHARS} -1 RINSIDEINCL)
#include_directories(${RINSIDEINCL})
SET(RINSIDE_INCLUDE_DIR ${RINSIDE_INCLUDE_DIR} ${RINSIDEINCL})

execute_process(COMMAND ${RSCRIPT_PATH} -e "RInside:::LdFlags()"
                OUTPUT_VARIABLE RINSIDELIBS)

if (CMAKE_HOST_WIN32)
    string(LENGTH "libRcpp.a" lenRcppName)
    string(LENGTH ${RCPPLIBS} lenRcppFQName)
	
    math(EXPR RLibPathLen ${lenRcppFQName}-${lenRcppName}-1)
    string(SUBSTRING ${RCPPLIBS} 0 ${RLibPathLen} RCPPLIBS_L)
    #link_directories(${RCPPLIBS_L})
    SET(RINSIDE_LIB_DIR ${RINSIDE_LIB_DIR} ${RCPPLIBS_L})

	
    math(EXPR RLibPathLen ${RLibPathLen}+1)
    string(SUBSTRING ${RCPPLIBS} ${RLibPathLen} -1 RCPPLIBS_l)
	
	#Remove the quotes
    string(SUBSTRING ${RINSIDELIBS} 1 -1 RINSIDELIBS)
    string(LENGTH ${RINSIDELIBS} lenRInsideFQNameLen)
    math(EXPR lenRInsideFQNameLen ${lenRInsideFQNameLen}-1)
    string(SUBSTRING ${RINSIDELIBS} 0 ${lenRInsideFQNameLen} RINSIDELIBS)

    string(LENGTH "libRInside.a" lenRInsideName)
    string(LENGTH ${RINSIDELIBS} lenRInsideFQName)

    math(EXPR RLibPathLen ${lenRInsideFQName}-${lenRInsideName}-1)
    string(SUBSTRING ${RINSIDELIBS} 0 ${RLibPathLen} RINSIDELIBS_L)

    math(EXPR RLibPathLen ${RLibPathLen}+1)
    string(SUBSTRING ${RINSIDELIBS} ${RLibPathLen} -1 RINSIDELIBS_l)

    #link_directories(${RINSIDELIBS_L})
    SET(RINSIDE_LIB_DIR ${RINSIDE_LIB_DIR} ${RINSIDELIBS_L})
else()	
    if (${RCPPLIBS} MATCHES "[-][L]([^ ;])+")
        string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RCPPLIBS_L)
        #link_directories(${RCPPLIBS_L} )
        SET(RINSIDE_LIB_DIR ${RINSIDE_LIB_DIR} ${RCPPLIBS_L})
    endif()

    if (${RCPPLIBS} MATCHES "[-][l][R]([^;])+")
        string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RCPPLIBS_l)
    endif()

    if (${RINSIDELIBS} MATCHES "[-][L]([^ ;])+")
        string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RINSIDELIBS_L)
        #link_directories(${RINSIDELIBS_L})
        SET(RINSIDE_LIB_DIR ${RINSIDE_LIB_DIR} ${RINSIDELIBS_L})
    endif()

    if (${RINSIDELIBS} MATCHES "[-][l][R]([^;])+")
        string(SUBSTRING ${CMAKE_MATCH_0} ${NUM_TRUNC_CHARS} -1 RINSIDELIBS_l)
    endif()
endif()

execute_process(COMMAND ${RPATH} CMD config CXXFLAGS
                OUTPUT_VARIABLE RCXXFLAGS)

execute_process(COMMAND ${RPATH} CMD config BLAS_LIBS
                OUTPUT_VARIABLE RBLAS)

execute_process(COMMAND ${RPATH} CMD config LAPACK_LIBS
                OUTPUT_VARIABLE RLAPACK)

#set(CMAKE_CXX_FLAGS "-W -Wall -pedantic -Wextra ${CMAKE_CXX_FLAGS}")
#SET(RINSIDE_FLAGS )
SET(RINSIDE_LIBS ${RLDFLAGS_l} ${BLAS_LIBS} ${LAPACK_LIBS} ${RINSIDELIBS_l} ${RCPPLIBS_l})
SET(RINSIDE_FOUND TRUE)

# foreach (next_SOURCE ${sources})
#    get_filename_component(source_name ${next_SOURCE} NAME_WE)
#    add_executable( ${source_name} ${next_SOURCE} )
#    
#    target_link_libraries(${source_name} ${RLDFLAGS_l})
#    target_link_libraries(${source_name} ${BLAS_LIBS})
#    target_link_libraries(${source_name} ${LAPACK_LIBS})
#    target_link_libraries(${source_name} ${RINSIDELIBS_l})
#    target_link_libraries(${source_name} ${RCPPLIBS_l})
#       
# endforeach (next_SOURCE ${sources})