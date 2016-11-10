ADD_LIBRARY(minc2-simple SHARED
            minc2-simple.c
            minc2-simple.h
            )

ADD_LIBRARY(minc2-simple-static STATIC
             minc2-simple.c
             minc2-simple.h
           )
            
TARGET_LINK_LIBRARIES(minc2-simple ${LIBMINC_LIBRARIES})
TARGET_LINK_LIBRARIES(minc2-simple-static ${LIBMINC_LIBRARIES})


set_target_properties(minc2-simple minc2-simple-static PROPERTIES MACOSX_RPATH ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})