#LINK_LIBRARIES(${LIBMINC_LIBRARIES})

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

