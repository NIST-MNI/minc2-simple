LINK_LIBRARIES(${LIBMINC_LIBRARIES})

ADD_LIBRARY(minc2-simple 
            minc2-simple.c
            minc2-simple.h
            )
            
            
TARGET_LINK_LIBRARIES(minc2-simple ${LIBMINC_LIBRARIES})


