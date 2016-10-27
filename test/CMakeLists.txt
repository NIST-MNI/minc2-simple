LINK_LIBRARIES(${LIBMINC_LIBRARIES})


ADD_EXECUTABLE(test-io-full   test-io-full.c)
TARGET_LINK_LIBRARIES(test-io-full minc2-simple ${LIBMINC_LIBRARIES})

ADD_EXECUTABLE(test-io-hyper   test-io-hyper.c)
TARGET_LINK_LIBRARIES(test-io-hyper minc2-simple ${LIBMINC_LIBRARIES})

#TODO: add proper arguments to testing functions here
ADD_TEST(test-io-full test-io-full ) 
ADD_TEST(test-io-hyper test-io-hyper )


#kate: indent-width 2; replace-tabs on; 