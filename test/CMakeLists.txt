LINK_LIBRARIES(${LIBMINC_LIBRARIES})


ADD_EXECUTABLE(test-reading   test-reading.c)
TARGET_LINK_LIBRARIES(test-reading minc2-simple ${LIBMINC_LIBRARIES})


ADD_TEST(test-reading test-reading) 