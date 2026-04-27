ADD_EXECUTABLE(test-io-full   test-io-full.c)
TARGET_LINK_LIBRARIES(test-io-full minc2-simple)

ADD_EXECUTABLE(test-io-hyper   test-io-hyper.c)
TARGET_LINK_LIBRARIES(test-io-hyper minc2-simple)

ADD_EXECUTABLE(test-io-hyper-norm   test-io-hyper-norm.c)
TARGET_LINK_LIBRARIES(test-io-hyper-norm minc2-simple)

ADD_EXECUTABLE(test-io-iterator   test-io-iterator.c)
TARGET_LINK_LIBRARIES(test-io-iterator minc2-simple)

ADD_EXECUTABLE(test-nan     test-nan.c)
TARGET_LINK_LIBRARIES(test-nan minc2-simple m)

ADD_EXECUTABLE(test-tags     test-tags.c)
TARGET_LINK_LIBRARIES(test-tags minc2-simple)

ADD_EXECUTABLE(test-variable test-variable.c)
TARGET_LINK_LIBRARIES(test-variable minc2-simple)

# Fixture generator for the 4D-irregular test. Built so the .mnc fixture
# can be regenerated, but not registered as a CTest. Run manually:
#   ./test/gen-test_4D_irregular_offsets ../test/test_4D_irregular_offsets.mnc
ADD_EXECUTABLE(gen-test_4D_irregular_offsets gen-test_4D_irregular_offsets.c)
TARGET_LINK_LIBRARIES(gen-test_4D_irregular_offsets minc2-simple)

ADD_EXECUTABLE(test-4D-irregular test-4D-irregular.c)
TARGET_LINK_LIBRARIES(test-4D-irregular minc2-simple m)


#TODO: add proper arguments to testing functions here
ADD_TEST(test-io-full       test-io-full ${CMAKE_CURRENT_SOURCE_DIR}/test-nan-float-history.mnc ${CMAKE_CURRENT_BINARY_DIR}/test-nan-float_.mnc) 
ADD_TEST(test-io-hyper      test-io-hyper ${CMAKE_CURRENT_SOURCE_DIR}/test-nan-float.mnc ${CMAKE_CURRENT_BINARY_DIR}/test-nan-float_.mnc)
ADD_TEST(test-io-hyper-norm test-io-hyper-norm ${CMAKE_CURRENT_SOURCE_DIR}/test-nan-float.mnc ${CMAKE_CURRENT_BINARY_DIR}/test-nan-float_.mnc)
ADD_TEST(test-io-iterator   test-io-iterator ${CMAKE_CURRENT_SOURCE_DIR}/test-nan-float.mnc ${CMAKE_CURRENT_BINARY_DIR}/test-nan-float_.mnc)
ADD_TEST(test-tags          test-tags ${CMAKE_CURRENT_SOURCE_DIR}/tags.tag ${CMAKE_CURRENT_BINARY_DIR}/tags.tag )
ADD_TEST(test-nan-float     test-nan ${CMAKE_CURRENT_BINARY_DIR}/test-nan-float.mnc 0)
ADD_TEST(test-nan-short     test-nan ${CMAKE_CURRENT_BINARY_DIR}/test-nan-short.mnc 1)
ADD_TEST(test-variable      test-variable ${CMAKE_CURRENT_SOURCE_DIR}/test-nan-float_4D.mnc)
ADD_TEST(test-4D-irregular  test-4D-irregular ${CMAKE_CURRENT_SOURCE_DIR}/test_4D_irregular_offsets.mnc ${CMAKE_CURRENT_BINARY_DIR}/test_4D_irregular_offsets_out.mnc)


#kate: indent-width 2; replace-tabs on; 
