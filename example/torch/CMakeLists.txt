find_package(Torch REQUIRED)


#IF(TORCH_FOUND)

add_executable(minc_torch_segment minc_torch_segment.cpp)
#target_compile_features(minc_torch_segment.cxx PUBLIC cxx_range_for)
set_property(TARGET minc_torch_segment PROPERTY CXX_STANDARD 11)

target_link_libraries(minc_torch_segment ${TORCH_LIBRARIES} ${LIBMINC_LIBRARIES})


#ENDIF(TORCH_FOUND)