# Find HDF5 — needed as a direct public dependency because minc2-simple.h
# transitively includes minc2.h -> hdf5.h, and libminc's install-tree export
# does not propagate HDF5 include directories through its target.
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE ON)
find_package(HDF5 REQUIRED COMPONENTS C)

IF(BUILD_MINC2_SIMPLE_SHARED)
  message("Building shared")
  ADD_LIBRARY(minc2-simple SHARED
              minc2-simple.c
              minc2-matrix-ops.c
              minc2-simple.h
              minc2-simple-int.h
              )
  TARGET_LINK_LIBRARIES(minc2-simple PUBLIC ${LIBMINC_TARGET})
  target_include_directories(minc2-simple
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>
      ${HDF5_INCLUDE_DIRS}
  )
  set_target_properties(minc2-simple PROPERTIES MACOSX_RPATH ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})

  INSTALL(TARGETS minc2-simple
          EXPORT minc2-simple-targets
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          PUBLIC_HEADER DESTINATION include)
ENDIF()


IF(BUILD_MINC2_SIMPLE_STATIC)
  message("Building static")
  ADD_LIBRARY(minc2-simple-static STATIC
              minc2-simple.c
              minc2-matrix-ops.c
              minc2-simple.h
              minc2-simple-int.h
            )

  TARGET_LINK_LIBRARIES(minc2-simple-static PUBLIC ${LIBMINC_TARGET})
  target_include_directories(minc2-simple-static
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>
      ${HDF5_INCLUDE_DIRS}
  )

  set_target_properties(minc2-simple-static PROPERTIES MACOSX_RPATH ${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX})
  set_target_properties(minc2-simple-static PROPERTIES PUBLIC_HEADER "minc2-simple.h;minc2-simple-int.h")

  INSTALL(TARGETS minc2-simple-static
          EXPORT minc2-simple-targets
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          PUBLIC_HEADER DESTINATION include)

ENDIF()
