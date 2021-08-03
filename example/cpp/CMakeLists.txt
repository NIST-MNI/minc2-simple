OPTION(BUILD_IMGUI_DEMO "Build ImGUI Demo APP" OFF)

SET(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)

add_subdirectory(glfw)

add_library(imgui

            imgui/imconfig.h
            imgui/imgui.cpp
            imgui/imgui_demo.cpp
            imgui/imgui_draw.cpp
            imgui/imgui.h
            imgui/imgui_internal.h
            imgui/imgui_tables.cpp
            imgui/imgui_widgets.cpp
            imgui/imstb_rectpack.h
            imgui/imstb_textedit.h
            imgui/imstb_truetype.h

            imgui/backends/imgui_impl_glfw.h
            imgui/backends/imgui_impl_glfw.cpp

            imgui/backends/imgui_impl_opengl2.h
            imgui/backends/imgui_impl_opengl2.cpp

)

target_include_directories(imgui PUBLIC imgui)
target_include_directories(imgui PUBLIC . .. imgui/examples)
target_compile_definitions(imgui PUBLIC -DIMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui PUBLIC glfw)

add_executable(viewer viewer.cpp)

target_include_directories(viewer PUBLIC imgui imgui/backends )
target_include_directories(viewer PUBLIC glfw/include)

target_link_libraries(viewer PUBLIC minc2-simple imgui glfw ${LIBMINC_LIBRARIES})

if(TARGET OpenGL::GL)
    target_link_libraries(viewer PUBLIC OpenGL::GL)
else()
    target_link_libraries(viewer PUBLIC ${OPENGL_gl_LIBRARY})
endif()


IF(BUILD_IMGUI_DEMO)
    add_executable(imgui_demo imgui/imgui_demo.cpp imgui_demo_main.cpp)
    target_include_directories(imgui_demo PUBLIC imgui imgui/backends )
    target_include_directories(imgui_demo PUBLIC glfw/include)

    target_link_libraries(imgui_demo PUBLIC imgui glfw)

    if(TARGET OpenGL::GL)
        target_link_libraries(imgui_demo PUBLIC OpenGL::GL)
    else()
        target_link_libraries(imgui_demo PUBLIC ${OPENGL_gl_LIBRARY})
    endif()

ENDIF()