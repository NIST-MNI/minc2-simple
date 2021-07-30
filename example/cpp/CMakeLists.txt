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

target_link_libraries(imgui PRIVATE glfw)

add_executable(viewer viewer.cpp)

target_include_directories(viewer PRIVATE imgui imgui/backends )
target_include_directories(viewer PRIVATE glfw/include)

target_link_libraries(viewer PRIVATE minc2-simple imgui glfw ${LIBMINC_LIBRARIES})

if(TARGET OpenGL::GL)
    target_link_libraries(viewer PRIVATE OpenGL::GL)
else()
    target_link_libraries(viewer PRIVATE ${OPENGL_gl_LIBRARY})
endif()
