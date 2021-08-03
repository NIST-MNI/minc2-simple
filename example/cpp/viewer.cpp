#include <stdexcept>
#include <iostream>

#include "cxxopts.hpp"
#include "minc2-simple.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"
#include <stdio.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#include <GLFW/glfw3.h>

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


int main(int argc, char *argv[])
{
  cxxopts::Options options(argv[0], "minc viewer");
  options
      .positional_help("[optional args]")
      .show_positional_help();
  
  options.add_options()
    ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
    ("i,input", "Input file name", cxxopts::value<std::string>())
    ("help", "Print help")
  ;
  options.parse_positional({"input"});

  try {
    auto par = options.parse(argc, argv);

    if (par.count("help") || !par.count("input") )
    {
        std::cout << options.help({"", "Group"}) << std::endl;
        return 1;
    }

    minc2_file_handle  h=minc2_allocate0();
    if(!h)
        throw std::runtime_error("Can't allocate handle");
    if(minc2_open(h,par["input"].as<std::string>().c_str())!=MINC2_SUCCESS)
        throw std::runtime_error("Can't open file");   
    
    int ndim;
    minc2_ndim(h,&ndim);
    //TODO: check 3d

    minc2_setup_standard_order(h);
    
    struct minc2_dimension *_dims;
    minc2_get_representation_dimensions(h,&_dims);

    for(int i=0;i<ndim;++i)
    {
        std::cout<<"Dimension:"<<i<<" "<< _dims[i].id << " length:" << _dims[i].length << std::endl;
    }

    std::vector<float> x_slice_buffer(_dims[1].length * _dims[2].length);
    std::vector<float> y_slice_buffer(_dims[0].length * _dims[2].length);
    std::vector<float> z_slice_buffer(_dims[0].length * _dims[1].length);

    // init GLFW & IMGUI

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("Error initializing GLFW");
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL2 example", NULL, NULL);
    if (window == NULL)
        throw std::runtime_error("Error Creating GLFW window");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    // Create a OpenGL texture identifier
    GLuint x_slice_texture;
    glGenTextures(1, &x_slice_texture);
    glBindTexture(GL_TEXTURE_2D, x_slice_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    GLuint y_slice_texture;
    glGenTextures(1, &y_slice_texture);
    glBindTexture(GL_TEXTURE_2D, y_slice_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    GLuint z_slice_texture;
    glGenTextures(1, &z_slice_texture);
    glBindTexture(GL_TEXTURE_2D, z_slice_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // initial slice position
        static int x_slice_start[3]={_dims[0].length/2, 0, 0};
        static int x_slice_count[3]={1, _dims[1].length,_dims[2].length};

        static int y_slice_start[3]={0, _dims[1].length/2, 0};
        static int y_slice_count[3]={_dims[0].length, 1, _dims[2].length};

        static int z_slice_start[3]={0, 0, _dims[2].length/2};
        static int z_slice_count[3]={_dims[0].length,_dims[1].length,1};

        // intensity normalization
        static float norm_min = 0.0;
        static float norm_max = 100.0;

        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // file options
        {
            ImGui::Begin(par["input"].as<std::string>().c_str(),nullptr,ImGuiWindowFlags_AlwaysAutoResize);
            static int counter = 0;
            ImGui::Text("%dx%dx%d", _dims[0].length, _dims[1].length, _dims[2].length);                // Display some text (you can use a format strings too)

            ImGui::InputFloat("Min", &norm_min);
            ImGui::SameLine();
            ImGui::InputFloat("Max", &norm_max);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        //2. X Slice View
        {
            ImGui::Begin("Sagittal",nullptr,ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::SliderInt("X ", &x_slice_start[0], 0.0, _dims[0].length-1);  

            // load slice into buffer
            if(minc2_read_hyperslab(h,x_slice_start,x_slice_count,&x_slice_buffer[0],MINC2_FLOAT)==MINC2_SUCCESS)
            {   
                for(auto i=x_slice_buffer.begin();i!=x_slice_buffer.end();++i)
                    (*i)=((*i)-norm_min)/(norm_max-norm_min);

                glBindTexture(GL_TEXTURE_2D, x_slice_texture);
                // Upload pixels into texture
                #if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
                    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                #endif
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, x_slice_count[1], x_slice_count[2], 0, GL_LUMINANCE, GL_FLOAT, &x_slice_buffer[0]);

                ImGui::Image((void*)(intptr_t)x_slice_texture, ImVec2(x_slice_count[1], x_slice_count[2]), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));
            } else {
                ImGui::Text("Error loading slab");
            }

            ImGui::End();
        }

        //2. Y Slice View
        {
            ImGui::Begin("Coronal",nullptr,ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::SliderInt("Y", &y_slice_start[1], 0.0, _dims[1].length-1);  

            // load slice into buffer
            if(minc2_read_hyperslab(h,y_slice_start,y_slice_count,&y_slice_buffer[0],MINC2_FLOAT)==MINC2_SUCCESS)
            {   
                for(auto i=y_slice_buffer.begin();i!=y_slice_buffer.end();++i)
                    (*i)=((*i)-norm_min)/(norm_max-norm_min);

                glBindTexture(GL_TEXTURE_2D, y_slice_texture);
                // Upload pixels into texture
                #if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
                    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                #endif
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, y_slice_count[0], y_slice_count[2], 0, GL_LUMINANCE, GL_FLOAT, &y_slice_buffer[0]);

                ImGui::Image((void*)(intptr_t)y_slice_texture, ImVec2(y_slice_count[0], y_slice_count[2]), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));
            } else {
                ImGui::Text("Error loading slab");
            }

            ImGui::End();
        }

        //2. Y Slice View
        {
            ImGui::Begin("Axial",nullptr,ImGuiWindowFlags_AlwaysAutoResize);                          // Create a window and append into it.
            ImGui::SliderInt("Z", &z_slice_start[2], 0.0, _dims[2].length-1);  

            // load slice into buffer
            if(minc2_read_hyperslab(h,z_slice_start,z_slice_count,&z_slice_buffer[0],MINC2_FLOAT)==MINC2_SUCCESS)
            {   
                for(auto i=z_slice_buffer.begin();i!=z_slice_buffer.end();++i)
                    (*i)=((*i)-norm_min)/(norm_max-norm_min);

                glBindTexture(GL_TEXTURE_2D, z_slice_texture);
                // Upload pixels into texture
                #if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
                    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
                #endif
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, z_slice_count[0], z_slice_count[1], 0, GL_LUMINANCE, GL_FLOAT, &z_slice_buffer[0]);

                ImGui::Image((void*)(intptr_t)z_slice_texture, ImVec2(z_slice_count[0], z_slice_count[1]), ImVec2(0.0, 1.0), ImVec2(1.0, 0.0));
            } else {
                ImGui::Text("Error loading slab");
            }

            ImGui::End();
        }


        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        // If you are using this code with non-legacy OpenGL header/contexts (which you should not, prefer using imgui_impl_opengl3.cpp!!),
        // you may need to backup/reset/restore other state, e.g. for current shader using the commented lines below.
        //GLint last_program;
        //glGetIntegerv(GL_CURRENT_PROGRAM, &last_program);
        //glUseProgram(0);
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        //glUseProgram(last_program);

        glfwMakeContextCurrent(window);
        glfwSwapBuffers(window);
    }
    // Cleanup
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);

    minc2_close(h);
    minc2_free(h);
    glfwTerminate();

    return 0;
  } catch (const cxxopts::OptionException& e) {
    std::cerr << "error parsing options: " << e.what() << std::endl;
    return 1;
  } catch(std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
  } catch(...) {
        std::cerr << "Unknown exception caught" << std::endl;
  }

  glfwTerminate();

  return 1;
}

