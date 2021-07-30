#include <stdexcept>
#include <iostream>

#include "cxxopts.hpp"
#include "minc2-simple.h"


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


    minc2_close(h);
    minc2_free(h);
    
    return 0;
  } catch (const cxxopts::OptionException& e) {
    std::cerr << "error parsing options: " << e.what() << std::endl;
    return 1;
  } catch(std::exception& ex) {
        std::cerr << "Exception caught: " << ex.what() << std::endl;
  } catch(...) {
        std::cerr << "Unknown exception caught" << std::endl;
  }
  return 1;
}

