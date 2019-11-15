#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>



// for get_opt_long
#include <getopt.h>

void show_usage(const char *name)
{
  std::cerr 
    << "Usage: "<<name<<" <script.pth> <input>  <output> " << std::endl
    << "Optional parameters:" << std::endl
    << "\t--verbose be verbose" << std::endl
    << "\t--clobber clobber the output files" << std::endl;
}


int main(int argc,char **argv) 
{
    int clobber=0;
    int verbose=0;
    std::string mask_f;
    int c;
    // read the arguments
    static struct option long_options[] =
    {
        {"verbose", no_argument, &verbose, 1},
        {"quiet", no_argument, &verbose, 0},
        {"clobber", no_argument, &clobber, 1},
        {"mask", required_argument, 0, 'm'},
        {0, 0, 0, 0}
    };

    for (;;)
    {
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long (argc, argv, "v", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c)
        {
        case 0:
            break;
        case 'm':
            mask_f=optarg;
            break;
        case '?':
                /* getopt_long already printed an error message. */
        default:
                show_usage(argv[0]);
                return 1;
        }
    }
    if ((argc - optind) < 3)
    {
        show_usage(argv[0]);
        return 1;
    }
    
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[optind]);

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::ones({1, 1, 64, 64, 64}));

        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << "Output:" << output.size() << std::endl;
        //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/1) << '\n';

        //std::cout<<module<<std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "ok\n";    
    return 0;
}
