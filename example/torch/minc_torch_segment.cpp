#include <torch/torch.h>
#include <torch/script.h> // One-stop header


#include "minc2-simple.h"

#include <iostream>
#include <memory>



// for get_opt_long
#include <getopt.h>

void show_usage(const char *name)
{
  std::cerr 
    << "Usage: "<<name<<" <script.pth> <input> <output> [debug]" << std::endl
    << "Optional parameters:" << std::endl
    << "\t--channels <n> add more input channels, set them to 38.81240207 for now" << std::endl
    << "\t--mask <file> use binary mask to restrict application of the network (TODO)" << std::endl
    << "\t--verbose be verbose" << std::endl
    << "\t--clobber clobber the output files" << std::endl;
}


int main(int argc,char **argv) 
{
    int clobber=0;
    int verbose=0;
    std::string mask_f;
    int channels=1;
    int c;
    // read the arguments
    static struct option long_options[] =
    {
        {"verbose", no_argument, &verbose, 1},
        {"quiet",   no_argument, &verbose, 0},
        {"clobber", no_argument, &clobber, 1},
        {"mask",    required_argument, 0, 'm'},
        {"channels",    required_argument, 0, 'c'},
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
        case 'c':
            channels=atoi(optarg);
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
    bool debug_output = (argc - optind)>3;
    
    try {
        int ndim;
        int nelement;
        const int patch_sz = 80;
        const int stride = patch_sz/2;
        struct minc2_dimension * _dims;
        struct minc2_dimension * store_dims;
        torch::NoGradGuard no_grad; // disable gradient calculation (?)
        // Deserialize the ScriptModule from a file using torch::jit::load().
        torch::jit::script::Module module = torch::jit::load(argv[optind]);
        module.eval();

        // dump module
        // module.dump(true,true,false);

        // load minc file
        minc2_file_handle h = minc2_allocate0();
        minc2_file_handle o = minc2_allocate0();
        minc2_file_handle oo = minc2_allocate0();

        if(minc2_open(h, argv[optind+1])!=MINC2_SUCCESS)
        {
            std::cerr << "Can't open " << argv[optind]  << " for reading" << std::endl;
            return 1;
        }
        //TODO: verify that volume is 3D 
        minc2_get_store_dimensions(h, &store_dims);
        minc2_define(o,store_dims, MINC2_UBYTE, MINC2_UBYTE);

        if(minc2_create(o, argv[optind+2])!=MINC2_SUCCESS)
        {
            std::cerr << "Can't open " << argv[optind+2] << "for writing" << std::endl;
            return 1;
        }

        if(debug_output)
        {
            minc2_define(oo, store_dims, MINC2_FLOAT, MINC2_FLOAT);

            if(minc2_create(oo, argv[optind+3])!=MINC2_SUCCESS)
            {
                std::cerr << "Can't open " << argv[optind+3] << "for writing" << std::endl;
                return 1;
            }
        }

        minc2_setup_standard_order(h);
        minc2_setup_standard_order(o);

        if(debug_output)
            minc2_setup_standard_order(oo);

        minc2_ndim(h, &ndim);
        minc2_nelement(h, &nelement);
        minc2_get_store_dimensions(h, &_dims);

        float *buffer=(float*)calloc(nelement, sizeof(float));

        if(minc2_load_complete_volume(h, buffer, MINC2_FLOAT)!=MINC2_SUCCESS)
        {
            std::cerr << "Error reading data from " << argv[optind+1] << std::endl;
            return 1;
        }
        // convert to tensor format
        torch::Tensor input    = torch::from_blob(buffer, { _dims[0].length, _dims[1].length, _dims[2].length }, torch::kFloat32 );
        torch::Tensor output;
        torch::Tensor output_d;

        torch::Tensor ones;
        torch::Tensor weights;

        // convert to 5D format
        input = input.unsqueeze(0).unsqueeze(0);

        // add more input channels 
        // TODO: read them from file
        if(channels>1) {
            torch::Tensor dummy = torch::full({ _dims[0].length, _dims[1].length, _dims[2].length },38.81240207);
            dummy = dummy.unsqueeze(0).unsqueeze(0);
            std::vector<torch::Tensor> _inputs={input};
            for(int i=1;i<channels;i++)
                _inputs.push_back(dummy);
            // append new channels
            input = torch::cat(_inputs,1);
        }


        // TODO: make patches overlap, and then aggregate results
        for(int z=0; z< (_dims[0].length - _dims[0].length%patch_sz); z+=stride)
        {
            for(int y=0; y< (_dims[1].length- _dims[1].length%patch_sz); y+=stride)
            {
                for(int x=0; x< (_dims[2].length- _dims[2].length%patch_sz); x+=stride)
                {
                    int _x = ( x + patch_sz) > _dims[2].length ? _dims[2].length - patch_sz:x;
                    int _y = ( y + patch_sz) > _dims[1].length ? _dims[1].length - patch_sz:y;
                    int _z = ( z + patch_sz) > _dims[0].length ? _dims[0].length - patch_sz:z;

                    std::cout << _x << "," << _y << "," << _z <<std::endl;
                    // Create a vector of inputs.
                    std::vector<torch::jit::IValue> inputs;
                    auto inp = input.slice(2,_z,_z+patch_sz).slice(3,_y,_y+patch_sz).slice(4, _x, _x+patch_sz);
                    // extract patch
                    inputs.push_back( inp );

                    // Execute the model and turn its output into a tensor.
                    auto out = torch::log_softmax(module.forward(inputs).toTensor(), /*dim=*/1);
                    out = out.squeeze(0);

                    if(!output.defined()) // neeed to allocate
                    {
                        std::cout<<"out:"<< out.size(0)<<","<< out.size(1)<<","<< out.size(2)<<","<< out.size(3) << std::endl;

                        std::cout<<"Allocating output:" << out.size(0) << ","<< _dims[0].length << ","<< _dims[1].length << "," << _dims[2].length << std::endl;
                        std::cout<<"Allocating ones:"   << out.size(0) << ","<< patch_sz << "," << patch_sz << "," << patch_sz << std::endl;

                        output  = torch::zeros( {out.size(0), _dims[0].length, _dims[1].length, _dims[2].length });
                        weights = torch::zeros( {out.size(0), _dims[0].length, _dims[1].length, _dims[2].length });
                        ones    = torch::ones(  {out.size(0), patch_sz, patch_sz, patch_sz } );
                    }

                    output.slice(1, _z,_z+patch_sz).slice(2,_y,_y+patch_sz).slice(3,_x,_x+patch_sz)  += out;
                    weights.slice(1,_z,_z+patch_sz).slice(2,_y,_y+patch_sz).slice(3,_x,_x+patch_sz) += ones;


                    /// DEBUG
                    //break;
                }

                /// DEBUG
                //break;
            }
            /// DEBUG
            //break;
        }

        output/=weights; // normalize

        if(debug_output)
            output_d = output.slice(0,0,1).squeeze(0);

        // extract prediction
        auto out_ = output.argmax(0).to(torch::kU8); 
        
        if(debug_output)
        {
            if(minc2_save_complete_volume(oo, output_d.data_ptr<float>(), MINC2_FLOAT)!=MINC2_SUCCESS)
            {
                std::cerr << "Error writing data to " << "debug.mnc" << std::endl;
                return 1;
            }
        }
        if(minc2_save_complete_volume(o, out_.data_ptr<u_char>(), MINC2_BYTE)!=MINC2_SUCCESS)
        {
            std::cerr << "Error writing data to " << argv[optind+1] << std::endl;
            return 1;
        }

        free(buffer);

        minc2_close(h);
        minc2_close(o);
        if(debug_output)
            minc2_close(oo);
        /*deallocate*/
        minc2_free(h);
        minc2_free(o);
        minc2_free(oo);
    }

    catch (const c10::Error& e) 
    {
        std::cerr << "error:"<< e.what() << std::endl;
        return -1;
    }
    std::cout << "ok\n";
    return 0;
}
