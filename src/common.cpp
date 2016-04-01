#include "common.h"

#include <cstring>

namespace multiverso { namespace ftrl 
{
    const int64_t kMB = 1024 * 1024;

    // -- Begin: Config definitioin and defalut values --------------------- //
    int32_t Config::num_features = 7000000;
    int32_t Config::num_iterations = 1;
    int32_t Config::num_servers = 1;
    int32_t Config::num_local_workers = 1;
    int32_t Config::num_aggregator = 1;
    int32_t Config::max_delay = 0;
    float Config::alpha_ = 0.01f;
    float Config::beta_ = 1.f;
    float Config::l1_= 1.f;
    float Config::l2_ = 1.f;
    std::string Config::server_file = "";
    std::string Config::input_files_list = "";
    bool Config::predict = false;
    bool Config::out_of_core = false;
    int64_t Config::data_capacity = 1024 * kMB;
    int64_t Config::model_capacity = 512 * kMB;
    // -- End: Config definitioin and defalut values ----------------------- //

    void Config::Init(int argc, char* argv[])
    {
        if (argc < 2)
        {
            PrintUsage();
        }
        for (int i = 1; i < argc; ++i)
        {
            if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "--help") == 0)
            {
                PrintUsage();
            }
            if (strcmp(argv[i], "-num_features") == 0) num_features = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-num_iterations") == 0) num_iterations = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-num_servers") == 0) num_servers = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-num_local_workers") == 0) num_local_workers = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-num_aggregator") == 0) num_aggregator = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-max_delay") == 0) max_delay = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-alpha") == 0) alpha_ = static_cast<float>(atof(argv[i + 1]));
            if (strcmp(argv[i], "-beta") == 0) beta_ = static_cast<float>(atof(argv[i + 1]));
            if (strcmp(argv[i], "-l1") == 0) l1_ = static_cast<float>(atof(argv[i + 1]));
            if (strcmp(argv[i], "-l2") == 0) l2_ = static_cast<float>(atof(argv[i + 1]));
            if (strcmp(argv[i], "-input_files_list") == 0) input_files_list = std::string(argv[i + 1]);
            if (strcmp(argv[i], "-server_file") == 0) server_file = std::string(argv[i + 1]);
            if (strcmp(argv[i], "-out_of_core") == 0) out_of_core = true;
            if (strcmp(argv[i], "-data_capacity") == 0) data_capacity = atoi(argv[i + 1]) * kMB;
            if (strcmp(argv[i], "-model_capacity") == 0) model_capacity = atoi(argv[i + 1]) * kMB;    
        }
        Check();
    }

    void Config::PrintTrainingUsage()
    {
        printf("ftrl usage: \n");
      
        printf("-num_servers <arg>       Number of servers. Default: 1\n");
        printf("-num_local_workers <arg> Number of local training threads. Default: 4\n");
        printf("-num_aggregator <arg>    Number of local aggregation threads. Default: 1\n");
        printf("-server_file <arg>       Server endpoint file. Used by MPI-free version\n"); 
        printf("-num_features <arg>        Size of dataset vocabulary \n");
        printf("-num_iterations <arg>    Number of iteratioins. Default: 1\n");
        printf("-alpha <arg>             learning rate alpha. Default: 0.01\n");
        printf("-beta <arg>              hyperpara beta. Default: 1\n\n");
        printf("-l1 <arg>              l1 reguralization. Default: 1\n\n");
        printf("-l2 <arg>              l2 reguralization. Default: 1\n\n");
        printf("-input_files_list <arg>         files of input data, containing multiple files\n");
        printf("-num_local_workers <arg> Number of local training threads. Default: 4\n");
        printf("-out_of_core             Use out of core computing \n\n");
        printf("-data_capacity <arg>     Memory pool size(MB) for data storage, \n");
        printf("                         should larger than the any data block\n");
        exit(0);
    }

    void Config::PrintPredictUsage()
    {
        printf("ftrl Inference usage: \n");
        printf("-num_features <arg>        Size of dataset vocabulary \n");
        printf("-num_iterations <arg>    Number of iteratioins. Default: 1\n");
        printf("-alpha <arg>             learning rate alpha. Default: 0.01\n");
        printf("-beta <arg>              hyperpara beta. Default: 1\n\n");
        printf("-l1 <arg>              l1 reguralization. Default: 1\n\n");
        printf("-l2 <arg>              l2 reguralization. Default: 1\n\n");
        printf("-input_files_list <arg>         files of input data, containing multiple files\n");
        printf("-num_local_workers <arg> Number of local training threads. Default: 4\n");
        printf("-out_of_core             Use out of core computing \n\n");
        printf("-data_capacity <arg>     Memory pool size(MB) for data storage, \n");
        printf("                         should larger than the any data block\n");
        exit(0);
    }

    void Config::PrintUsage()
    {
    		PrintTrainingUsage();
            PrintPredictUsage();
    }

	void Config::Check(){
		if(input_files_list =="" ||num_features <= 0 )
            PrintUsage();
	}
} // namespace ftrl
} // namespace multiverso
