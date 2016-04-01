/*! 
 * \file common.h
 * \brief Defines common settings in ftrl
 */

#ifndef ftrl_COMMON_H_
#define ftrl_COMMON_H_

#include <cstdint>
#include <string>
#include <unordered_map>

namespace multiverso { namespace ftrl
{
    /*! \brief constant variable for table id */
    const int32_t kWeightGradTable = 0; //store gradient
    const int32_t kWeightAdaGradTable = 1; // store sum(pow(grad,2))
    /*
     * \brief Defines ftrl configs
     */
    struct Config
    {
    public:
        /*! \brief Inits configs from command line arguments */
        static void Init(int argc, char* argv[]);
        /*! \brief size of features */
        static int32_t num_features;
        /*! \brief number of iterations for trainning */
        static int32_t num_iterations;
        /*! \brief number of servers for Multiverso setting */
        static int32_t num_servers;
        /*! \brief server endpoint file */
        static std::string server_file;
        /*! \brief number of worker threads */
        static int32_t num_local_workers;
        /*! \brief number of local aggregation threads */
        static int32_t num_aggregator;
        static int32_t max_delay;
        /*! \brief hyper-parameter for ftrl */
        static float alpha_;
        /*! \brief hyper-parameter for ftrl */
        static float beta_;
        static float l1_,l2_;
        /*! \brief path of input files_list */
        static std::string input_files_list;
        static std::string output_model_file;
        
        /*! \brief predict mode */
        static bool predict;
        /*! \brief option specity whether use out of core computation */
        static bool out_of_core;
        /*! \brief memory capacity settings, for memory pools */
        static int64_t data_capacity;
        static int64_t model_capacity;
    private:
        /*! \brief Print usage */
        static void PrintUsage();
		static void PrintTrainingUsage();
        static void PrintPredictUsage();
        /*! \brief Check if the configs are valid */
		static void Check();
    };
} // namespace ftrl
} // namespace multiverso

#endif // ftrl_COMMON_H_
