﻿#include "common.h"
#include "trainer.h"
#include "dsample.h"
#include "data_block.h"
#include "util.h"
#include <vector>
#include <unordered_set>
#include <iostream>
#include <multiverso/barrier.h>
#include <multiverso/log.h>
#include <multiverso/row.h>

namespace multiverso { namespace ftrl
{     
	static int col_size = 10;
    class Ftrl_Lr
    {
    public:
        static void Run(int argc, char** argv)
        {
            Config::Init(argc, argv);
            
           // Barrier* barrier = new Barrier(Config::num_local_workers);
            std::vector<TrainerBase*> trainers;
            for (int32_t i = 0; i < Config::num_local_workers; ++i)
            {
                Trainer* trainer = new Trainer();
                trainers.push_back(trainer);
            }

            ParamLoader* param_loader = new ParamLoader();
            multiverso::Config config;
            config.num_servers = Config::num_servers;
            config.max_delay = Config::max_delay;
            config.num_aggregator = Config::num_aggregator;
            config.server_endpoint_file = Config::server_file;

            Multiverso::Init(trainers, param_loader, config, &argc, &argv);

            Log::ResetLogFile("Ftrl_Lr."
                + std::to_string(clock()) + ".log");
            // factory class to construct memory stream or disk stream
            // data_stream = CreateDataStream();
            InitMultiverso();
            Train();
			std::string model = "./local_model";
			reinterpret_cast<Trainer*>(trainers[0])->SaveModel(model);

            Multiverso::Close();
            
            for (auto& trainer : trainers)
            {
                delete trainer;
            }
            delete param_loader;
            
            //DumpModel();       
            //delete barrier;
        }
    private:
        static void Train()
        {
            Multiverso::BeginTrain();
            std::queue<DataBlock*> datablock_queue;
            for (int32_t i = 0; i < Config::num_iterations; ++i)
            {
                // Train corpus block by block
                std::string input_files = Config::input_files_list;
				std::ifstream fin(input_files);
				std::string single_train_file;
                while(getline(fin,single_train_file ))
                {
		    	Multiverso::BeginClock();
             	DataBlock * ftrl_block = new DataBlock();
                    ftrl_block->Read(single_train_file );
		    		if(ftrl_block->HasLoad()){
                        ftrl_block->set_iteration(i);
                        PushDataBlock(datablock_queue, ftrl_block);
                    }
                    else{
                        Log::Info("read train file %s failed\n",single_train_file.c_str());
						delete ftrl_block;
					}
                    Multiverso::EndClock();
                }
            }
            Multiverso::EndTrain();
        }

        static void InitMultiverso()
        {
            Multiverso::BeginConfig();
            CreateTable();
            ConfigTable();
            Initialize();
            Multiverso::EndConfig();
        }

        static void Initialize()
        {
           for (int32_t i = 0; i < Config::num_features / col_size; ++i)
           {
           	for (int32_t j = 0; j < col_size ; ++j)
			{
                Multiverso::AddToServer<float>(kWeightGradTable,i, j, 0.);
                Multiverso::AddToServer<float>(kWeightAdaGradTable,i, j, 0.); 
			}
           }
            Multiverso::Flush();
        }

        static void PushDataBlock(
            std::queue<DataBlock*> &datablock_queue, DataBlock* data_block)
        {
            
            multiverso::Multiverso::PushDataBlock(data_block);
            
            datablock_queue.push(data_block);
            //limit the max size of total datablocks to avoid out of memory
            while (datablock_queue.size() > 3 )
            {
                std::chrono::milliseconds dura(200);
                std::this_thread::sleep_for(dura);
                //Remove the datablock which is delt by parameterloader and trainer
                RemoveDoneDataBlock(datablock_queue);
            }
        }
         static void RemoveDoneDataBlock(
            std::queue<DataBlock*> &datablock_queue)
        {
            while (datablock_queue.empty() == false 
                && datablock_queue.front()->IsDone())
            {
                DataBlock *p_data_block = datablock_queue.front();
                datablock_queue.pop();
                delete p_data_block;
				Log::Info("delete data block \n");
            }
        }


        static void CreateTable()
        {
            int32_t num_features = Config::num_features;
            Type float_type = Type::Float;
            multiverso::Format dense_format = multiverso::Format::Dense;
            //multiverso::Format sparse_format = multiverso::Format::Sparse;

			int feature_group = num_features / col_size;
            Multiverso::AddServerTable(kWeightGradTable, feature_group,
                col_size, float_type, dense_format);
            Multiverso::AddCacheTable(kWeightGradTable, feature_group,
                col_size, float_type, dense_format);
            Multiverso::AddAggregatorTable(kWeightGradTable, feature_group,
                col_size, float_type, dense_format);
            Multiverso::AddServerTable(kWeightAdaGradTable, feature_group,
                col_size, float_type, dense_format);
            Multiverso::AddCacheTable(kWeightAdaGradTable, feature_group,
                col_size, float_type, dense_format);
            Multiverso::AddAggregatorTable(kWeightAdaGradTable, feature_group,
                col_size, float_type, dense_format);

        }
        
        static void ConfigTable()
        {
            multiverso::Format dense_format = multiverso::Format::Dense;
            //multiverso::Format sparse_format = multiverso::Format::Sparse;
            for (int32_t w = 0; w < Config::num_features / col_size; ++w)
            {
                    Multiverso::SetServerRow(kWeightGradTable,
                            w, dense_format, col_size);
                    Multiverso::SetCacheRow(kWeightAdaGradTable,
                            w, dense_format, col_size);
            }
        }
    };

    } // namespace ftrl
} // namespace multiverso


int main(int argc, char** argv)
{
    multiverso::ftrl::Ftrl_Lr::Run(argc, argv);
    return 0;
}
