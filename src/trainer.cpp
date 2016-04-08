#include "trainer.h"

#include "common.h"
#include "dsample.h"
#include "data_block.h"

#include <multiverso/barrier.h>
#include <multiverso/stop_watch.h>
#include <multiverso/log.h>

namespace multiverso { namespace ftrl
{

	static int32_t col_size = 10;
    //Trainer::Trainer(multiverso::Barrier *barrier){
    //    barrier_ = barrier;
    //}
    void Trainer::TrainIteration(DataBlockBase* data_block)
    {
        StopWatch watch; watch.Start();
        DataBlock* ftrl_data_block =
            reinterpret_cast<DataBlock*>(data_block);

        int32_t sample_size = ftrl_data_block->Size();

        int32_t id = TrainerId();
        int32_t trainer_num = TrainerCount();
        if (id == 0)
        {
            Log::Info("Rank = %d, Iter = %d Sample size = %d\n",
                Multiverso::ProcessRank(), ftrl_data_block->iteration(),sample_size);
        }
     
        // Train with ftrl sampler
        for (int32_t st = id; st < sample_size; st += trainer_num)
        {
            FtrlSample* samp = ftrl_data_block->GetOneSample(st); // return a reference
            int32_t samp_feat_num = samp->Size();
			if (samp_feat_num < 2)
				Log::Info("feature num less %d\n",samp_feat_num);
            float func_val = 0.;
            std::vector<float> w_vec,grad_vec,ada_grad_vec;
            for(int32_t i = 0; i < samp_feat_num;i++)
            {
                 int32_t feat_index  = samp->feature_index(i);
				 int32_t feat_group  = feat_index / col_size;
				 int32_t feat_remain = feat_index % col_size;
                 Row<float>& grad_row = GetRow<float>(kWeightGradTable,feat_group);
                 Row<float>& ada_grad_row = GetRow<float>(kWeightAdaGradTable,feat_group);

                 //get weight to do
				 float grad = 0.,adagrad = 0.;
				 grad_row.At(feat_remain,&grad);ada_grad_row.At(feat_remain,&adagrad);
                 float wi = GetWeight(grad,adagrad);

				 float fea_val = samp->feature_val(i);
                 func_val += wi * fea_val;

                 w_vec.push_back(wi);
                 grad_vec.push_back(fea_val); 
				 ada_grad_vec.push_back(adagrad);
            }

            float grad = -samp->label() + sigmoid(func_val);
            for(int32_t i = 0; i < samp_feat_num;i++){
                    int32_t feat_index = samp->feature_index(i);
                    Update(feat_index,w_vec[i],grad_vec[i] * grad,ada_grad_vec[i]);
             }
        }
        if (TrainerId() == 0)
        {
            Log::Info("Rank = %d, Training Time used: %.2f s \n", 
                Multiverso::ProcessRank(), watch.ElapsedSeconds());
        }

        watch.Restart();
    }
    float Trainer::GetWeight(float grad,float ada_grad){

        float sign = 1.;
        float val = 0.;
        if (grad < 0) {
            sign = -1.;
        }

        if ( sign * grad <= Config::l1_) {
            val = 0.;
        } else {
            val = (sign * Config::l1_ - grad ) / ((Config::beta_ + sqrt(ada_grad)) / Config::alpha_ + Config::l2_);
        }
        return val;
    }
    void Trainer::SaveModel(const std::string model_path){
                FILE* fid = nullptr;
                fid = fopen(model_path.c_str(), "wt");
				float grad = 0.,adagrad = 0.;

                for (int i = 0; i < Config::num_features / col_size; ++i)
                {
                    Row<float>& grad_row = GetRow<float>(kWeightGradTable,i);
                    Row<float>& ada_grad_row = GetRow<float>(kWeightAdaGradTable,i);
                    //get weight to do
					for (int j = 0; j < col_size; ++j)
					{
						grad_row.At(j,&grad);ada_grad_row.At(j,&adagrad);
						float wi = GetWeight(grad,adagrad);
						fprintf(fid, "%lf\n", wi);
					}
                }
                fclose(fid);
    }
    void Trainer::Update(int32_t feat_index,float w,float grad,float ada_grad){
            float grad_square = grad * grad;
            float sigma = (sqrt(ada_grad + grad_square ) - sqrt(ada_grad)) / Config::alpha_;
            float grad_update = grad - sigma * w;
            float ada_grad_update = grad_square;

			int32_t feat_group  = feat_index / col_size;
			int32_t feat_remain = feat_index % col_size;
            Add<float>(kWeightGradTable, feat_group, feat_remain, grad_update);
            Add<float>(kWeightAdaGradTable, feat_group, feat_remain, ada_grad_update);
    }
   
    void ParamLoader::ParseAndRequest(DataBlockBase* data_block)
    {
        DataBlock* ftrl_data_block =
            reinterpret_cast<DataBlock*>(data_block);
        for(auto p : ftrl_data_block->feature_index_set)
        {
		   int32_t feat_group  = p / col_size;
           RequestRow(kWeightGradTable, feat_group);
           RequestRow(kWeightAdaGradTable, feat_group);
        }
       
    }
} // namespace ftrl
} // namespace multiverso
