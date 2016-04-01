/*!
 * \file trainer.h
 * \brief Defines multiverso interface for parameter loading and data training
 */

#ifndef ftrl_TRAINER_H_
#define ftrl_TRAINER_H_

#include <mutex>
#include <algorithm>

#include <multiverso/multiverso.h>
#include <multiverso/barrier.h>

namespace multiverso { namespace ftrl
{
    class DataBlock;
   

    /*! \brief Trainer is responsible for training a data block */
    class Trainer : public TrainerBase
    {
    public:
        /*!
         * \brief Defines Trainning method for a data_block in one iteration
         * \param data_block pointer to data block base
         */
         void TrainIteration(DataBlockBase* data_block) override;
         //multiverso::Barrier *barrier_;
         void SaveModel(const std::string model_path);
	private:
		 inline float sigmoid(float x);
    	 float GetWeight(float grad,float ada_grad);
    	 void Update(int32_t feat_index,float w,float grad,float ada_grad);

    };
	inline float Trainer::sigmoid(float x)
	{
		float max_exp = 50;
		float one = 1.;
		return one / (one + std::exp(std::max(std::min(-x, max_exp), -max_exp) ) );
	}

    /*! 
     * \brief ParamLoader is responsible for parsing a data block and
     *        preload parameters needed by this block
     */
    class ParamLoader : public ParameterLoaderBase
    {
        /*!
         * \brief Parse a data block to record which parameters  is 
         *        needed for training this block
         */
        void ParseAndRequest(DataBlockBase* data_block) override;
    };

} // namespace ftrl
} // namespace multiverso

#endif // ftrl_TRAINER_H_
