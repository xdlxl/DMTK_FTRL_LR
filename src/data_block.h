/*!
 * \file data_block.h
 * \brief Defines the training data block
 */

#ifndef ftrl_DATA_BLOCK_H_
#define ftrl_DATA_BLOCK_H_

#include "common.h"
//#include "dsample.h"
#include <multiverso/multiverso.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

namespace multiverso { namespace ftrl
{
    class FtrlSample;
    /*!
     * \brief DataBlock is the an unit of the training dataset, 
     *  it correspond to a data block file in disk. 
     */
 
    class DataBlock : public DataBlockBase
    {
    public:
        DataBlock();
        ~DataBlock();
        /*! \brief Reads a block of data into data block from disk */
        void Read(std::string file_name);

        /*! \brief Gets the size (number of samples) of data block */
        int32_t Size() const;
        /*!
         * \brief Gets one document
         * \param index index of document
         * \return pointer to document
         */
        FtrlSample* GetOneSample(int32_t index);
        std::unordered_set<int32_t> feature_index_set;
		inline bool HasLoad() const ;
		inline int32_t iteration() const ;
    	inline void set_iteration(int32_t iteration);
		int32_t iteration_;
	

    private:
        std::vector<std::shared_ptr<FtrlSample>> samples_;
        /*! \brief number of document in this block */
        char *buf;
        int32_t num_samples_;
         //for request parameter
        void ParseSample();
		bool has_read_;
        // No copying allowed
        DataBlock(const DataBlock&);
        void operator=(const DataBlock&);
    };


    // -- inline functions definition area --------------------------------- //

    inline bool DataBlock::HasLoad() const { return has_read_; }
    inline FtrlSample* DataBlock::GetOneSample(int32_t index)
    { 
        return samples_[index].get(); 
    }
    inline void DataBlock::set_iteration(int32_t iteration)
    { 
        iteration_ = iteration; 
    }
    inline int32_t DataBlock::iteration() const
    { 
        return iteration_ ;
    }



    // -- inline functions definition area --------------------------------- //

} // namespace ftrl
} // namespace multiverso

#endif // ftrl_DATA_BLOCK_H_
