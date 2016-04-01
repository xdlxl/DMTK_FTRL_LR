/*!
 * \file SAMPLE.h
 * \brief Defines SAMPLE data structure
 */

#ifndef FtrlSample_H_
#define FtrlSample_H_

#include "common.h"
#include <string.h>
#include <vector>

namespace multiverso
{
    template <typename T> class Row;
}

namespace multiverso { namespace ftrl
{
    /*!
     * \brief SAMPLE presents a SAMPLE. own a vector for feature index and a vector for feature val;
     */
    class FtrlSample
    {
    public:
        /*!
         * \brief Constructs a SAMPLE based on the start and end pointer
         */
        FtrlSample();
        /*! \brief Get the length of the SAMPLE */
        int32_t Size() const;
        int32_t feature_index(int32_t index) const;
        float feature_val(int32_t index) const;
        float label() const;
        bool valid();
        bool init(char* sample_str);
    private:
       std::vector<float> vals;
       std::vector<int32_t> feat_index;
       bool is_valid;
       float label_;

        // No copying allowed
        FtrlSample(const FtrlSample&);
        void operator=(const FtrlSample&);
    };

    // -- inline functions definition area --------------------------------- //
    inline int32_t FtrlSample::Size() const
    {
        return (int32_t)feat_index.size();
    }
    // -- inline functions definition area --------------------------------- //

} // namespace ftrl
} // namespace multiverso

#endif // FtrlSample_H_
