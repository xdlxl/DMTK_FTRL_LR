#include "dsample.h"

#include <multiverso/row.h>
#include <multiverso/log.h>

namespace multiverso { namespace ftrl
{

     FtrlSample::FtrlSample(){
	 
	 
	 }
     bool FtrlSample::init(char* sample_str)
     {	
     	if(sample_str == NULL)
    		return false;

     	char *endptr, *ptr;

		char *cl = strtok_r(sample_str, " \t", &ptr);
		if (cl == NULL) return false;

		float click = strtof(cl, &endptr);
		char *im = strtok_r(NULL, " \t\n", &ptr);
		if (im == NULL) return false;
		float impre = strtof(im, &endptr);
		//Log::Info("click %f impre %f\n",click,impre);

	    label_ = click;
		bool error_found = false;

	    while (1) {
	        char *idx = strtok_r(NULL, " \t", &ptr);
	        char *val = strtok_r(NULL, " \t", &ptr);
	        if (val == NULL) break;

	        int32_t k = (int32_t) strtof(idx, &endptr);
	        k++;
	        if (endptr == idx || *endptr != '\0' || static_cast<int>(k) < 0) {
	            error_found = true;
	        }

	        float v = strtof(val, &endptr);
	        if (endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
	            error_found = true;
	        }
	        if (!error_found) {
	            vals.push_back(v);
	            feat_index.push_back(k);
	        }
	    }


		if (!error_found)
			return true;
		return false;
     }
    int32_t FtrlSample::feature_index(int32_t index) const
    {
        //if (index >= vals.size())
         //   return 0.;
        return feat_index[index];
    }
    float FtrlSample::feature_val(int32_t index) const
    {
       // if (index >= vals.size())
       //     return 0.;
        return vals[index];
    }
    bool FtrlSample::valid()
    {
        return is_valid;
    }
    float FtrlSample::label() const
    {
        return label_;
    }

} // namespace ftrl
} // namespace multiverso
