#include "data_block.h"
#include "dsample.h"
#include "common.h"

#include <multiverso/log.h>
#include <zlib.h>
#include <string.h>
#include <fstream>
//#include <memory>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else 
#include <stdio.h>

#endif


namespace multiverso { namespace ftrl
{
	class FtrlSample;
    DataBlock::DataBlock()
        : has_read_(false),num_samples_(0),iteration_(0)
    {
		buf = new char[65536];
		memset(buf,65536,0);
    }
	DataBlock::~DataBlock()
	{
		feature_index_set.clear();	
		delete [] buf;
	}

    void DataBlock::Read(std::string file_name)
    {
        //load samples from gz file or hdfs[to do]
        gzFile gz_file_desc = gzopen(file_name.c_str(), "r");
        if (!gz_file_desc) 
        {
            printf("OpenFile(): open first file %s failed!\n", file_name.c_str());
			has_read_ =false;
            return ;
        }
        int buf_size = 65536; //assumption line length less than 65536,may be wrong
        //char * buf = new char[buf_size];
        while(gzgets(gz_file_desc, buf, buf_size-1) != NULL)
        {
            if (strrchr(buf, '\n') == NULL) 
            {
                printf("error , line char num greater than 65536 \n");
            }
            //size_t len = strlen(buf);
            ParseSample();
			memset(buf,65536,0);
        }
        gzclose(gz_file_desc);
        has_read_ = true;
    }
	void  DataBlock::ParseSample(){

    	//FtrlSample* FtrlSample = new FtrlSample();
		std::shared_ptr<FtrlSample> ftrl_sample( new FtrlSample());
    	if (ftrl_sample.get()->init(buf)){
			samples_.push_back(ftrl_sample);
			num_samples_++;
			for(int i = 0;i < ftrl_sample.get()->Size();i++){
				int32_t idx = ftrl_sample.get()->feature_index(i);
				if (feature_index_set.count(idx) == 0)
					feature_index_set.insert(idx);
			}
    	}
}
int32_t DataBlock::Size() const{
    return num_samples_;
}


   
} // namespace ftrl
} // namespace multiverso
