# DMTK_FTRL_LR
A trival version of asynchronous distributed FTRL Logstic regression application.    
Based on microsoft dmtk framework.  
A simple command line example:  
mpiexec -machinefile mahchie_list ./bin/ftrl -num_features 7000000 -input_files_list train_files -num_iterations 2                -num_local_workers 4 -max_delay -1 -num_aggregator 1 -alpha 0.001    
input_files_list contains per file a line, only support gz format for now.  
in the train file , each line contains a sample, in format  
click \t impre \t feat_index1 \t val1 feat_index2 \t val2 ...  
