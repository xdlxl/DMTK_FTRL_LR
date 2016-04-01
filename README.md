# DMTK_FTRL_LR
A trival version of asynchronous distributed FTRL Logstic regression application.
Based on microsoft dmtk framework.
A simple command line example:
mpiexec -machinefile mahchie_list ./bin/ftrl -num_features 7000000 -input_files_list train_files -num_iterations 2            -num_local_workers 4 -max_delay -1 -num_aggregator 1 -alpha 0.001
