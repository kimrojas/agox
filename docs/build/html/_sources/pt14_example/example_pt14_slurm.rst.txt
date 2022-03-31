Pt14-Au (EMT): Submission
====================================

If running the search on a slurm based cluster the script below might be useful. Generally 
the search is restarted a number of times, here 10, both to improve the accuracy of the search 
and to gather statistics about its performance. 

.. literalinclude:: job_array.sh

The script create a folder on the scratch directory of the compute node, moves the runscript 
their and executes it. The '-i' argument supplies the array ID of the job which is used both for 
naming naming the database and as the seed for numpy's random generator which makes reproduction of 
runs possible. 

When the searches have finished we will have ten folders each containing their own database (.db) file. 

In the next section we will look at ways of analyzing the search using these database files. 

