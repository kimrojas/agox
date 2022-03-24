Pt14-Au (EMT): Analysis
==================================

Having restarted the script a number of times leaves us with database files that can be analyzed. A quick overview of both 
the performance of the search and the results can be obtained using the batch_analysis.py program contained as part of AGOX. 

.. code-block:: console

   python $AGOX_DIR/utils/batch_analysis.py -d databases/ -DE 0.25

Where databases/ is a folder that contains all of the database (.db) files produced by a particular restart of the search. 
With the files produced by the previous example the output looks lke so: 

.. figure:: batch_analysis_example.png

The plot on the left shows the best structure found by the search, the arrow keys can be used to view the best structure 
found by each restart. The center plot shows the success curve, which counts the number of restarts that have found a structure 
that is within 0.25 eV (as specified with the -DE 0.25 argument in the command line) of the best structure found. Finally 
on the right a histogram of the best structures found by each restart is displayed. 

Often we may want to to perform a more detailed inspection of the structures found by the search, in which case it is 
useful to extract all of them into a single ASE trajectory file, which can be done like so: 

.. literalinclude:: extract.py 
    :language: python

Which will produce a trajectory file containing all 5000 structures the 10 restarts have investigated during the searches. 
