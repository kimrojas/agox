Observers
================================

The AGOX code relies on an observer-pattern coding scheme to enable the code to be adaptive to new ideas. In fact the 
main function that a normal run is started by just starts an iterative call to the observers of the AGOX object. 

Observers in general 
--------------------

An attempt at illustrating observers is shown in the figure below. In the top row two subprograms are called sequentially, 
in the second row we have a similar situation, however the second program calls an observer. In this example there is not 
really a difference between this and sequentially calling the three programs :math:`P_1, \ P_2, \ O_1`. The key difference 
is that with observers is that :math:`O_1` is not hardcoded to follow :math:`P_2` so the code skeleton enables both 
overall programs and in fact it allows any sequence of subprograms. 

.. figure:: images/observer_figure.png

Observers to AGOX
-----------------
The 'highest level' of observers in the AGOX code are those to that are observers to the AGOX object. 
The main loop, the one that runs over episodes, of the search function is in the AGOX class of main_agox.py written 
as such

.. literalinclude:: ../../agox/main.py
    :language: python
    :lines: 91-99

The output of a run will include a description of what is actually going on in this loop that looks something like this::

    |=================================== Observers ===================================|
    |  Order 1 - Name: SamplerKMeans.setup                                            |
    |  Order 2 - Name: TimeDependantCollector.generate_candidates                     |
    |  Order 3.0 - Name: IO_relax.postprocess_candidates                              |
    |  Order 4 - Name: LCBAcquisitor.prioritize_candidates                            |
    |  Order 5 - Name: EnergyEvaluator.evaluate_candidates                            |
    |  Order 6 - Name: Database.store_in_database                                     |
    |=================================================================================|

This tells us about the order in which the observers are called, and we can understand the overall behaviour of the algorithm 
in each iteration. First the sampler is updated, then the collector calls the generators to make new candidates. 
These candidates are relaxed using a postprocessor before a lower-confidence-bound acquisitor picks the most promising one. 
The selected candidate is passed on to the energy evaluator and finally the evaluated the candidate is added to the database. 


We may alter this behaviour from the runscript without any changes to the existing code, as an example we can attach 
an additional observer

.. literalinclude:: ../../agox/modules/helpers/helper_observers/episode_timer.py
    :language: python

This defines a very small class that has three methods that are interesting for us, namely the 'start_timer', 'end_timer' 
and 'easy_attach' functions. We can use this class like this 

.. code-block:: python 

    agox = AGOX(environment=environment, db=database, collector=collector, sampler=sampler, 
            acquisitor=acquisitor, seed=run_idx)

    et = EpisodeTimer()
    et.easy_attach(agox)

    agox.run(N_episodes=NUM_EPISODES)

The description that the program outputs will now look like this::

    ========================= Observers =========================
    Order -1 - Name: start_timer
    Order 1 - Name: TimeDependantCollector_make_candidates
    Order 2 - Name: LCBAcquisitor.acquire_next_candidate
    Order 5 - Name: finish_timer
    =============================================================

And at the end of each episode it will print something like this:: 

    Episode time: 64.62845606356859

So we have added functionality without making any changes to any of the already defined modules of the code, in this case 
it is a somewhat trivial addition, but the underlying principle is very powerful!


Observers to the Database
-------------------------


