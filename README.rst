================================================================
Covid19 Self supervised training combined with Transfer learning
================================================================

An optimized synthesis between self-supervised learning and transfer learning for COVID-19 Diagnosis based on CT scan


Installation
============

.. code:: sh

  git clone <this repo>
  cd <this repo>
  pip install -r requirements.txt

* for GPU support make sure that you have Cuda drivers


Run:
^^^^

Basic run:

1. Set the desired parameters in experiment_manager.py
2. To enable experiment tracking with Neptune.ai, connect your API token in the init function in the experiment manager
3. Run `python experiment_manager.py`

Run with Bayesian optimization:

1. Run Bayesian_optimization.py
2. To enable experiment tracking with Neptune.ai, connect your API token in the init function in the experiment manager
