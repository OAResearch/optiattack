Installation
============

Requirements
-----------

* Python 3.9 or higher
* pip (Python package installer)

Installation Steps
----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/oaresearch/optiattack.git
      cd optiattack

2. Create a virtual environment (recommended):

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the package:

   .. code-block:: bash

      pip install -e .

Verification
-----------

To verify the installation, run:

.. code-block:: python

   import optiattack
   print(optiattack.__version__) 