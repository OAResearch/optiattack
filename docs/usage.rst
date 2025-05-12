Usage Guide
===========

Basic Usage
----------

Here's a simple example of how to use OptiAttack:

.. code-block:: python

    from optiattack import OptiAttack
    
    # Initialize the attack
    attack = OptiAttack()
    
    # Configure attack parameters
    attack.configure(
        target_model="your_model",
        attack_type="optimization",
        parameters={
            "max_iterations": 100,
            "learning_rate": 0.01
        }
    )
    
    # Run the attack
    results = attack.run()

Advanced Usage
-------------

For more advanced usage, you can customize various parameters:

.. code-block:: python

    # Advanced configuration
    attack.configure(
        target_model="your_model",
        attack_type="optimization",
        parameters={
            "max_iterations": 1000,
            "learning_rate": 0.001,
            "constraints": {
                "epsilon": 0.1,
                "norm": "inf"
            }
        }
    )

Configuration Options
--------------------

The following configuration options are available:

* ``target_model``: The model to attack
* ``attack_type``: Type of attack to perform
* ``parameters``: Dictionary of attack-specific parameters
* ``constraints``: Dictionary of attack constraints

For more detailed information about configuration options, see the :doc:`api` documentation. 