"""Line B -- the same estimation and control loop driven through a KUKA iiwa 14.

Swaps oracle hinge state for proprioceptive FK and a simulated wrist F/T sensor,
which is what makes the estimate hardware-realizable. Imports the estimator and
controller from ``baseline`` unchanged.
"""
