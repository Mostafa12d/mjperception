"""Line A -- the RLS system-identification baseline on the bare door.

``run_door_dynamics_validation`` is the spine of the whole repository: its
``simulate()`` log dict is the data contract that every other estimator, both
classical and learned, consumes. Nothing here imports from ``latent_mechanics``;
the dependency runs strictly the other way.
"""
