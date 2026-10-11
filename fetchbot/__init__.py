"""fetchbot: a language-commanded fetch robot (PyBullet backend).

The package is organized by responsibility. Every hardware- or backend-specific
piece sits behind an abstract base class so the same task logic can drive another
simulator or a real robot later (an Isaac Sim / ROS 2 backend exists only as a
stub).

Subpackages:
    robots      Robot abstraction and concrete platforms.
    data        Sensor streams (camera, lidar, odometry).
    perception  Object detectors (open-vocabulary, color baseline).
    mapping     Semantic map and frontier exploration.
    navigation  Grid A* navigation and path following.
    tasks       Task state machine (Explore -> ... -> Done).
    language    Natural-language command parsing.
    world       Domain randomization and dynamic actors.
    evaluation  Batch evaluation runner and metrics.
    ui          CLI and live GUI front ends.
    export      Video, report and map exporters.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
