# Artificial Ossification

A GhPython algorithm for shape and size optimization that dynamically visualises ossification principles, including osteoblast and osteoclast movement.

![](animation.gif)

# Setup

software requirements:

- Rhino 6 SR35
- Karamba3D 1.3.3

To test the algorithm simply open the .gh file and increase the iteration slider plugged into the main GhPython component.

If you are unfamiliar with the algorithm, please explore the sample files first. The main .gh file might be unstable.

The file "GhPython_dump.py" is not a standalone version. In the current stage it only serves the understanding of the algorithm.

# Usage

Explanation for the input parameters of the main GhPython component:

- beam_divisions 
    - Determines the number of evenly spaced probe points on each beam, visible by the agents.
- agent\_vision\_radius_factor
    - The agent vision range is set in relation to the overall geometry scale, but can be modified by this factor.
- agent\_effect\_radius_factor
    - The agent effect range is set in relation to the overall geometry scale, but can be modified by this factor.
- osteoblast_number
    - self-explainatory
- osteoclast_number
    - self-explainatory
- agent_strength
    - The amount added to a cross-section radius by an agent.
- frame_pt
    - A 2D frame or 3D volume is created between this input point and the origin of coordinates (0,0,0)
- iterations
    - Set the current iteration. Be careful, this can be very computationally intensive.
- voronoi_divisions
    - Determines the number of voronoi cells for the frame specified trough the \[frame_pt\].
- seed
    - Randomize the initial latticework.

# Contribution and Citation

I'm happy for every kind of contribution to the project. If you find this useful for your research, a paper reference will be available here shortly.

*<p align="center">This project is published under [MIT](LICENSE).</p>*
