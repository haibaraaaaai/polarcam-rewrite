# PolarCam Rewrite

Fresh architecture focused on responsiveness (GUI on main thread, processing on worker threads).

## Quick start
# Install base
python -m pip install -e .

# Install IDS Python bindings
python -m pip install --no-deps ids-peak ids-peak-ipl

# Make sure the IDS Peak runtime/driver is installed from IDS.