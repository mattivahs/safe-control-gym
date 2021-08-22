from setuptools import setup

setup(name='safe_control_gym',
    version='0.5.0',
    install_requires=[
        'matplotlib', 
        'scikit-optimize', 
        'munch', 
        'pyyaml', 
        'imageio', 
        'dict-deep', 
        'gym', 
        'torch', 
        'tensorboard', 
        'casadi', 
        'pybullet', 
        'gpytorch', 
        'cvxpy', 
        'pytope', 
        'Mosek']
)