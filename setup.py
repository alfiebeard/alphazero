from setuptools import setup, find_packages

setup(
    name="alphazero",
    version="0.1.0",
    author="Alfie Beard",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    descriptions="A python package implementing AlphaZero for self play reinforcement learning with some practical examples.",
    install_requires=[
        "tictactoe_gym",
        "numpy==1.23.2",
        "matplotlib==3.5.2",
        "tensorflow==2.9.2; sys_platform != 'darwin' or platform_machine != 'arm64'",
        "tensorflow-macos==2.9.2; sys_platform == 'darwin' and platform_machine == 'arm64'"
    ]
)
