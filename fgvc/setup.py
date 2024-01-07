from setuptools import setup, find_packages

setup(
    name='cal',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'wandb',
        'numpy',
        'matplotlib',
        'tqdm'
    ],
)
