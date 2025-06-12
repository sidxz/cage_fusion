from setuptools import setup, find_packages

setup(
    name='cage_fusion',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'h5py',
        'tqdm',
        'joblib',
        'transformers',
        'chemprop',
        'rdkit'
    ],
)
