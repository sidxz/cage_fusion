from setuptools import setup, find_packages

setup(
    name='cage_fusion',
    version='0.2.0',
    packages=find_packages(),
    package_data={"cage_fusion": ["py.typed"]},
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'h5py',
        'tqdm',
        'joblib',
        'transformers',
        'huggingface_hub',
        'chemprop',
        'rdkit'
    ],
)
