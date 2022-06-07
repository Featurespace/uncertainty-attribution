import setuptools


setuptools.setup(
    name="uncertainty_library",
    python_requires='>= 3.8',
    packages=["uncertainty_library"],
    install_requires=[
        'tensorflow == 2.5.0',
        'protobuf >= 3.9.2, < 3.20',
        'tqdm',
        'numpy',
        'pandas',
        'matplotlib',
        'tensorflow_probability == 0.13',
        'scikit-image',
        'scikit-learn',
    ],
)
