import setuptools


setuptools.setup(
    name="uncertainty_library",
    python_requires='>= 3.8',
    packages=["uncertainty_library"],
    install_requires=[
        'tensorflow == 2.5.0',
        'tqdm',
        'numpy',
        'pandas',
        'matplotlib',
        'tensorflow_probability',
        'scikit-image',
        'scikit-learn',
    ],
)
