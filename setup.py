import setuptools

# Define the list of dependencies required for the package
install_requires = [
    'numpy>=1.18.0',
    'scipy>=1.4.0',
    'scikit-learn>=0.22.0',
    'tensorflow>=2.0.0',
    'pandas>=1.0.0',
    'h5py>=2.10.0',
]

setuptools.setup(
    name='IMATAC',
    version='1.0.0',
    description='IMATAC is a deep hierarchical network with denoising autoencoder designed for the imputation of high-dimensional sparse scATAC-seq data.',
    license='Apache License, Version 2.0',
    url='https://github.com/lhqxinghun/IMATAC',

    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
)
