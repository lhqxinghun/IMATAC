import setuptools
import os
# Define the list of dependencies required for the package
def get_requirements():
    """Read requirements from requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.startswith(('#', '-'))
            ]
    except FileNotFoundError:
        raise RuntimeError(
            f"requirements.txt not found at {requirements_path}. "
            "Please create it with your project dependencies."
        )
    return requirements

setuptools.setup(
    name='IMATAC',
    version='1.0.0',
    description='IMATAC is a deep hierarchical network with denoising autoencoder designed for the imputation of high-dimensional sparse scATAC-seq data.',
    license='Apache License, Version 2.0',
    url='https://github.com/lhqxinghun/IMATAC',

    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
)
