import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

# Get the required packages
with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="dice_ml",
    version="0.2",
    author="Ramaravind Mothilal, Amit Sharma, Chenhao Tan",
    author_email="raam.arvind93@gmail.com",
    description="Generates Diverse Counterfactual Explanations for machine learning models",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/microsoft/DiCE",
    packages=setuptools.find_packages(), #setuptools.find_packages(), ['dice_ml', 'dice_ml.data_interfaces', 'dice_ml.dice_interfaces', 'dice_ml.model_interfaces', 'dice_ml.utils']
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=install_requires,
    include_package_data=True,
    package_data={
        # If any package contains *.h5 files, include them:
        '': ['*.h5']
    }
)
