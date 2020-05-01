import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

# Get the required packages
with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="dice_ml",
    version="0.2",
    license="MIT",
    author="Ramaravind Mothilal, Amit Sharma, Chenhao Tan",
    author_email="raam.arvind93@gmail.com",
    description="Generate Diverse Counterfactual Explanations for any machine learning model.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/interpretml/DiCE",
    download_url="https://github.com/interpretml/DiCE/archive/v0.2.tar.gz",
    python_requires='>=3.4',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='machine-learning explanation interpretability counterfactual',
    install_requires=install_requires,
    include_package_data=True,
    package_data={
        # If any package contains *.h5 files, include them:
        '': ['*.h5']
    }
)
