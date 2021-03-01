import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

# Get the required packages
with open('requirements.txt', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

# Deep learning packages are optional to install
extras = ["deeplearning"]
extras_require = dict()
for e in extras:
    req_file = "requirements-{0}.txt".format(e)
    with open(req_file) as f:
        extras_require[e] = [line.strip() for line in f]

setuptools.setup(
    name="dice_ml",
    version="0.5",
    license="MIT",
    author="Ramaravind Mothilal, Amit Sharma, Chenhao Tan",
    author_email="raam.arvind93@gmail.com",
    description="Generate Diverse Counterfactual Explanations for any machine learning model.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/interpretml/DiCE",
    download_url="https://github.com/interpretml/DiCE/archive/v0.5.tar.gz",
    python_requires='>=3.4',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='machine-learning explanation interpretability counterfactual',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        # If any package contains *.h5 files, include them:
        '': ['*.h5']
    }
)
