from setuptools import setup

setup(
    name="numpy-dl",
    version="DEV",
    description="Numpy Deep Learning Framework",
    author="illuin",
    author_email='manuel.fay@gmail.com',
    packages=["numpy_dl"],
    install_requires=[
        "numpy"
    ],
    python_requires=">=3.6,<4.0",
)
