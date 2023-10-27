from setuptools import setup, find_packages

setup(
    name="aneo_forecasting",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True
)
