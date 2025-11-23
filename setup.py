from setuptools import setup, find_packages

setup(
    name="real-estate-ai-finder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pandas>=2.0.0",
        "mlflow>=2.9.0",
        "prefect>=2.14.0",
    ],
)
