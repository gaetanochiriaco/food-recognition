import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='foodrecognition',
    version='0.1',
    author='Gaetano Chiriaco',
    author_email='g.chiriaco@campus.unimib.it',
    description='Food Recognition toolkit',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/gaetanochiriaco/food-recognition',
    license='MIT',
    packages=['food_recognition'],
    install_requires=requirements,
)