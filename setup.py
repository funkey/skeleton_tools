from setuptools import setup

setup(
        name='skeleton_tools',
        version='0.1',
        description='Tools to read/write and evaluate on skeleton graphs.',
        requires=['networkx','scipy'],
        packages=['skeleton_tools'],
)
