from setuptools import find_packages, setup

setup(name='metacol',
      packages=find_packages(exclude=(
          'tests',
          '*.egg-info',
      )),
      description='A unified meta cost model for Colossal-AI auto-parallel',
      python_requires='>=3.6',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: Apache Software License',
          'Environment :: GPU :: NVIDIA CUDA',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: System :: Distributed Computing',
      ],)