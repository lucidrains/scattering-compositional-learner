from setuptools import setup, find_packages

setup(
  name = 'scattering-transform',
  packages = find_packages(),
  version = '0.0.7',
  license='MIT',
  description = 'Scattering Transform module from the paper Scattering Compositional Learner',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/scattering-compositional-learner',
  keywords = ['artificial intelligence', 'deep learning', 'reasoning'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)