from setuptools import setup

setup(name='BiasedRandomWalks',
      version='0.0.4',
      description='Provides classes around biased random walks on networks.',
      url='https://www.github.com/benmaier/BiasedRandomWalks',
      author='Benjamin F. Maier',
      author_email='bfmaier@physik.hu-berlin.de',
      license='MIT',
      packages=['BiasedRandomWalks'],
      install_requires=[
          'numpy>=1.14',
          'networkx>=2',
      ],
      zip_safe=False)
