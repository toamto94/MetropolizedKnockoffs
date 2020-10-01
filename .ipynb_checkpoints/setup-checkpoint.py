from distutils.core import setup

setup(name='MetropolizedKnockoffs',
      version='0.0.2',
      description='Knockoff sampler for multivariate Gaussians and Ising distributions',
      author='Tom Mueller',
      author_email='tom_mueller94@gmx.de',
      url='https://github.com/toamto94/MetropolizedKnockoffs',
      packages=['MetropolizedKnockoffs'],
      install_requires=['pandas==1.0.1', 'numpy==1.18.1']
     )
