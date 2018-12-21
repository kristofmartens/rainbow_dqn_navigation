from setuptools import setup

setup(
    name='rainbow_dqn_navigation',
    version='0.0.1',
    packages=['rainbow_dqn_agent', 'banana'],
    package_dir={
        'rainbow_dqn_agent': 'src/rainbow_dqn_agent',
        'banana': 'src/banana'
    },
    package_data={
        'banana': ['data/*']
    },

    install_requires=['torch', 'numpy', 'unityagents', 'gym'],

    author="Kristof Martens",
    author_email='kristof.martens@brightml.org',
    licence='MIT',
    keywords="rainbow dqn project nanodegree navigation",
    url="https://github.com/kristofmartens/rainbow_dqn_navigation"
)
