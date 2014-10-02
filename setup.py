from distutils.core import setup

setup(
    name='XorReduction',
    version='0.95a',

    packages=[
        'XorReduction',
    ],

    package_data={
        'XorReduction': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

