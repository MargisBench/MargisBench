from setuptools import setup, find_packages

#Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='MargisBench',
    version='0.9',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'margis=cli:main',  # Maps 'margis' command to main() in cli.py
        ],
    },
)