from pathlib import Path

from setuptools import find_packages, setup

dir_path = Path(__file__).resolve().parent


def read_requirements_file(filename):
    req_file = dir_path.joinpath(filename)
    with req_file.open('r') as f:
        return [line.strip() for line in f]


packages = find_packages(exclude=[])
pkgs = []
for p in packages:
    if p == 'dexenv' or p.startswith('dexenv.'):
        pkgs.append(p)

setup(
    name='dexenv',
    author='Tao Chen',
    author_email='taochen904@gmail.com',
    license='MIT',
    packages=pkgs,
    install_requires=read_requirements_file('requirements.txt'),
    include_package_data=True,
)
