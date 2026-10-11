from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list[str]:
    """Read requirements.txt, skipping comments, blank lines and '-e .'"""
    requirements = []
    with open(file_path) as f:
        for line in f:
            line = line.split('#')[0].strip()   # remove inline comments
            if line and line != "-e .":
                requirements.append(line)
    return requirements

setup(
    name='anomaly_discovery',
    packages=find_packages(),     # needs __init__.py in src/ and every subfolder
    version='0.1.0',
    description='A machine learning platform that automatically discovers, flags, and explains unusual patterns in data.',
    author='Amit Kumar',
    license='MIT',
)
