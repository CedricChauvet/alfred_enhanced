from setuptools import setup, find_packages

setup(
    name='alfred_experiments',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Vos dépendances additionnelles
        'pyyaml',
        'wandb',  # Pour tracking expériences
        'tensorboard',
    ],
    python_requires='>=3.7',
)