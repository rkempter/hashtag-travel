from setuptools import setup, find_packages

setup(
    name = "InstaCollector",
    version = "0.1",
    packages = find_packages(),
    install_requires = ['flask', 'python-instagram', 'foursquare', 'pandas', 'gunicorn'],

    author = "EPFL - Renato Kempter",
    author_email = "renato.kempter@gmail.com",
    description = "Collect public instagram medias around Paris",
)
