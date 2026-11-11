from setuptools import setup, find_packages

setup(
    name="echopulse-audio",
    version="1.0.0",
    description="Immediate Audio Pattern Recognition & Real-Time Spectral Analysis System",
    author="AI Audio Developer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sounddevice>=0.4.6",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "rich>=13.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.30.0",
    ],
    entry_points={
        "console_scripts": [
            "echopulse=apps.terminal_ui:main",
        ],
    },
)

