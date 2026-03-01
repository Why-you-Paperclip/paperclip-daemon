from setuptools import setup, find_packages

setup(
    name="paperclip-daemon",
    version="0.1.0",
    description="Paperclip GPU provider daemon and CLI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.27.0",
        "click>=8.1.7",
        "rich>=13.7.0",
    ],
    extras_require={
        "gpu": [
            "torch>=2.3.0",
            "torchaudio>=2.3.0",
            "transformers>=4.42.0",
            "peft>=0.11.0",
            "datasets>=2.20.0",
            "accelerate>=0.31.0",
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "paperclip=paperclip.cli:cli",
        ],
    },
)
