from setuptools import find_packages, setup

setup(
    name="salmon_aihwkit",
    version="0.2.0",  # 업데이트된 버전
    description="Integrated SALMON and AIHWKIT Project",  # 프로젝트 설명 수정
    author="IBM Research",
    author_email="aihwkit@us.ibm.com",
    url="https://github.com/IBM/aihwkit",
    install_requires=[
        'torch>=1.9', 'torchvision', 'scipy', 'requests>=2.25,<3', 'numpy>=1.22',
        'protobuf>=4.21.6', 'lightning', 'hydra-core==1.3.2', 'omegaconf==2.1.0',
        'cmake>=3.18', 'scikit-build>=0.11.1', 'pybind11>=2.6.2',
        'matplotlib>=3.0'  # visualization 의존성 포함
    ],
    extras_require={
        "fitting": ["lmfit"],
        "bert": ["transformers", "evaluate", "datasets", "wandb", "tensorboard"],
    },
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
    python_requires=">=3.7",
    zip_safe=False
)
