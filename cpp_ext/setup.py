from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='memory_ext',
    ext_modules=[
        CppExtension(
            name='memory_ext',
            sources=['memory_ext.cpp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

