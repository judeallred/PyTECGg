# Installation

`PyTECGg` features a performance-critical core written in Rust and a high-level API in Python. Pre-compiled binaries (wheels) for most common platforms are provided.

## 📦 From PyPI (recommended)

For most users, the simplest way is to install the pre-built package from [PyPI](https://pypi.org/project/pytecgg/):

```shell
pip install pytecgg
```

This will also install all required Python dependencies automatically.

## 🛠️ From source distribution

If you want to compile the package from source (e.g., to benefit from specific CPU optimizations or for inspection), you can force `pip` to use the source distribution:

```shell
pip install pytecgg --no-binary :all:
```

!!! warning "Note"
    Building from source requires a working Rust toolchain (`rustc`, `cargo`). 
    You can install it via [rustup.rs](https://rustup.rs/).

## 👩🏻‍💻 For development

If you want to contribute to the project or modify the source code, a development installation with [`maturin`](https://www.maturin.rs/) is recommended.

1. **Prerequisites**

    Make sure you have a working Rust toolchain, or get it via [rustup.rs](https://rustup.rs/).

2. **Clone the repository**
    
    ```shell
    git clone https://github.com/viventriglia/PyTECGg.git
    cd PyTECGg
    ```

3. **Install and build**
    
    ```shell
    maturin develop
    ```

    `maturin` will compile the Rust core and link it to your environment. Any Python change will be reflected immediately, while Rust changes will require re-running `maturin develop` to recompile the binary.