## Simplified MINC2 API for C,Python and LUA
### More interfaces to come

## Goal
To provide access to most functionality of MINC2 file format with minimal effort, with consistent interface



## Installing

### Common Requirements
 * MINC2 library with headers, either by itself or as part of minc-toolkit or minc-toolkit-v2, see http://bic-mni.github.io/
 

### C
 * Requirements: cmake
 * Installation:
    ```
    mkdir build
    cmake .. 
    make 
    ```
 * If location of minc2 library is not found:
    ```
    cmake -DLIBMINC_DIR:PATH=<location of libminc>
    ```
 
### LUA
 * Requirements: torch ( http://torch.ch/ )
 * Installation:
    ```
    cd lua
    luarocks make
    ```
  * If location of minc2 library is not found:
    ```
    cd build.luarocks/
    cmake -DLIBMINC_DIR:PATH=<location of libminc>
    cd ..
    luarocks make
    ```
    
### Python
 * Requirements: cffi, numpy, six
 * Installation:
    ```
    python setup.py build
    python setup.py install 
    ```
 * If libminc is not found: edit `python/minc2/build_minc2-simple.py` and set `minc_prefix`

