## 一些变量的设置
### CMAKE_BUILD_TYPE
```cmake
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
# 在 "Release" 构建类型下，通常会启用编译器优化和禁用调试信息，以提高程序的运行速度和减小可执行文件的大小。
```

### CMAKE_CXX_FLAGS 
CMAKE_CXX_FLAGS 是一个预定义的变量，用于指定 C++ 编译器的编译选项。通过使用 set() 命令来设置 CMAKE_CXX_FLAGS 的值。
```cmake
set(CMAKE_CXX_FLAGS "-O3")
# -O3 是一个常见的编译选项，表示启用最高级别的优化。它告诉编译器进行更多的优化，以提高生成的机器代码的执行速度和性能。
set(CMAKE_CXX_FLAGS "-std=c++11")
# -std=c++11 是一个编译选项，表示使用 C++11 标准进行编译。
```