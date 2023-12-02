# 一些实际使用
## 一个最简单的实例
```cmake
cmake_minimum_required(VERSION 2.8)
project(test)
add_executable(test test.cpp)
```

## 和eigen使用
```cmake
cmake_minimum_required(VERSION 2.8)
project(test)
include_directories("/usr/include/eigen3")
add_executable(test test.cpp)
```

# Pangolin的使用
```cmake
cmake_minimum_required( VERSION 2.8 )
project( visualizeGeometry )

find_package( Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )
add_executable( visualizeGeometry visualizeGeometry.cpp )
target_link_libraries( visualizeGeometry ${Pangolin_LIBRARIES} )

```

