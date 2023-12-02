# pangolin编译报错
```bash
/usr/local/include/sigslot/signal.hpp:1180:65: error: ‘slots_reference’ was not declared in this scope
 1180 |         cow_copy_type<list_type, Lockable> ref = slots_reference();
```
在CMakeLists.txt文件中加上一下代码即可，其实是 C++ 11 不支持编译，把 C++版本换到 C++14 就可以了。
```cmake
set(CMAKE_CXX_FLAGS "-std=c++14")
```

# sophus编译报错
```bash
/usr/local/include/sophus/common.hpp:36:10: fatal error: fmt/format.h: 没有那个文件或目录
   36 | #include <fmt/format.h>
      |          ^~~~~~~~~~~~~~
compilation terminated.
```
之所以出现该问题是因为原书使用Sophous库时，仅仅需要EIgen一个依赖，而如今版本的Sophous库还需要fmt依赖。
因此，要解决此问题安装该库即可：
```bash
git clone https://github.com/fmtlib/fmt.git
cd fmt
mkdir build
cd build
cmake ..
make
sudo make install
```

# 编译问题
像 Pangolin、Sophus，安装完后他们的头文件在/usr/local/include对应的文件夹下，CMakeLists.txt下可以不用设置 `include_directories( ${Pangolin_INCLUDE_DIRS} )`，在vscode中点击也能跳转到对应的头文件。但eigen得头文件在/usr/include/eigen3，需要在CMakeLists.txt中添加`include_directories( /usr/include/eigen3 )`，才能编译；在c_cpp_properties.json中添加`/usr/include/eigen3`才能跳转。