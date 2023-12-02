# tips
## 添加头文件路径
1. ctrl+shift+p，搜索【c/c++:编辑配置(JSON)】，会在.vscode里面新增一个c_cpp_properties.json文件。
2. 在文件的includePath里面添加头文件路径。
```json
c_cpp_properties.json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/include/eigen3"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++14",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```