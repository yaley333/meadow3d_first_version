# Meadow3D — 3D цветочная долина

Проект на C, OpenGL 3.3 Core, GLFW и GLAD. Работает в Windows 10/11 при наличии драйвера OpenGL 3.3+.

## Зависимости
- CMake 3.20+
- компилятор C (Visual Studio Build Tools или MinGW-w64)
- vcpkg
- библиотеки `glfw3` и `glad` из vcpkg

## Сборка в Windows с Visual Studio и vcpkg

1. Установи CMake и vcpkg: https://github.com/microsoft/vcpkg
2. В терминале установи переменную `VCPKG_ROOT`, указывающую на папку vcpkg, например:
   `set VCPKG_ROOT=C:\\dev\\vcpkg`
3. Из папки проекта выполни:

```bat
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

Запуск:

```bat
build\\Release\\Meadow3D.exe
```

Если используешь MinGW, укажи генератор и toolchain подходящим для своей установки.

## Управление
- WASD — ходить
- мышь — смотреть по сторонам
- Shift — быстрее
- F — включить/выключить полноэкранный режим
- Tab — отпустить/захватить курсор
- Esc — отпустить курсор; повторный Esc закрывает окно

Геометрия ландшафта и растительности генерируется программно и отправляется в GPU-буферы. Внешние текстуры не требуются.
