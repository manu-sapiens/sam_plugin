@echo off
set url=%1
set filename=%2
set directory=%3

if not exist "%directory%" (
    mkdir "%directory%"
)

pushd "%directory%"

if not exist "%filename%" (
    curl -LJO curl -LJO "%url%/%filename%"
)

popd
