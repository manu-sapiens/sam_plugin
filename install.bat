if exist ".\venv" (
    echo Venv directory already exists. Aborting.
    exit /b
)

python -m venv venv
call .\venv\Scripts\activate.bat
call .\update.bat