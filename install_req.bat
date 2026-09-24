@echo off
setlocal
cd /d "%~dp0"

:: Nome da pasta do ambiente virtual a criar/usar
set "VENV_NAME=venv"

echo ======================================================
echo   Instalar Dependencias do Projeto
echo ======================================================
echo.

:: 1. Verificar se o Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao foi encontrado no sistema.
    echo Por favor instala o Python e tenta novamente.
    pause
    exit /b 1
)

:: 2. Verificar se o requirements.txt existe
if not exist "requirements.txt" (
    echo [ERRO] Ficheiro requirements.txt nao encontrado!
    pause
    exit /b 1
)

:: 3. Criar ambiente virtual se nao existir
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo A criar ambiente virtual (%VENV_NAME%)...
    python -m venv %VENV_NAME%
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual.
        pause
        exit /b 1
    )
)

:: 4. Ativar o ambiente virtual
echo A ativar ambiente virtual...
call "%VENV_NAME%\Scripts\activate.bat"

:: 5. Atualizar o pip e instalar as dependencias
echo.
echo A atualizar o pip...
python -m pip install --upgrade pip >nul

echo.
echo A instalar dependencias a partir do requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERRO] Ocorreu um problema ao instalar algumas dependencias.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo   Instalacao concluida com sucesso!
echo   O ambiente virtual foi configurado em "%VENV_NAME%".
echo ======================================================
echo.
pause
endlocal