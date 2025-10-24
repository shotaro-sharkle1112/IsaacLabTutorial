@echo off
setlocal

rem バッチファイルの置き場所を作業ディレクトリに（任意）
cd /d "%~dp0"

echo [1/3] Running Limo-Pendulum-NoNoise-R2-64...
isaac scripts/rsl_rl/train.py --task Limo-Pendulum-NoNoise-R2-64 --max_iterations 3

echo [2/3] Running Limo-Pendulum-NoNoise-R2-32...
isaac scripts/rsl_rl/train.py --task Limo-Pendulum-NoNoise-R2-32 --max_iterations 3

echo [3/3] Running Limo-Pendulum-NoNoise-R2-128...
isaac scripts/rsl_rl/train.py --task Limo-Pendulum-NoNoise-R2-128 --max_iterations 3

echo.
echo Done.
pause
