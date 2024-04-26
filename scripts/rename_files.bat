@echo off
setlocal enabledelayedexpansion

set "folder_path=C:\data\DenHaag_processed\las_2\ahn3"

for %%F in ("%folder_path%\*_2.las") do (
    set "file=%%~nF"
    set "newname=!file:_2=!"
    ren "%%F" "!newname!.las"
)

endlocal