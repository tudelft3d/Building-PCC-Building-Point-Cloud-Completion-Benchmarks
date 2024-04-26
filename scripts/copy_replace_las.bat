@echo off
setlocal enabledelayedexpansion

set "source_folder=D:\3DBAG_PCC\Rotterdam\pcl_buildings\AHN4\Waalhaven-Eemhaven"
set "destination_folder=C:\data\data\Rotterdam_processed\las_0\ahn4"
set "file_extension=.las"

for %%F in ("%source_folder%\*%file_extension%") do (
    set "file_name=%%~nF%file_extension%"
    if exist "%destination_folder%\!file_name!" (
        copy /Y "%%F" "%destination_folder%\!file_name!"
    )
)

endlocal