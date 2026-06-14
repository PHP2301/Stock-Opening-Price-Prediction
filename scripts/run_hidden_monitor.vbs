Set objFSO = CreateObject("Scripting.FileSystemObject")
strPath = objFSO.GetParentFolderName(WScript.ScriptPosition)
Set WshShell = CreateObject("WScript.Shell")
' Chạy file run_auto_monitor.bat ở chế độ ẩn (window style = 0) và không chặn luồng (wait on return = False)
WshShell.Run "cmd.exe /c """ & strPath & "\run_auto_monitor.bat""", 0, False
