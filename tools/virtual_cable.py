"""Install the VB-CABLE virtual sound card on Windows.

The realtime voice changer plays into the cable and games / voice chat use the
cable's other end as their microphone. The driver is not redistributed here: it
is fetched from the vendor's own server and installed by its own setup program,
which needs administrator rights (one UAC prompt).
"""

import ctypes
import os
import platform
import shutil
import tempfile
import urllib.request
import zipfile

PACK_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
HOME_URL = "https://vb-audio.com/Cable/"
DOWNLOAD_TIMEOUT = 60
INSTALL_TIMEOUT_MS = 180000

SEE_MASK_NOCLOSEPROCESS = 0x00000040
SW_HIDE = 0
ERROR_CANCELLED = 1223


class InstallError(Exception):
    pass


class Cancelled(InstallError):
    """The user refused the administrator prompt."""


def setup_name():
    """The vendor's setup program for this machine."""
    return "VBCABLE_Setup_x64.exe" if platform.machine().endswith("64") else "VBCABLE_Setup.exe"


def fetch(folder):
    """Download and unpack the driver pack into ``folder``; returns the setup path."""
    archive = os.path.join(folder, "vbcable.zip")
    request = urllib.request.Request(PACK_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            with open(archive, "wb") as out:
                shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as pack:
            pack.extractall(folder)
    except (OSError, zipfile.BadZipFile) as error:
        raise InstallError("download failed: %s" % error)
    setup = os.path.join(folder, setup_name())
    if not os.path.isfile(setup):
        raise InstallError("%s is missing from the driver pack" % setup_name())
    return setup


def run_elevated(program, arguments, folder):
    """Run ``program`` as administrator and wait for it; returns its exit code."""
    from ctypes import wintypes

    class ShellExecuteInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(ShellExecuteInfo)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = program
    info.lpParameters = arguments
    info.lpDirectory = folder
    info.nShow = SW_HIDE
    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        code = ctypes.get_last_error()
        if code == ERROR_CANCELLED:
            raise Cancelled("administrator permission was refused")
        raise InstallError("could not start the installer (error %d)" % code)
    if not info.hProcess:
        return 0
    try:
        kernel32.WaitForSingleObject(info.hProcess, INSTALL_TIMEOUT_MS)
        exit_code = wintypes.DWORD(0)
        kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code))
        return exit_code.value
    finally:
        kernel32.CloseHandle(info.hProcess)


def install():
    """Download VB-CABLE and run its installer silently (-i -h).

    Raises InstallError. Success of the driver install itself is judged by the
    caller: the cable shows up in the device list.
    """
    folder = tempfile.mkdtemp(prefix="vbcable_")
    try:
        setup = fetch(folder)
        run_elevated(setup, "-i -h", folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
