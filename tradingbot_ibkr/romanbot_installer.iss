; Inno Setup installer for the PyInstaller onedir RomanBot package.
; scripts/package_desktop.py supplies OUTPUT_DIR and RELEASE_DIR.

#define SourceDir GetEnv("OUTPUT_DIR")
#define ReleaseDir GetEnv("RELEASE_DIR")

#if SourceDir == ""
  #error "OUTPUT_DIR is not set. Run through scripts/package_desktop.py --windows-installer."
#endif

#if ReleaseDir == ""
  #define ReleaseDir "."
#endif

[Setup]
AppId={{73CE74EA-08BA-46FC-91BA-D904154BBAC2}
AppName=RomanBot
AppVersion=0.1.0
AppPublisher=RomanBot
DefaultDirName={autopf}\RomanBot
DefaultGroupName=RomanBot
OutputDir={#ReleaseDir}
OutputBaseFilename=RomanBotInstaller
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\RomanBot.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\RomanBot"; Filename: "{app}\RomanBot.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\RomanBot"; Filename: "{app}\RomanBot.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\RomanBot.exe"; Description: "Launch RomanBot"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
