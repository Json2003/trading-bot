; Inno Setup script for RomanBot
[Setup]
AppName=RomanBot
AppVersion=0.1.0
DefaultDirName={pf}\RomanBot
DefaultGroupName=RomanBot
OutputBaseFilename=RomanBotInstaller
Compression=lzma2
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Main single-file EXE
Source: "{#SourceExe}"; DestDir: "{app}"; Flags: ignoreversion

; Optional model_store and datafiles (if present in build folder)
Source: "model_store\*"; DestDir: "{app}\model_store"; Flags: recursesubdirs createallsubdirs
Source: "datafiles\*"; DestDir: "{app}\datafiles"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\RomanBot"; Filename: "{app}\RomanBot.exe"

[Run]
Filename: "{app}\RomanBot.exe"; Description: "Launch RomanBot"; Flags: nowait postinstall skipifsilent

; Preprocessor helper: set SourceExe based on dist output if available
#define SourceExe "dist\\RomanBot.exe"
