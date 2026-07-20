#define AppVersion GetEnv("WATER_DESKTOP_VERSION")
#if AppVersion == ""
  #define AppVersion "1.0.5"
#endif
#define DistDir GetEnv("WATER_DESKTOP_DIST_DIR")
#if DistDir == ""
  #define DistDir "F:\\code\\dist\\水体反应管检测系统"
#endif
#define OutputDir GetEnv("WATER_WINDOWS_OUTPUT_DIR")
#if OutputDir == ""
  #define OutputDir "F:\\code\\正式发布准备-v1.0.5\\Windows"
#endif

[Setup]
AppId={{B2F7E04A-2BCE-4B0C-A4A4-7A2DE1B93D1D}
AppName=水体反应管检测系统
AppVersion={#AppVersion}
AppPublisher=水体反应管检测系统
DefaultDirName={localappdata}\Programs\WaterReactionLab
DefaultGroupName=水体反应管检测系统
OutputDir={#OutputDir}
OutputBaseFilename=water-detection-desktop-v{#AppVersion}-setup
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
Uninstallable=yes
SetupLogging=yes

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\水体反应管检测系统"; Filename: "{app}\水体反应管检测系统.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\水体反应管检测系统"; Filename: "{app}\水体反应管检测系统.exe"; WorkingDir: "{app}"

[Run]
Filename: "{app}\水体反应管检测系统.exe"; Description: "启动水体反应管检测系统"; Flags: postinstall nowait skipifsilent
