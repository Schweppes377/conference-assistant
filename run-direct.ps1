# Run the project's Python without proxy settings inherited from Happ/WinINET.
# All environment changes are process-local and restored when Python exits.

$ErrorActionPreference = "Stop"

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    Write-Error "Project Python was not found at: $pythonPath"
    exit 1
}

$proxyVariables = @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy"
)

# Windows environment variable names are case-insensitive. Save canonical
# values so the caller's PowerShell environment can be restored afterwards.
$savedVariables = @{}
foreach ($name in @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "CONFERENCE_ASSISTANT_DIRECT",
    "PYTHONPATH"
)) {
    $savedVariables[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

$pythonExitCode = 1
try {
    foreach ($name in $proxyVariables) {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }

    # Without this, Python's urllib falls back to the Windows WinINET proxy and
    # exposes Happ as socks4://127.0.0.1:10808. The wildcard bypass is inherited
    # only by the child Python process.
    [Environment]::SetEnvironmentVariable("NO_PROXY", "*", "Process")
    [Environment]::SetEnvironmentVariable(
        "CONFERENCE_ASSISTANT_DIRECT",
        "1",
        "Process"
    )
    $existingPythonPath = $savedVariables["PYTHONPATH"]
    $directPythonPath = $PSScriptRoot
    if (-not [string]::IsNullOrWhiteSpace($existingPythonPath)) {
        $directPythonPath += [IO.Path]::PathSeparator + $existingPythonPath
    }
    [Environment]::SetEnvironmentVariable(
        "PYTHONPATH",
        $directPythonPath,
        "Process"
    )

    & $pythonPath @args
    $pythonExitCode = $LASTEXITCODE
}
finally {
    foreach ($entry in $savedVariables.GetEnumerator()) {
        if ($null -eq $entry.Value) {
            Remove-Item -LiteralPath "Env:$($entry.Key)" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable(
                $entry.Key,
                $entry.Value,
                "Process"
            )
        }
    }
}

exit $pythonExitCode
