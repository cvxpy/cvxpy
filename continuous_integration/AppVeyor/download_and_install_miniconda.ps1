# Download miniconda

$miniconda_url = "https://repo.continuum.io/miniconda/"
$fileurl = $miniconda_url + $env:MINICONDA_FILENAME
$filepath = $pwd.Path + "\" + $env:MINICONDA_FILENAME
$client = new-object System.Net.WebClient
$client.DownloadFile($fileurl,  $filepath)

# Install miniconda

$install_args = "/InstallationType=JustMe /S /RegisterPython=1 /D=" + $env:PYTHON
Write-Host $filepath $install_args
Start-Process -Filepath $filepath -ArgumentList $install_args -Wait -Passthru

# Update conda and call "conda init" to handle path management issues on Windows

$env:PATH = "${env:PYTHON};${env:PYTHON}\Scripts;" + $env:PATH
echo "The PATH environment variable is"
echo $env:PATH
conda update conda -y
conda init
